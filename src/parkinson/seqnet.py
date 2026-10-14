"""Bi-directional Transformer sequence model for Parkinson's motor score prediction.

Team: parKINGson
Members: Damien CATAU, Malo GERVAISE, Esteban GAUBERT, Maximilien STUDER

Treats each patient's visit timeline as a padded sequence and predicts a
normalized residual per visit. The residual is anchored to a medication baseline
(off if observed, else on) so the model corrects known biased measurements rather
than predicting raw motor scores from scratch.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Sinusoidal temporal embedding
# ---------------------------------------------------------------------------


class SinusoidalTemporalEmbedding(nn.Module):
    """Project scalar Δage intervals to a fixed-frequency sinusoidal embedding.

    Uses the standard Transformer frequency schedule:
        ω_k = 1 / 10000^(2k / d_time)
    producing d_time/2 sin and d_time/2 cos pairs concatenated into a
    d_time-dimensional vector per visit.

    Parameters
    ----------
    d_time:
        Embedding dimension. Must be even. Default 16.
    """

    def __init__(self, d_time: int = 16) -> None:
        super().__init__()
        if d_time % 2 != 0:
            raise ValueError(f"d_time must be even, got {d_time}.")
        self.d_time = d_time
        # Pre-compute and register frequency bases as a buffer (not trained).
        half = d_time // 2
        freq = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float32) / half))
        self.register_buffer("freq", freq)  # (half,)

    def forward(self, delta_age: Tensor) -> Tensor:
        """Compute sinusoidal temporal embeddings.

        Parameters
        ----------
        delta_age:
            Shape (B, T) — age interval since the previous visit (0 for t=0).

        Returns
        -------
        Tensor
            Shape (B, T, d_time).
        """
        # delta_age: (B, T) → (B, T, 1)
        t = delta_age.unsqueeze(-1)
        # freq: (half,) → (1, 1, half)
        freq = self.freq.view(1, 1, -1)
        angles = t * freq  # (B, T, half)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, T, d_time)


# ---------------------------------------------------------------------------
# SeqNet
# ---------------------------------------------------------------------------


class SeqNet(nn.Module):
    """Bi-directional Transformer for patient visit sequence regression.

    Architecture:
    1. Continuous features → linear projection (d_cont → d_cont_proj).
    2. Categorical IDs → per-column nn.Embedding → concat → project (d_cat_total → d_cat_proj).
       Note: d_cat_proj = sum of embedding dims (no extra projection needed
       when the concat already matches the allocated budget).
    3. Δage → SinusoidalTemporalEmbedding (d_time).
    4. Token: concat [proj_cont, embed_cat, sin_temp] → (B, T, d_model).
    5. TransformerEncoder (num_layers layers, nhead heads, batch_first=True).
    6. Linear residual head → scalar r̂ per visit → (B, T).
    7. Anchored prediction: ŷ = base + residual_scale × r̂.

    Parameters
    ----------
    d_cont:
        Number of continuous input features per visit.
    cat_vocab_sizes:
        List of vocabulary sizes (including <unk>) for each categorical column.
    cat_embed_dims:
        List of embedding dimensions corresponding to each categorical column.
    d_time:
        Sinusoidal temporal embedding dimension. Default 16.
    d_model:
        Total token dimension fed to the Transformer. Default 128.
    nhead:
        Number of attention heads. Default 4.
    num_layers:
        Number of TransformerEncoder layers. Default 2.
    dim_feedforward:
        Feed-forward sublayer width inside each encoder layer. Default 256.
    dropout:
        Dropout probability. Default 0.1.
    residual_scale:
        Scale factor applied to the predicted residual: ŷ = base + scale × r̂.
        Default 20.0.
    """

    def __init__(
        self,
        d_cont: int,
        cat_vocab_sizes: list[int],
        cat_embed_dims: list[int],
        d_time: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        residual_scale: float = 20.0,
    ) -> None:
        super().__init__()

        if len(cat_vocab_sizes) != len(cat_embed_dims):
            raise ValueError(
                "cat_vocab_sizes and cat_embed_dims must have the same length."
            )

        self.d_cont = d_cont
        self.d_time = d_time
        self.d_model = d_model
        self.residual_scale = residual_scale

        # --- Categorical embeddings ---
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(vocab, dim)
                for vocab, dim in zip(cat_vocab_sizes, cat_embed_dims)
            ]
        )
        d_cat_total = sum(cat_embed_dims)

        # --- Continuous projection ---
        # Allocate d_cont_proj = d_model - d_cat_total - d_time
        d_cont_proj = d_model - d_cat_total - d_time
        if d_cont_proj <= 0:
            raise ValueError(
                f"d_model ({d_model}) too small: d_cat_total={d_cat_total} + "
                f"d_time={d_time} already exceeds it."
            )
        self.cont_proj = nn.Linear(d_cont, d_cont_proj)

        # --- Temporal embedding ---
        self.temporal_emb = SinusoidalTemporalEmbedding(d_time)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Residual prediction head ---
        self.head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialization for projection layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        features: Tensor,
        cat_ids: Tensor,
        delta_age: Tensor,
        attention_mask: Tensor,
        base: Tensor,
    ) -> Tensor:
        """Compute anchored per-visit predictions.

        Parameters
        ----------
        features:
            Continuous features, shape (B, T, d_cont). Already z-score normalized.
        cat_ids:
            Integer category IDs, shape (B, T, n_cat). Each column corresponds
            to one entry in ``cat_vocab_sizes``.
        delta_age:
            Δage intervals, shape (B, T). First visit in sequence = 0.
        attention_mask:
            Boolean mask, shape (B, T). True = valid token, False = padding.
        base:
            Anchoring baseline per visit, shape (B, T). Base = off if observed
            else on. Used in the final anchored prediction formula.

        Returns
        -------
        Tensor
            Predicted MDS-UPDRS scores, shape (B, T). Padding positions are 0.
        """
        _, _, _ = features.shape

        # 1. Continuous projection.
        x_cont = self.cont_proj(features)  # (B, T, d_cont_proj)

        # 2. Categorical embeddings.
        cat_embeds = [emb(cat_ids[..., i]) for i, emb in enumerate(self.embeddings)]
        x_cat = torch.cat(cat_embeds, dim=-1)  # (B, T, d_cat_total)

        # 3. Sinusoidal temporal embedding.
        x_time = self.temporal_emb(delta_age)  # (B, T, d_time)

        # 4. Combine into token embedding.
        x = torch.cat([x_cont, x_cat, x_time], dim=-1)  # (B, T, d_model)

        # 5. TransformerEncoder (src_key_padding_mask: True = ignore).
        # PyTorch convention is inverted from our attention_mask (True=valid).
        pad_mask = ~attention_mask  # (B, T), True = padding position
        x = self.transformer(x, src_key_padding_mask=pad_mask)  # (B, T, d_model)

        # 6. Residual head.
        r = self.head(x).squeeze(-1)  # (B, T)

        # 7. Anchored prediction: ŷ = base + scale × r̂.
        predictions = base + self.residual_scale * r  # (B, T)

        # Zero out padding positions.
        predictions = predictions * attention_mask.float()

        return predictions


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_seqnet(
    d_cont: int,
    cat_vocab_sizes: list[int],
    d_time: int = 16,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    residual_scale: float = 20.0,
) -> SeqNet:
    """Construct a ``SeqNet`` with standard per-column embedding dims.

    Embedding dimensions follow a log-proportional heuristic:
    ``dim = max(4, min(16, round(math.log2(vocab_size + 1) * 2)))``.

    Parameters
    ----------
    d_cont:
        Number of continuous input features.
    cat_vocab_sizes:
        Vocabulary sizes per categorical column.
    d_time:
        Sinusoidal temporal embedding dimension.
    d_model:
        Total Transformer model dimension.
    nhead:
        Number of attention heads.
    num_layers:
        Number of encoder layers.
    dim_feedforward:
        Feed-forward width.
    dropout:
        Dropout probability.
    residual_scale:
        Anchored prediction scale factor.

    Returns
    -------
    SeqNet
    """
    cat_embed_dims = [
        max(4, min(16, round(math.log2(v + 1) * 2))) for v in cat_vocab_sizes
    ]
    return SeqNet(
        d_cont=d_cont,
        cat_vocab_sizes=cat_vocab_sizes,
        cat_embed_dims=cat_embed_dims,
        d_time=d_time,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        residual_scale=residual_scale,
    )
