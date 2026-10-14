"""Time-decay attention Transformer + Bidirectional GRU for Parkinson's motor score.

Architecture
------------
1. Token embedding (same as SeqNet): continuous projection + categorical embeddings
   + sinusoidal temporal embedding → (B, T, d_model).
2. Time-decay self-attention via a learnable exponential bias added to raw attention
   logits before softmax:
       attn_ij = (q_i · k_j) / sqrt(d_k) - softplus(log_gamma) * |t_i - t_j|
   This soft-penalises attending across large time gaps without hard masking.
3. Two standard TransformerEncoderLayers (pre-norm, batch_first).
4. Bidirectional GRU layer (hidden_size=64, bidirectional → output d=128).
5. Residual prediction head: ŷ_t = base_t + residual_scale * head(h_t).
6. Loss: Huber(delta=4) on visit scores + 0.3 * MSE on consecutive score differences.

The time-decay bias is implemented as a custom MultiheadAttention wrapper that
injects the |Δt| penalty into the attention scores before the softmax.  The rest
of the training infrastructure (dataset, normalizer, categorical encoder,
DataLoader) is identical to experiment 09 / 10.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Time-decay attention layer
# ---------------------------------------------------------------------------


class TimeDecayAttention(nn.Module):
    """Multi-head attention with learned exponential time-decay bias.

    For each head, a scalar log_gamma is learned.  The bias term
        - softplus(log_gamma) * |t_i - t_j|
    is added to the raw attention logits before softmax, penalising
    long-range temporal dependencies without hard masking.

    Parameters
    ----------
    d_model:
        Model dimension (must be divisible by nhead).
    nhead:
        Number of attention heads.
    dropout:
        Dropout on attention weights. Default 0.0.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # One learnable log_gamma per head; initialized so softplus(0) ≈ 0.69.
        self.log_gamma = nn.Parameter(torch.zeros(nhead))
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        age: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute time-decay multi-head self-attention.

        Parameters
        ----------
        x:
            Shape (B, T, d_model).
        age:
            Shape (B, T) — raw (un-normalized) age values for each visit.
        key_padding_mask:
            Shape (B, T) — True where position is a padding token.

        Returns
        -------
        Tensor
            Shape (B, T, d_model).
        """
        B, T, _ = x.shape

        Q = self.q_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)  # (B,H,T,d_k)
        K = self.k_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)

        # Raw attention logits.
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, T, T)

        # Time-decay bias: |t_i - t_j| for each pair.
        # age: (B, T) → (B, 1, T, 1) and (B, 1, 1, T)
        t_i = age.unsqueeze(1).unsqueeze(-1)   # (B, 1, T, 1)
        t_j = age.unsqueeze(1).unsqueeze(-2)   # (B, 1, 1, T)
        dt = (t_i - t_j).abs()                 # (B, 1, T, T), broadcast

        # gamma: (H,) → (1, H, 1, 1)
        gamma = F.softplus(self.log_gamma).view(1, self.nhead, 1, 1)
        scores = scores - gamma * dt           # (B, H, T, T)

        # Mask padding positions.
        if key_padding_mask is not None:
            # (B, T) → (B, 1, 1, T)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Time-decay Transformer encoder layer (pre-norm)
# ---------------------------------------------------------------------------


class TimeDecayEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer with time-decay self-attention.

    Parameters
    ----------
    d_model:
        Model dimension.
    nhead:
        Number of attention heads.
    dim_feedforward:
        Feed-forward hidden width. Default 256.
    dropout:
        Dropout probability. Default 0.1.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = TimeDecayAttention(d_model, nhead, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        age: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        # Pre-norm self-attention.
        x = x + self.self_attn(self.norm1(x), age, key_padding_mask)
        # Pre-norm feed-forward.
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# TimeDecaySeqNet
# ---------------------------------------------------------------------------


class TimeDecaySeqNet(nn.Module):
    """Time-decay Transformer + Bidirectional GRU for visit sequence regression.

    Architecture:
    1.  Continuous features → linear projection.
    2.  Categorical IDs → per-column nn.Embedding → concat.
    3.  Δage → SinusoidalTemporalEmbedding.
    4.  Token: concat [proj_cont, embed_cat, sin_temp] → (B, T, d_model).
    5.  Two TimeDecayEncoderLayers (pre-norm).
    6.  Bidirectional GRU (hidden_size = d_model // 2, so output = d_model).
    7.  Residual head → scalar r̂ → ŷ = base + residual_scale * r̂.

    Parameters
    ----------
    d_cont:
        Number of continuous input features.
    cat_vocab_sizes:
        Vocabulary sizes for each categorical column.
    cat_embed_dims:
        Embedding dimensions per categorical column.
    d_time:
        Sinusoidal temporal embedding dimension. Default 16.
    d_model:
        Total token dimension. Default 128.
    nhead:
        Number of attention heads. Default 4.
    num_layers:
        Number of TimeDecayEncoderLayers. Default 2.
    dim_feedforward:
        Feed-forward width. Default 256.
    gru_hidden:
        GRU hidden size per direction. Output = 2 * gru_hidden. Default 64.
    dropout:
        Dropout probability. Default 0.1.
    residual_scale:
        Scale factor for the anchored prediction. Default 20.0.
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
        gru_hidden: int = 64,
        dropout: float = 0.1,
        residual_scale: float = 20.0,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.residual_scale = residual_scale

        # --- Categorical embeddings ---
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab, dim) for vocab, dim in zip(cat_vocab_sizes, cat_embed_dims)]
        )
        d_cat_total = sum(cat_embed_dims)

        # Allocate continuous projection to fill d_model.
        d_cont_proj = d_model - d_cat_total - d_time
        if d_cont_proj <= 0:
            raise ValueError(
                f"d_model ({d_model}) too small for d_cat_total={d_cat_total} + "
                f"d_time={d_time}."
            )
        self.cont_proj = nn.Linear(d_cont, d_cont_proj)

        # --- Sinusoidal temporal embedding (reuse from seqnet) ---
        from parkinson.seqnet import SinusoidalTemporalEmbedding
        self.temporal_emb = SinusoidalTemporalEmbedding(d_time)

        # --- Time-decay Transformer layers ---
        self.layers = nn.ModuleList([
            TimeDecayEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # --- Bidirectional GRU ---
        # gru_hidden per direction; 2 * gru_hidden = d_model is expected.
        assert 2 * gru_hidden == d_model, (
            f"2 * gru_hidden ({2 * gru_hidden}) must equal d_model ({d_model})."
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        # --- Residual head ---
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(
        self,
        features: Tensor,
        cat_ids: Tensor,
        delta_age: Tensor,
        attention_mask: Tensor,
        base: Tensor,
        age: Tensor,
    ) -> Tensor:
        """Compute anchored per-visit predictions.

        Parameters
        ----------
        features:
            Continuous features, shape (B, T, d_cont). Already z-score normalized.
        cat_ids:
            Integer category IDs, shape (B, T, n_cat).
        delta_age:
            Δage intervals, shape (B, T). First visit = 0.
        attention_mask:
            Boolean mask, shape (B, T). True = valid token.
        base:
            Anchor baseline per visit, shape (B, T).
        age:
            Raw (un-normalized) visit ages, shape (B, T). Used for time-decay bias.

        Returns
        -------
        Tensor
            Predicted scores, shape (B, T). Padding positions are 0.
        """
        # 1. Continuous projection.
        x_cont = self.cont_proj(features)

        # 2. Categorical embeddings.
        cat_embeds = [emb(cat_ids[..., i]) for i, emb in enumerate(self.embeddings)]
        x_cat = torch.cat(cat_embeds, dim=-1)

        # 3. Temporal embedding.
        x_time = self.temporal_emb(delta_age)

        # 4. Token.
        x = torch.cat([x_cont, x_cat, x_time], dim=-1)  # (B, T, d_model)

        # 5. Time-decay Transformer layers.
        pad_mask = ~attention_mask  # True = padding
        for layer in self.layers:
            x = layer(x, age, key_padding_mask=pad_mask)

        # 6. Bidirectional GRU.
        # Pack padded sequence for efficiency.
        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.gru(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            gru_out, batch_first=True, total_length=x.shape[1]
        )  # (B, T, d_model)

        # 7. Residual prediction.
        r = self.head(x).squeeze(-1)  # (B, T)
        predictions = base + self.residual_scale * r
        predictions = predictions * attention_mask.float()

        return predictions


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_td_seqnet(
    d_cont: int,
    cat_vocab_sizes: list[int],
    d_time: int = 16,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    gru_hidden: int = 64,
    dropout: float = 0.1,
    residual_scale: float = 20.0,
) -> TimeDecaySeqNet:
    """Construct a TimeDecaySeqNet with log-proportional embedding dims.

    Parameters
    ----------
    d_cont:
        Number of continuous input features.
    cat_vocab_sizes:
        Vocabulary sizes per categorical column.
    d_time, d_model, nhead, num_layers, dim_feedforward, gru_hidden, dropout,
    residual_scale:
        See ``TimeDecaySeqNet``.

    Returns
    -------
    TimeDecaySeqNet
    """
    cat_embed_dims = [
        max(4, min(16, round(math.log2(v + 1) * 2))) for v in cat_vocab_sizes
    ]
    return TimeDecaySeqNet(
        d_cont=d_cont,
        cat_vocab_sizes=cat_vocab_sizes,
        cat_embed_dims=cat_embed_dims,
        d_time=d_time,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        gru_hidden=gru_hidden,
        dropout=dropout,
        residual_scale=residual_scale,
    )


# ---------------------------------------------------------------------------
# Combined Huber + consecutive-difference loss
# ---------------------------------------------------------------------------


def td_loss(
    preds: Tensor,
    targets: Tensor,
    mask: Tensor,
    huber_delta: float = 4.0,
    diff_weight: float = 0.3,
) -> Tensor:
    """Huber loss on scores + weighted MSE on consecutive score differences.

    Parameters
    ----------
    preds:
        Model predictions, shape (B, T).
    targets:
        True scores, shape (B, T).
    mask:
        Boolean mask (attention_mask & valid_target), shape (B, T).
    huber_delta:
        Huber threshold. Default 4.0.
    diff_weight:
        Weight on the consecutive-difference term. Default 0.3.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    huber = F.huber_loss(preds[mask], targets[mask], delta=huber_delta, reduction="mean")

    # Consecutive difference loss on valid adjacent pairs within each sequence.
    pred_diff = preds[:, 1:] - preds[:, :-1]    # (B, T-1)
    tgt_diff = targets[:, 1:] - targets[:, :-1]  # (B, T-1)
    # Mask: both positions must be valid.
    diff_mask = mask[:, 1:] & mask[:, :-1]

    if diff_mask.any():
        diff_l = F.mse_loss(pred_diff[diff_mask], tgt_diff[diff_mask], reduction="mean")
    else:
        diff_l = torch.tensor(0.0, device=preds.device)

    return huber + diff_weight * diff_l
