"""Unit tests for SeqNet and ParkinsonDataset.

Covers:
1. Forward pass output shape and zero-padding guarantee.
2. Attention masking: padding positions do not influence valid token outputs.
3. Stochastic OFF masking: augmented dataset modifies some observed OFF visits.
4. Backward determinism: identical batches with fixed seeds produce identical gradients.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Make src importable from tests/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from parkinson.dataset import (
    CONT_COLS,
    CategoricalEncoder,
    ParkinsonDataset,
    VisitNormalizer,
    collate_visits,
    prepare_dataframe,
)
from parkinson.seqnet import SeqNet, SinusoidalTemporalEmbedding, build_seqnet

# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------

# Fixed hyperparameters for fast tests.
_D_MODEL = 64
_D_TIME = 8
_NHEAD = 2
_NUM_LAYERS = 1
_DIM_FF = 32

# Minimal vocab sizes matching CAT_COLS = [cohort, gene, rater_id].
_CAT_VOCAB_SIZES = [3, 6, 10]
_D_CONT = len(CONT_COLS)


def _make_raw_df(
    n_patients: int = 4, visits_per_patient: tuple[int, ...] = (5, 3, 7, 4)
) -> pd.DataFrame:
    """Construct a minimal raw visits DataFrame with realistic column types.

    The first visit per patient always has an observed OFF score so masking
    tests are never skipped due to unlucky random draws.
    """
    rng = np.random.default_rng(42)
    rows = []
    idx = 0
    for p_idx, n_visits in enumerate(visits_per_patient[:n_patients]):
        for v in range(n_visits):
            # Guarantee at least one observed OFF per patient (first visit).
            force_off = v == 0
            off_val = float(rng.uniform(20, 60)) if (force_off or rng.random() > 0.42) else float("nan")
            on_val = float(rng.uniform(15, 55)) if rng.random() > 0.30 else float("nan")
            rows.append(
                {
                    "Index": idx,
                    "patient_id": f"P{p_idx:03d}",
                    "cohort": rng.choice(["A", "B"]),
                    "sexM": int(rng.choice([0, 1])),
                    "gene": rng.choice(["GBA", "LRRK2", "nan", "SNCA"]),
                    "age_at_diagnosis": 55.0 + p_idx,
                    "age": 60.0 + p_idx + v * 0.5,
                    "ledd": float(rng.uniform(300, 800))
                    if rng.random() > 0.37
                    else float("nan"),
                    "time_since_intake_on": float(rng.uniform(0.5, 4.0))
                    if rng.random() > 0.46
                    else float("nan"),
                    "time_since_intake_off": float(rng.uniform(1.0, 6.0))
                    if rng.random() > 0.79
                    else float("nan"),
                    "rater_id": f"R{rng.integers(1, 10):02d}",
                    "on": on_val,
                    "off": off_val,
                    "target": float(rng.uniform(20, 70)),
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def _make_prepared_df(
    n_patients: int = 4,
) -> tuple[pd.DataFrame, VisitNormalizer, CategoricalEncoder]:
    """Return prepared + normalized + encoded DataFrame plus fitted transformers."""
    df_raw = _make_raw_df(n_patients=n_patients)
    X = df_raw.drop(columns=["target"])
    y = df_raw["target"]

    df = prepare_dataframe(X, y)
    norm = VisitNormalizer()
    norm.fit(df)
    df = norm.transform(df)

    enc = CategoricalEncoder()
    enc.fit(df)
    df = enc.transform(df)

    return df, norm, enc


def _make_batch(n_patients: int = 3) -> dict:
    """Return a collated batch from a small synthetic dataset."""
    df, _norm, _enc = _make_prepared_df(n_patients=n_patients)
    dataset = ParkinsonDataset(df, augment=False)
    items = [dataset[i] for i in range(len(dataset))]
    return collate_visits(items)


def _make_model(cat_vocab_sizes: list[int] | None = None) -> SeqNet:
    """Construct a minimal SeqNet for testing."""
    vocab = cat_vocab_sizes if cat_vocab_sizes is not None else _CAT_VOCAB_SIZES
    return build_seqnet(
        d_cont=_D_CONT,
        cat_vocab_sizes=vocab,
        d_time=_D_TIME,
        d_model=_D_MODEL,
        nhead=_NHEAD,
        num_layers=_NUM_LAYERS,
        dim_feedforward=_DIM_FF,
    )


# ---------------------------------------------------------------------------
# Test 1: Forward pass shape and zero-padding
# ---------------------------------------------------------------------------


def test_seqnet_forward_shape_and_padding():
    """Output shape matches (B, T_max) and padding positions are exactly 0."""
    torch.manual_seed(0)
    batch = _make_batch(n_patients=3)
    model = _make_model()
    model.eval()

    with torch.no_grad():
        preds = model(
            batch["features"],
            batch["cat_ids"],
            batch["delta_age"],
            batch["attention_mask"],
            batch["base"],
        )

    B, T_max = batch["attention_mask"].shape
    assert preds.shape == (B, T_max), f"Expected ({B}, {T_max}), got {preds.shape}"

    # Padding positions (attention_mask == False) must be 0.
    pad_positions = ~batch["attention_mask"]
    if pad_positions.any():
        pad_preds = preds[pad_positions]
        assert (pad_preds == 0.0).all(), f"Padding positions not zero: {pad_preds}"


# ---------------------------------------------------------------------------
# Test 2: Attention mask isolation — padding does not affect valid tokens
# ---------------------------------------------------------------------------


def test_attention_mask_padding_isolation():
    """Changing padding values has zero effect on valid token predictions."""
    torch.manual_seed(1)
    batch = _make_batch(n_patients=4)
    model = _make_model()
    model.eval()

    with torch.no_grad():
        preds_orig = model(
            batch["features"],
            batch["cat_ids"],
            batch["delta_age"],
            batch["attention_mask"],
            batch["base"],
        ).clone()

    # Overwrite padding positions with random noise.
    pad_mask = ~batch["attention_mask"]
    if not pad_mask.any():
        pytest.skip("No padding in this batch — skip isolation test.")

    noisy_features = batch["features"].clone()
    noisy_features[pad_mask] = torch.randn_like(noisy_features[pad_mask]) * 100.0

    with torch.no_grad():
        preds_noisy = model(
            noisy_features,
            batch["cat_ids"],
            batch["delta_age"],
            batch["attention_mask"],
            batch["base"],
        )

    # Valid token predictions must be identical (within float32 precision).
    valid = batch["attention_mask"]
    assert torch.allclose(preds_orig[valid], preds_noisy[valid], atol=1e-5), (
        "Padding noise changed valid token predictions."
    )


# ---------------------------------------------------------------------------
# Test 3: Stochastic OFF masking modifies some observed OFF visits
# ---------------------------------------------------------------------------


def test_off_mask_stochastic():
    """With augment=True and p=1.0, all observed OFF visits are masked."""
    df, _, _ = _make_prepared_df(n_patients=6)

    # Confirm at least one observed-OFF visit exists.
    has_observed_off = (df["off_is_missing"] == 0).any()
    if not has_observed_off:
        pytest.skip("No observed OFF visits in synthetic data — skip masking test.")

    # Use p=1.0 so ALL observed OFF visits are masked.
    dataset_aug = ParkinsonDataset(df, augment=True, mask_prob=1.0)
    dataset_plain = ParkinsonDataset(df, augment=False)

    # Gather off_is_missing from all patients.
    def _collect_off_missing(ds: ParkinsonDataset) -> list[int]:
        flags = []
        for i in range(len(ds)):
            item = ds[i]
            flags.extend(item["off_is_missing_raw"].tolist())
        return flags

    # With augment=False, the flags should match the original data.
    plain_flags = _collect_off_missing(dataset_plain)
    n_observed_off_plain = sum(1 for f in plain_flags if not f)

    # With augment=True + p=1.0, all originally-observed OFF visits should be masked.
    # Since augmentation modifies off_is_missing in the group copy but
    # off_is_missing_raw reflects the POST-augmentation state, verify
    # that augmented dataset has more missing flags than the plain one.
    # (We check the features tensor directly via a single patient.)
    for i in range(len(dataset_aug)):
        item_aug = dataset_aug[i]
        item_plain = dataset_plain[i]
        off_miss_plain = item_plain["off_is_missing_raw"]
        # All positions that were observed (False) in plain should be masked (True) in aug.
        originally_observed = ~off_miss_plain
        if originally_observed.any():
            # Check off_filled feature column index (index 6 in CONT_COLS).
            off_idx = CONT_COLS.index("off_filled")
            # Under p=1.0 masking, off_filled for originally-observed visits should be 0.
            aug_off_vals = item_aug["features"][originally_observed, off_idx]
            assert (aug_off_vals == 0.0).all(), (
                f"Patient {i}: off_filled not zeroed for masked visits: {aug_off_vals}"
            )

    assert n_observed_off_plain > 0, (
        "Expected at least one observed OFF in plain dataset."
    )


# ---------------------------------------------------------------------------
# Test 4: Backward determinism — identical batches → identical gradients
# ---------------------------------------------------------------------------


def test_backward_determinism():
    """Two forward+backward passes on identical inputs produce identical gradients.

    Dropout introduces stochasticity — both passes use the same seed so the
    dropout masks are identical and gradients must match exactly.
    """
    torch.manual_seed(99)
    batch = _make_batch(n_patients=4)
    # Use dropout=0.0 to guarantee deterministic gradients without seed juggling.
    model = build_seqnet(
        d_cont=_D_CONT,
        cat_vocab_sizes=_CAT_VOCAB_SIZES,
        d_time=_D_TIME,
        d_model=_D_MODEL,
        nhead=_NHEAD,
        num_layers=_NUM_LAYERS,
        dim_feedforward=_DIM_FF,
        dropout=0.0,
    )
    model.train()

    criterion = torch.nn.MSELoss()
    valid = batch["attention_mask"] & batch["valid_target"]

    def _one_pass() -> dict[str, torch.Tensor]:
        model.zero_grad()
        preds = model(
            batch["features"],
            batch["cat_ids"],
            batch["delta_age"],
            batch["attention_mask"],
            batch["base"],
        )
        loss = criterion(preds[valid], batch["target"][valid])
        loss.backward()
        return {
            name: param.grad.clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }

    grads_1 = _one_pass()
    grads_2 = _one_pass()

    for name in grads_1:
        assert torch.allclose(grads_1[name], grads_2[name], atol=1e-6), (
            f"Gradient for '{name}' differs between identical passes."
        )


# ---------------------------------------------------------------------------
# Test 5: SinusoidalTemporalEmbedding shape
# ---------------------------------------------------------------------------


def test_sinusoidal_temporal_embedding_shape():
    """SinusoidalTemporalEmbedding produces the expected output shape."""
    emb = SinusoidalTemporalEmbedding(d_time=16)
    delta_age = torch.tensor([[0.0, 0.5, 1.2, 0.0], [0.0, 0.3, 0.0, 0.0]])  # (2, 4)
    out = emb(delta_age)
    assert out.shape == (2, 4, 16), f"Expected (2, 4, 16), got {out.shape}"
    assert not out.isnan().any(), "NaN in sinusoidal embedding output."
