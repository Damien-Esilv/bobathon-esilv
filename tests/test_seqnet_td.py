"""Unit tests for TimeDecaySeqNet and td_loss.

Tests cover:
1. Forward pass produces correct output shape.
2. All predictions are finite (no NaN / inf).
3. Padding positions output zero.
4. td_loss is a finite non-negative scalar.
5. Gradient flows through the time-decay gamma parameter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_MPS_FOR_TESTS", "1")

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from parkinson.seqnet_td import TimeDecaySeqNet, build_td_seqnet, td_loss

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_D_CONT = 12
_CAT_VOCAB = [10, 5, 8]
_B, _T = 3, 6


def _make_model(d_model: int = 64, gru_hidden: int = 32) -> TimeDecaySeqNet:
    """Small model for fast testing."""
    return build_td_seqnet(
        _D_CONT,
        _CAT_VOCAB,
        d_time=8,
        d_model=d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        gru_hidden=gru_hidden,
        dropout=0.0,
        residual_scale=10.0,
    )


def _make_batch(B: int = _B, T: int = _T) -> dict:
    torch.manual_seed(0)
    return {
        "features": torch.randn(B, T, _D_CONT),
        "cat_ids": torch.zeros(B, T, len(_CAT_VOCAB), dtype=torch.long),
        "delta_age": torch.zeros(B, T),
        "attention_mask": torch.ones(B, T, dtype=torch.bool),
        "base": torch.full((B, T), 35.0),
        "age": torch.linspace(60, 65, T).unsqueeze(0).expand(B, -1).clone(),
        "target": torch.full((B, T), 40.0),
        "valid_target": torch.ones(B, T, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Test 1: output shape
# ---------------------------------------------------------------------------


def test_output_shape():
    """Forward pass returns shape (B, T)."""
    model = _make_model()
    batch = _make_batch()
    with torch.no_grad():
        out = model(
            batch["features"],
            batch["cat_ids"],
            batch["delta_age"],
            batch["attention_mask"],
            batch["base"],
            batch["age"],
        )
    assert out.shape == (_B, _T), f"Expected ({_B}, {_T}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: all outputs finite
# ---------------------------------------------------------------------------


def test_outputs_finite():
    """All predictions are finite (no NaN or inf)."""
    model = _make_model()
    batch = _make_batch()
    with torch.no_grad():
        out = model(
            batch["features"],
            batch["cat_ids"],
            batch["delta_age"],
            batch["attention_mask"],
            batch["base"],
            batch["age"],
        )
    assert torch.isfinite(out).all(), "Non-finite values in model output"


# ---------------------------------------------------------------------------
# Test 3: padding positions output zero
# ---------------------------------------------------------------------------


def test_padding_positions_zero():
    """Padding positions (attention_mask=False) produce zero output."""
    model = _make_model()
    B, T = 2, 5
    batch = _make_batch(B=B, T=T)
    # Mask: patient 0 has 3 valid visits, patient 1 has all 5.
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, :3] = True
    mask[1, :5] = True
    batch["attention_mask"] = mask

    with torch.no_grad():
        out = model(
            batch["features"][:B, :T],
            batch["cat_ids"][:B, :T],
            batch["delta_age"][:B, :T],
            mask,
            batch["base"][:B, :T],
            batch["age"][:B, :T],
        )

    # Padding positions of patient 0 (positions 3 and 4) should be 0.
    pad_preds = out[0, 3:]
    assert (pad_preds == 0.0).all(), f"Padding positions are not zero: {pad_preds}"


# ---------------------------------------------------------------------------
# Test 4: td_loss is finite and non-negative
# ---------------------------------------------------------------------------


def test_td_loss_finite_nonneg():
    """td_loss returns a finite, non-negative scalar."""
    model = _make_model()
    batch = _make_batch()
    out = model(
        batch["features"],
        batch["cat_ids"],
        batch["delta_age"],
        batch["attention_mask"],
        batch["base"],
        batch["age"],
    )
    mask = batch["attention_mask"] & batch["valid_target"]
    loss = td_loss(out, batch["target"], mask)
    assert loss.ndim == 0, "Loss must be scalar"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() >= 0.0, f"Loss is negative: {loss.item()}"


# ---------------------------------------------------------------------------
# Test 5: gradient flows through time-decay gamma
# ---------------------------------------------------------------------------


def test_gradient_flows_through_gamma():
    """Backward pass produces nonzero gradients on log_gamma parameters."""
    model = _make_model()
    batch = _make_batch()
    out = model(
        batch["features"],
        batch["cat_ids"],
        batch["delta_age"],
        batch["attention_mask"],
        batch["base"],
        batch["age"],
    )
    mask = batch["attention_mask"] & batch["valid_target"]
    loss = td_loss(out, batch["target"], mask)
    loss.backward()

    # Check at least one TimeDecayAttention layer has nonzero gamma grad.
    has_grad = False
    for layer in model.layers:
        grad = layer.self_attn.log_gamma.grad
        if grad is not None and grad.abs().max().item() > 0:
            has_grad = True
            break
    assert has_grad, "log_gamma has zero / no gradient — time-decay not in graph"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
