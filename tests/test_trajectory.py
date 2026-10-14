"""Unit tests for MonotonicBernsteinTrajectoryProjector.

Tests cover:
1. Output shape matches input.
2. Monotonicity guarantee on a well-spaced multi-visit patient.
3. Single-visit patient passthrough (raw prediction returned unchanged).
4. Results are finite (no NaN / inf leaking from the solver).
5. Monotonicity violation rate helper returns 0.0 on projected outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from parkinson.trajectory import MonotonicBernsteinTrajectoryProjector

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_df(
    n_patients: int = 4,
    visits_per_patient: int | list[int] = 5,
    rng_seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a synthetic (df, raw_preds) pair for testing."""
    rng = np.random.default_rng(rng_seed)
    if isinstance(visits_per_patient, int):
        visits_per_patient = [visits_per_patient] * n_patients

    rows = []
    idx = 0
    for p_idx in range(n_patients):
        n_v = visits_per_patient[p_idx % len(visits_per_patient)]
        for v in range(n_v):
            off_val = float(rng.uniform(20, 60)) if rng.random() > 0.42 else float("nan")
            rows.append(
                {
                    "Index": idx,
                    "patient_id": f"P{p_idx:03d}",
                    "age": 60.0 + p_idx + v * 0.5,
                    "off": off_val,
                    "time_since_intake_off": (
                        float(rng.uniform(1.0, 6.0))
                        if rng.random() > 0.79
                        else float("nan")
                    ),
                }
            )
            idx += 1

    df = pd.DataFrame(rows)
    # Raw predictions: rough ascending trend + noise so Bernstein has
    # something sensible to fit.
    base = 30.0 + 0.5 * df["age"].values
    noise = rng.normal(0, 3.0, size=len(df))
    raw_preds = np.clip(base + noise, 0.0, 132.0)
    return df, raw_preds


# ---------------------------------------------------------------------------
# Test 1: output shape matches input
# ---------------------------------------------------------------------------


def test_output_shape_matches_input():
    """Projector returns an array with the same length as df."""
    df, raw = _make_df(n_patients=4, visits_per_patient=5)
    proj = MonotonicBernsteinTrajectoryProjector()
    out = proj.transform(df, raw)

    assert isinstance(out, np.ndarray), "transform() must return np.ndarray"
    assert out.shape == (len(df),), f"Expected ({len(df)},), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: monotonicity guarantee
# ---------------------------------------------------------------------------


def test_monotonicity_guaranteed():
    """All consecutive within-patient (age-sorted) prediction pairs are non-decreasing."""
    df, raw = _make_df(n_patients=6, visits_per_patient=6)
    proj = MonotonicBernsteinTrajectoryProjector()
    out = proj.transform(df, raw)

    rate = proj.monotonicity_violation_rate(df, out, tol=1e-6)
    assert rate == 0.0, (
        f"Expected 0.0% monotonicity violations, got {rate * 100:.4f}%"
    )


# ---------------------------------------------------------------------------
# Test 3: single-visit patient passthrough
# ---------------------------------------------------------------------------


def test_single_visit_passthrough():
    """A patient with exactly one visit gets their raw prediction back unchanged."""
    # One multi-visit patient + one single-visit patient.
    df_multi, raw_multi = _make_df(n_patients=1, visits_per_patient=5, rng_seed=7)
    single_row = pd.DataFrame(
        [
            {
                "Index": 9999,
                "patient_id": "SINGLE",
                "age": 65.0,
                "off": float("nan"),
                "time_since_intake_off": float("nan"),
            }
        ]
    )
    single_pred = np.array([42.5])

    df_all = pd.concat([df_multi, single_row], ignore_index=True)
    raw_all = np.concatenate([raw_multi, single_pred])

    proj = MonotonicBernsteinTrajectoryProjector()
    out = proj.transform(df_all, raw_all)

    # Locate the single-visit patient's prediction in the output.
    single_mask = df_all["patient_id"] == "SINGLE"
    out_single = out[single_mask.values]

    np.testing.assert_allclose(
        out_single,
        single_pred,
        rtol=1e-9,
        err_msg="Single-visit patient prediction should pass through unchanged",
    )


# ---------------------------------------------------------------------------
# Test 4: no NaN / inf in output
# ---------------------------------------------------------------------------


def test_output_is_finite():
    """Projected predictions are all finite (no NaN or inf)."""
    df, raw = _make_df(n_patients=5, visits_per_patient=[3, 4, 5, 6, 7])
    proj = MonotonicBernsteinTrajectoryProjector()
    out = proj.transform(df, raw)

    assert np.all(np.isfinite(out)), (
        f"Found {(~np.isfinite(out)).sum()} non-finite predictions"
    )


# ---------------------------------------------------------------------------
# Test 5: monotonicity violation rate returns 0 on clean projected output
# ---------------------------------------------------------------------------


def test_violation_rate_zero_on_projected():
    """monotonicity_violation_rate returns 0.0 when given projected predictions."""
    df, raw = _make_df(n_patients=4, visits_per_patient=7, rng_seed=99)
    proj = MonotonicBernsteinTrajectoryProjector()
    projected = proj.transform(df, raw)

    rate = MonotonicBernsteinTrajectoryProjector.monotonicity_violation_rate(
        df, projected
    )
    assert rate == 0.0, f"Violation rate should be 0.0, got {rate}"


# ---------------------------------------------------------------------------
# Test 6: row ordering preserved (Index column alignment)
# ---------------------------------------------------------------------------


def test_row_order_alignment():
    """Output is aligned to df's original row order, not the age-sorted order."""
    # Build a patient whose visits are in reverse age order in the DataFrame.
    rows = [
        {"Index": 0, "patient_id": "P0", "age": 62.0, "off": 40.0, "time_since_intake_off": float("nan")},
        {"Index": 1, "patient_id": "P0", "age": 60.0, "off": 35.0, "time_since_intake_off": float("nan")},
        {"Index": 2, "patient_id": "P0", "age": 61.0, "off": 38.0, "time_since_intake_off": float("nan")},
    ]
    df = pd.DataFrame(rows)
    raw = np.array([50.0, 40.0, 45.0])  # aligned to df rows (ages 62, 60, 61)

    proj = MonotonicBernsteinTrajectoryProjector()
    out = proj.transform(df, raw)

    # After projection, the youngest visit (age 60, row 1) must have the
    # smallest prediction, and the oldest visit (age 62, row 0) the largest.
    assert out[1] <= out[2] <= out[0], (
        f"Age-order monotonicity violated after projection: "
        f"age60={out[1]:.4f}, age61={out[2]:.4f}, age62={out[0]:.4f}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
