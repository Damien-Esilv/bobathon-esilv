"""Unit tests for RetrospectiveSmoothingRegressor.

Uses a tiny synthetic dataset and reduced hyperparameters (epochs=2, d_model=32)
to run end-to-end without GPU compute.

Threading note
--------------
On macOS ARM, LightGBM's OpenMP thread pool conflicts with PyTorch's pthreads
when both run in the same process, causing SIGSEGV.  The fix is twofold:
  1. ``DISABLE_MPS_FOR_TESTS=1`` forces ``_resolve_device()`` to return CPU,
     avoiding the MPS fork-after-init crash.
  2. ``lgbm_params={"n_jobs": 1, ...}`` forces single-threaded LGBM so OpenMP
     does not spawn threads that conflict with PyTorch's already-running pool.
Both mitigations are portable (no-ops on Windows / Linux).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Must be set before torch is imported anywhere in this process.
# conftest.py sets this at collection time; repeat here for direct invocation.
os.environ.setdefault("DISABLE_MPS_FOR_TESTS", "1")

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from parkinson.pipeline import RetrospectiveSmoothingRegressor

# ---------------------------------------------------------------------------
# Shared synthetic dataset helpers
# ---------------------------------------------------------------------------

_N_PATIENTS = 6
_VISITS = (4, 3, 5, 3, 4, 4)


def _make_tiny_df(
    include_target: bool = True,
    n_patients: int = _N_PATIENTS,
) -> pd.DataFrame:
    """Minimal synthetic visits DataFrame matching the real column schema."""
    rng = np.random.default_rng(0)
    rows = []
    idx = 0
    for p_idx in range(n_patients):
        n_v = _VISITS[p_idx % len(_VISITS)]
        for v in range(n_v):
            off_val = (
                float(rng.uniform(20, 60)) if rng.random() > 0.42 else float("nan")
            )
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
                    **(
                        {"target": float(rng.uniform(20, 70))} if include_target else {}
                    ),
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def _fast_rsr(**kwargs) -> RetrospectiveSmoothingRegressor:
    """Construct RSR with minimal hyperparameters for fast, safe tests.

    ``n_jobs=1`` in lgbm_params prevents LightGBM from spawning OpenMP
    threads that conflict with PyTorch's pthreads on macOS ARM / Windows.
    """
    defaults = {
        "epochs": 2,
        "d_model": 32,
        "dim_feedforward": 32,
        "d_time": 8,
        "nhead": 2,
        "num_layers": 1,
        "batch_size": 16,
        # n_jobs=1 is mandatory here: RSR loads torch before LGBM, and
        # LightGBM's OpenMP conflicts with PyTorch's pthreads on macOS ARM.
        "lgbm_params": {"n_estimators": 10, "random_state": 0, "n_jobs": 1},
    }
    defaults.update(kwargs)
    return RetrospectiveSmoothingRegressor(**defaults)


# ---------------------------------------------------------------------------
# Test 1: fit and predict run without error, correct output shape
# ---------------------------------------------------------------------------


def test_rsr_fits_and_predicts():
    """RSR fits on a tiny DataFrame and predicts on the same data."""
    df = _make_tiny_df(include_target=True)
    X = df.drop(columns=["target"])
    y = df["target"]

    rsr = _fast_rsr()
    rsr.fit(X, y)

    preds = rsr.predict(X)
    assert isinstance(preds, np.ndarray), "predict() must return np.ndarray"
    assert preds.shape == (len(X),), f"Expected ({len(X)},), got {preds.shape}"


# ---------------------------------------------------------------------------
# Test 2: all predictions clipped to [0, 132]
# ---------------------------------------------------------------------------


def test_rsr_predictions_clipped():
    """All predictions lie within the physiological MDS-UPDRS range [0, 132]."""
    df = _make_tiny_df(include_target=True)
    X = df.drop(columns=["target"])
    y = df["target"]

    rsr = _fast_rsr()
    rsr.fit(X, y)
    preds = rsr.predict(X)

    assert (preds >= 0.0).all(), f"Prediction below 0: {preds.min()}"
    assert (preds <= 132.0).all(), f"Prediction above 132: {preds.max()}"


# ---------------------------------------------------------------------------
# Test 3: regime weighting is correct at extreme alpha values
# ---------------------------------------------------------------------------


def test_rsr_regime_weighting():
    """With alpha_obs=1, off-observed preds ≈ SeqNet. With alpha_miss=0, off-missing ≈ GBDT."""
    df = _make_tiny_df(include_target=True)
    X = df.drop(columns=["target"])
    y = df["target"]

    # alpha_obs=1, alpha_miss=0 → off-observed rows come entirely from SeqNet,
    # off-missing rows come entirely from GBDT.
    rsr_obs_seqnet = _fast_rsr(alpha_obs=1.0, alpha_miss=0.0)
    rsr_obs_seqnet.fit(X, y)

    preds_blend = rsr_obs_seqnet.predict(X)
    preds_seqnet_only = rsr_obs_seqnet._predict_seqnet(X)
    preds_gbdt_only = rsr_obs_seqnet._predict_gbdt(X)

    off_observed = ~X["off"].isna().values
    off_missing = X["off"].isna().values

    # For off-observed visits: blend = 1.0 * seqnet + 0.0 * gbdt → clipped seqnet.
    if off_observed.any():
        expected_obs = np.clip(preds_seqnet_only[off_observed], 0, 132)
        np.testing.assert_allclose(
            preds_blend[off_observed],
            expected_obs,
            rtol=1e-5,
            err_msg="off-observed blend != SeqNet with alpha_obs=1",
        )

    # For off-missing visits: blend = 0.0 * seqnet + 1.0 * gbdt → clipped gbdt.
    if off_missing.any():
        expected_miss = np.clip(preds_gbdt_only[off_missing], 0, 132)
        np.testing.assert_allclose(
            preds_blend[off_missing],
            expected_miss,
            rtol=1e-5,
            err_msg="off-missing blend != GBDT with alpha_miss=0",
        )


# ---------------------------------------------------------------------------
# Test 4: output shape matches input
# ---------------------------------------------------------------------------


def test_rsr_predict_shape_matches_input():
    """predict(X_test) has the same number of rows as X_test."""
    df_train = _make_tiny_df(include_target=True)
    df_test = _make_tiny_df(include_target=False, n_patients=3)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]

    rsr = _fast_rsr()
    rsr.fit(X_train, y_train)

    preds = rsr.predict(df_test)
    assert preds.shape == (len(df_test),), (
        f"predict shape {preds.shape} != input shape ({len(df_test)},)"
    )
