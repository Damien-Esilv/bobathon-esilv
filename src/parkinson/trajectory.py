"""Monotonic Bernstein trajectory projector for Parkinson's motor score visits.

Team: parKINGson
Members: Damien CATAU, Malo GERVAISE, Esteban GAUBERT, Maximilien STUDER

Applies a per-patient degree-3 Bernstein polynomial curve constrained to be
non-decreasing in age.  Raw blend predictions (from any upstream model) are
replaced by values read off the fitted monotone curve at each visit's normalized
age coordinate.

Mathematical summary
--------------------
For each patient i with K_i visits at ages t_1 ≤ … ≤ t_{K_i}:

1.  Normalized age:
        s = (t - t_1) / max(t_{K_i} - t_1, 0.25)   ∈ [0, 1]

2.  Degree-3 Bernstein basis (one row per visit):
        B = [[B_{0,3}(s), B_{1,3}(s), B_{2,3}(s), B_{3,3}(s)]]
    where B_{k,3}(s) are the standard Bernstein basis polynomials.

3.  Monotonicity reparameterization:
        θ = M @ α,  M = lower-triangular matrix of ones (4×4)
    Enforcing α[1:] ≥ 0 guarantees θ_0 ≤ θ_1 ≤ θ_2 ≤ θ_3, which in turn
    guarantees the Bernstein polynomial is non-decreasing for all s ∈ [0, 1].

4.  2-step IRLS via scipy.optimize.lsq_linear:
        Step 1: fit weighted constrained LS with observation weights and
                curvature regularization λ_curv = 0.01 on D₂ @ M @ α.
        Step 2: Huber-attenuate weights for visits whose residual exceeds
                3.0 points downward; re-solve once.

5.  The fitted curve is evaluated at each visit's s to obtain the monotone
    projection.  Single-visit patients pass through unchanged (the curve
    is underdetermined).

Column contract
---------------
The ``transform_predictions`` method expects a DataFrame with at minimum:
    patient_id              : patient identifier
    age                     : visit age (years, float)
    off                     : observed OFF score (float, NaN if missing)
    time_since_intake_off   : hours since last dose (float, NaN if not recorded)

``Index`` is no longer required — row alignment is tracked via an internal
``_orig_idx`` integer column so the output always matches the input row order
exactly, regardless of the DataFrame's own index or any ``Index`` column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Curvature regularization penalty on second differences of control points.
_LAMBDA_CURV: float = 0.01

# Minimum span denominator to avoid division-by-zero for patients whose
# first and last visit ages are identical or nearly so.
_MIN_SPAN: float = 0.25

# Huber threshold (UPDRS points) below the fitted curve.
_HUBER_DELTA: float = 3.0

# Observation weights by regime.
_W_OBS_TAU_RECORDED: float = 6.0   # off notna & time_since_intake_off notna
_W_OBS_TAU_NAN: float = 3.0        # off notna & time_since_intake_off isna
_W_MISSING_OFF: float = 1.0        # off isna

# Reparameterization matrix M: θ = M @ α gives θ_0 ≤ … ≤ θ_3 when α[1:] ≥ 0.
_M: np.ndarray = np.tril(np.ones((4, 4)))

# Second-difference matrix for 4 control points (shape [2, 4]).
_D2: np.ndarray = np.array(
    [[1.0, -2.0, 1.0, 0.0],
     [0.0,  1.0, -2.0, 1.0]],
    dtype=float,
)

# Bounds for lsq_linear: α[0] free, α[1:] ≥ 0.
_LB = np.array([-np.inf, 0.0, 0.0, 0.0])
_UB = np.full(4, np.inf)


# ---------------------------------------------------------------------------
# Basis helpers
# ---------------------------------------------------------------------------


def _bernstein_basis(s: np.ndarray) -> np.ndarray:
    """Return the degree-3 Bernstein basis matrix for normalized ages s.

    Parameters
    ----------
    s:
        Array of shape (n,) with values in [0, 1].

    Returns
    -------
    np.ndarray
        Matrix of shape (n, 4) where column k is B_{k,3}(s).
    """
    one_minus = 1.0 - s
    return np.column_stack([
        one_minus ** 3,
        3.0 * s * one_minus ** 2,
        3.0 * s ** 2 * one_minus,
        s ** 3,
    ])


def _normalized_age(ages: np.ndarray) -> np.ndarray:
    """Map visit ages to [0, 1] within a patient's time span.

    Parameters
    ----------
    ages:
        1-D array of ages for one patient, already sorted chronologically.

    Returns
    -------
    np.ndarray
        Normalized ages s ∈ [0, 1].
    """
    span = max(float(ages[-1]) - float(ages[0]), _MIN_SPAN)
    return (ages - ages[0]) / span


# ---------------------------------------------------------------------------
# Per-patient IRLS solve
# ---------------------------------------------------------------------------


def _observation_weights(
    off: np.ndarray,
    tau_off: np.ndarray,
) -> np.ndarray:
    """Compute initial observation weights for each visit.

    Parameters
    ----------
    off:
        Array of observed OFF scores (NaN where missing).
    tau_off:
        Array of time-since-last-dose values (NaN where not recorded).

    Returns
    -------
    np.ndarray
        Weight array of shape (n,).
    """
    w = np.full(len(off), _W_MISSING_OFF)
    obs = ~np.isnan(off)
    tau_rec = obs & ~np.isnan(tau_off)
    tau_nan = obs & np.isnan(tau_off)
    w[tau_rec] = _W_OBS_TAU_RECORDED
    w[tau_nan] = _W_OBS_TAU_NAN
    return w


def _solve_bernstein(
    s: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Solve a single constrained Bernstein fit via lsq_linear.

    Minimizes:
        0.5 * || sqrt(W) (B M α - y) ||² + λ || D₂ M α ||²

    with bounds α[0] ∈ ℝ, α[1:] ≥ 0.

    Parameters
    ----------
    s:
        Normalized age coordinates, shape (n,).
    y:
        Raw predictions, shape (n,).
    w:
        Observation weights, shape (n,).

    Returns
    -------
    np.ndarray
        Fitted control point values θ = M @ α, shape (4,).
    """
    B = _bernstein_basis(s)   # (n, 4)
    BM = B @ _M               # (n, 4) — combined design matrix in α space

    sqrt_w = np.sqrt(w)

    # Weighted observation system: augment with regularization rows.
    reg_rows = np.sqrt(_LAMBDA_CURV) * (_D2 @ _M)   # (2, 4)
    reg_rhs = np.zeros(2)

    A = np.vstack([BM * sqrt_w[:, None], reg_rows])
    b = np.concatenate([y * sqrt_w, reg_rhs])

    result = lsq_linear(A, b, bounds=(_LB, _UB), method="bvls", verbose=0)
    alpha = result.x
    return _M @ alpha


def _fit_patient(
    s: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Two-step IRLS for one patient.

    Parameters
    ----------
    s:
        Normalized age coordinates, shape (n,).
    y:
        Raw predictions, shape (n,).
    w:
        Initial observation weights, shape (n,).

    Returns
    -------
    np.ndarray
        Monotone projected predictions at each visit, shape (n,).
    """
    # --- Step 1: initial solve ---
    theta = _solve_bernstein(s, y, w)
    B = _bernstein_basis(s)
    y_hat = B @ theta
    residuals = y - y_hat

    # --- Huber attenuation for large downward residuals ---
    # A visit "plunges below" the curve when residual is strongly negative.
    below = -residuals  # positive when observation is below the curve
    w2 = w.copy()
    mask = below > 0.0
    if mask.any():
        w2[mask] = w2[mask] / (1.0 + (below[mask] / _HUBER_DELTA) ** 2)

    # --- Step 2: re-solve with attenuated weights ---
    theta = _solve_bernstein(s, y, w2)
    return _bernstein_basis(s) @ theta


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class MonotonicBernsteinTrajectoryProjector:
    """Post-process visit-level predictions to enforce per-patient monotonicity.

    For each patient, fits a degree-3 Bernstein polynomial curve over their
    normalized visit timeline using a 2-step IRLS constrained least-squares
    solve.  The monotonicity constraint (θ_0 ≤ θ_1 ≤ θ_2 ≤ θ_3) is enforced
    via the M-reparameterization, guaranteeing the curve is non-decreasing for
    all s ∈ [0, 1].

    Patients with only 1 visit are passed through unchanged.

    Parameters
    ----------
    w_obs_a:
        Weight for visits where ``off`` is observed and ``time_since_intake_off``
        is recorded (Slice A). Default 6.0.
    w_obs_b:
        Weight for visits where ``off`` is observed but ``time_since_intake_off``
        is NaN (Slice B). Default 3.0.
    w_mis:
        Weight for visits where ``off`` is missing. Default 1.0.
    lambda_curv:
        Curvature regularization penalty on D₂ control-point differences.
        Default 0.01.
    downward_threshold:
        Huber attenuation threshold (UPDRS points). Visits whose observation
        falls more than this amount below the fitted curve have their weight
        attenuated. Default 3.0.

    Usage
    -----
    >>> proj = MonotonicBernsteinTrajectoryProjector()
    >>> projected = proj.transform_predictions(df, raw_preds)
    """

    def __init__(
        self,
        w_obs_a: float = _W_OBS_TAU_RECORDED,
        w_obs_b: float = _W_OBS_TAU_NAN,
        w_mis: float = _W_MISSING_OFF,
        lambda_curv: float = _LAMBDA_CURV,
        downward_threshold: float = _HUBER_DELTA,
    ) -> None:
        self.w_obs_a = w_obs_a
        self.w_obs_b = w_obs_b
        self.w_mis = w_mis
        self.lambda_curv = lambda_curv
        self.downward_threshold = downward_threshold

    def _obs_weights(self, off: np.ndarray, tau_off: np.ndarray) -> np.ndarray:
        """Compute per-visit observation weights using instance parameters."""
        w = np.full(len(off), self.w_mis)
        obs = ~np.isnan(off)
        w[obs & ~np.isnan(tau_off)] = self.w_obs_a
        w[obs & np.isnan(tau_off)] = self.w_obs_b
        return w

    def _fit_patient_instance(
        self, s: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> np.ndarray:
        """Two-step IRLS using instance lambda_curv and downward_threshold."""
        B = _bernstein_basis(s)
        BM = B @ _M
        sqrt_w = np.sqrt(w)

        reg_rows = np.sqrt(self.lambda_curv) * (_D2 @ _M)
        reg_rhs = np.zeros(2)

        A = np.vstack([BM * sqrt_w[:, None], reg_rows])
        b_vec = np.concatenate([y * sqrt_w, reg_rhs])
        theta = _M @ lsq_linear(A, b_vec, bounds=(_LB, _UB), method="bvls", verbose=0).x

        # Huber attenuation for downward outliers.
        below = -(y - B @ theta)
        w2 = w.copy()
        mask = below > 0.0
        if mask.any():
            w2[mask] = w2[mask] / (1.0 + (below[mask] / self.downward_threshold) ** 2)

        # Step 2 re-solve.
        sqrt_w2 = np.sqrt(w2)
        A2 = np.vstack([BM * sqrt_w2[:, None], reg_rows])
        b2 = np.concatenate([y * sqrt_w2, reg_rhs])
        theta2 = _M @ lsq_linear(A2, b2, bounds=(_LB, _UB), method="bvls", verbose=0).x

        return _bernstein_basis(s) @ theta2

    def transform_predictions(
        self,
        df: pd.DataFrame,
        preds: np.ndarray,
    ) -> np.ndarray:
        """Apply monotone Bernstein projection to raw predictions.

        The output array is guaranteed to be in the exact same row order as the
        input ``df``.  Internally, visits are sorted by age per patient for
        fitting and then restored to the original row order via an ``_orig_idx``
        integer tracker.

        Parameters
        ----------
        df:
            DataFrame containing at minimum: ``patient_id``, ``age``,
            ``off``, ``time_since_intake_off``.
        preds:
            Raw prediction array of length ``len(df)``, aligned to ``df``'s
            row order.

        Returns
        -------
        np.ndarray
            Projected predictions in the exact same row order as ``df``.
        """
        df_work = df[["patient_id", "age", "off", "time_since_intake_off"]].copy()
        df_work["_orig_idx"] = np.arange(len(df_work))
        df_work["_raw_pred"] = np.asarray(preds, dtype=float)

        # Sort for chronological monotone fitting.
        df_sorted = df_work.sort_values(
            ["patient_id", "age"], kind="stable"
        ).reset_index(drop=True)

        proj_pred = np.empty(len(df_sorted), dtype=float)

        for pid, grp in df_sorted.groupby("patient_id", sort=False):
            pos = grp.index.values
            ages = grp["age"].values.astype(float)
            y = grp["_raw_pred"].values.astype(float)
            off = grp["off"].values.astype(float)
            tau = grp["time_since_intake_off"].values.astype(float)

            if len(grp) < 2:
                proj_pred[pos] = y
                continue

            s = _normalized_age(ages)
            w = self._obs_weights(off, tau)
            proj_pred[pos] = self._fit_patient_instance(s, y, w)

        df_sorted["_proj_pred"] = proj_pred

        # Restore exact original input order.
        df_restored = df_sorted.sort_values("_orig_idx").reset_index(drop=True)
        return df_restored["_proj_pred"].to_numpy()

    # Keep the old name as an alias for backward compatibility with tests.
    def transform(
        self,
        df: pd.DataFrame,
        raw_preds: np.ndarray,
    ) -> np.ndarray:
        """Alias for ``transform_predictions``; kept for backward compatibility."""
        return self.transform_predictions(df, raw_preds)

    @staticmethod
    def monotonicity_violation_rate(
        df: pd.DataFrame,
        preds: np.ndarray,
        tol: float = 1e-6,
    ) -> float:
        """Fraction of consecutive within-patient pairs with downward diff.

        Parameters
        ----------
        df:
            DataFrame with ``patient_id`` and ``age`` columns.
        preds:
            Prediction array aligned to ``df``'s row order.
        tol:
            Tolerance: a diff is "downward" only when it is < -tol.

        Returns
        -------
        float
            Fraction of consecutive pairs (sorted by age) that are downward.
            0.0 means strict monotonicity.
        """
        tmp = df[["patient_id", "age"]].copy().reset_index(drop=True)
        tmp["_pred"] = preds
        tmp = tmp.sort_values(["patient_id", "age"], kind="stable")

        grp_diff = tmp.groupby("patient_id")["_pred"].diff()
        # Only pairs within the same patient (first visit of each patient is NaN).
        pair_diffs = grp_diff.dropna().values
        if len(pair_diffs) == 0:
            return 0.0
        return float((pair_diffs < -tol).sum()) / len(pair_diffs)
