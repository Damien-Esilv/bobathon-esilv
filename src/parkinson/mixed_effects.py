"""Longitudinal patient trajectory features for Parkinson's motor score prediction.

Team: parKINGson
Members: Damien CATAU, Malo GERVAISE, Esteban GAUBERT, Maximilien STUDER

Computes per-patient disease progression statistics from the chronological
ON-score sequence. Two computational paths are supported:

* Default (``use_mixedlm=False``): lightweight per-patient OLS slopes via
  ``numpy.polyfit``. Fast, no extra dependencies beyond numpy/pandas.
* Optional (``use_mixedlm=True``): ``statsmodels MixedLM`` fit that produces
  BLUP random intercepts and slopes for each patient.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class PatientTrajectoryTransformer(BaseEstimator, TransformerMixin):
    """Add longitudinal patient trajectory features derived from visit history.

    During ``fit``, learns per-patient linear trends from the training fold
    using the ``on`` score as a proxy for motor severity over ``age``.
    During ``transform``, adds three columns to the input DataFrame:

    * ``oof_slope_age``: per-patient slope of ``on`` ~ ``age`` from the
      training fold. For patients not seen in training, falls back to 0.
    * ``patient_random_intercept``: per-patient mean of the residuals of
      ``on`` after removing the cohort-level OLS trend (approximates the
      MixedLM random intercept BLUP).
    * ``patient_residual``: visit-level deviation of ``on`` from the
      patient's own linear trend.

    When ``use_mixedlm=True``, an additional column ``mixedlm_intercept`` is
    added using ``statsmodels MixedLM`` BLUP random intercepts.

    Parameters
    ----------
    on_col:
        Name of the ON motor score column. Default ``"on"``.
    age_col:
        Name of the age column used as the time axis. Default ``"age"``.
    patient_col:
        Name of the patient identifier column. Default ``"patient_id"``.
    use_mixedlm:
        If ``True``, fit a ``statsmodels MixedLM`` model on the training fold
        to obtain BLUP random intercepts. Requires ``statsmodels``. Default
        ``False``.
    min_visits_for_slope:
        Minimum number of non-NaN ``on`` visits required to compute a
        per-patient slope. Patients below this threshold receive slope 0.
        Default 2.

    Attributes
    ----------
    patient_slopes_ : dict[str, float]
        Per-patient OLS slope (``on`` ~ ``age``) learned from training data.
    patient_intercepts_ : dict[str, float]
        Per-patient OLS intercept.
    patient_rand_intercepts_ : dict[str, float]
        Per-patient mean ON residual from the cohort-level trend (approximate
        random intercept BLUP).
    global_slope_ : float
        Cohort-level OLS slope of ``on`` ~ ``age`` across all training visits.
    global_intercept_ : float
        Cohort-level OLS intercept.
    mixedlm_rand_intercepts_ : dict[str, float] or None
        BLUP random intercepts from ``statsmodels MixedLM`` when
        ``use_mixedlm=True``, else ``None``.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(
        self,
        on_col: str = "on",
        age_col: str = "age",
        patient_col: str = "patient_id",
        use_mixedlm: bool = False,
        min_visits_for_slope: int = 2,
    ) -> None:
        self.on_col = on_col
        self.age_col = age_col
        self.patient_col = patient_col
        self.use_mixedlm = use_mixedlm
        self.min_visits_for_slope = min_visits_for_slope

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> PatientTrajectoryTransformer:
        """Learn per-patient progression slopes from the training fold.

        Parameters
        ----------
        X:
            Training features. Must contain ``on_col``, ``age_col``, and
            ``patient_col``.
        y:
            Unused. Present for sklearn API compatibility.

        Returns
        -------
        self
        """
        for col in (self.on_col, self.age_col, self.patient_col):
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in X.")

        self.n_features_in_ = X.shape[1]

        # Drop rows where on or age is missing — cannot fit a slope on NaN.
        mask = X[self.on_col].notna() & X[self.age_col].notna()
        X_valid = X[mask]

        # --- Cohort-level OLS (global trend) ---
        if len(X_valid) >= 2:
            global_coef = np.polyfit(X_valid[self.age_col], X_valid[self.on_col], 1)
            self.global_slope_ = float(global_coef[0])
            self.global_intercept_ = float(global_coef[1])
        else:
            self.global_slope_ = 0.0
            self.global_intercept_ = (
                float(X_valid[self.on_col].mean()) if len(X_valid) > 0 else 0.0
            )

        global_trend = (
            self.global_slope_ * X_valid[self.age_col] + self.global_intercept_
        )
        on_residuals = X_valid[self.on_col] - global_trend

        # --- Per-patient OLS slopes and random intercept approximation ---
        patient_slopes: dict[str, float] = {}
        patient_intercepts_: dict[str, float] = {}
        patient_rand_intercepts: dict[str, float] = {}

        for patient_id, group in X_valid.groupby(self.patient_col):
            ages = group[self.age_col].values
            on_vals = group[self.on_col].values

            if len(ages) >= self.min_visits_for_slope:
                coef = np.polyfit(ages, on_vals, 1)
                patient_slopes[patient_id] = float(coef[0])
                patient_intercepts_[patient_id] = float(coef[1])
            else:
                patient_slopes[patient_id] = 0.0
                patient_intercepts_[patient_id] = float(np.mean(on_vals))

            # Random intercept ≈ mean ON residual from the cohort trend.
            patient_resid = on_residuals.loc[group.index]
            patient_rand_intercepts[patient_id] = float(patient_resid.mean())

        self.patient_slopes_ = patient_slopes
        self.patient_intercepts_ = patient_intercepts_
        self.patient_rand_intercepts_ = patient_rand_intercepts

        # --- Optional MixedLM path ---
        self.mixedlm_rand_intercepts_: dict[str, float] | None = None
        if self.use_mixedlm:
            self._fit_mixedlm(X_valid)

        return self

    def _fit_mixedlm(self, X_valid: pd.DataFrame) -> None:
        """Fit statsmodels MixedLM and store BLUP random intercepts."""
        try:
            import statsmodels.formula.api as smf
        except ImportError as exc:
            raise ImportError(
                "statsmodels is required for use_mixedlm=True. "
                "Install it with: pip install statsmodels"
            ) from exc

        df_lm = X_valid[[self.patient_col, self.age_col, self.on_col]].copy()
        df_lm = df_lm.rename(
            columns={
                self.patient_col: "_patient",
                self.age_col: "_age",
                self.on_col: "_on",
            }
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    "_on ~ _age",
                    df_lm,
                    groups=df_lm["_patient"],
                )
                result = model.fit(reml=True, disp=False)

            rand_effects = result.random_effects
            self.mixedlm_rand_intercepts_ = {
                patient: float(effect.iloc[0])
                for patient, effect in rand_effects.items()
            }
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"MixedLM fitting failed ({exc}); falling back to OLS random intercepts.",
                stacklevel=2,
            )
            self.mixedlm_rand_intercepts_ = None

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add trajectory features to ``X``.

        Parameters
        ----------
        X:
            Features DataFrame. Must contain ``on_col``, ``age_col``, and
            ``patient_col``.

        Returns
        -------
        pd.DataFrame
            Copy of ``X`` with additional trajectory columns.
        """
        check_is_fitted(self, ["patient_slopes_", "global_slope_"])

        out = X.copy()

        # Per-patient OLS slope (0 for unseen patients).
        out["oof_slope_age"] = (
            X[self.patient_col].map(self.patient_slopes_).fillna(0.0).astype(float)
        )

        # Per-patient mean ON residual from cohort trend (random intercept approx).
        out["patient_random_intercept"] = (
            X[self.patient_col]
            .map(self.patient_rand_intercepts_)
            .fillna(0.0)
            .astype(float)
        )

        # Visit-level deviation of ON from the patient's own linear trend.
        patient_slope_vals = X[self.patient_col].map(self.patient_slopes_).fillna(0.0)
        patient_intercept_vals = (
            X[self.patient_col]
            .map(self.patient_intercepts_)
            .fillna(self.global_intercept_)
        )
        patient_trend = patient_slope_vals * X[self.age_col] + patient_intercept_vals
        out["patient_residual"] = X[self.on_col] - patient_trend

        # Optional MixedLM random intercept column.
        if self.use_mixedlm and self.mixedlm_rand_intercepts_ is not None:
            out["mixedlm_intercept"] = (
                X[self.patient_col]
                .map(self.mixedlm_rand_intercepts_)
                .fillna(0.0)
                .astype(float)
            )

        return out
