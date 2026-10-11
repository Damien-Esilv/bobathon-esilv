"""Rater calibration transformer for Parkinson's motor score prediction.

Encodes clinician-level scoring leniency as a target-encoded feature with
Bayesian shrinkage, preventing high-variance estimates from raters with few
visits from dominating the signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class RaterCalibrator(BaseEstimator, TransformerMixin):
    """Target-encode rater leniency with Bayesian shrinkage.

    During ``fit``, learns a per-rater mean target weighted toward the grand
    mean using a smoothing count (Bayesian shrinkage):

        leniency[r] = (n_r * mu_r + m * mu_grand) / (n_r + m)

    During ``transform``, adds a ``rater_leniency`` column to the input
    DataFrame. Raters not seen during training receive the grand mean.

    Parameters
    ----------
    rater_col:
        Name of the rater identifier column in ``X``. Must be present in both
        the training and inference DataFrames.
    shrinkage:
        Smoothing count *m*. Larger values pull rater estimates more strongly
        toward the grand mean. Default 10.

    Attributes
    ----------
    rater_means_ : dict[str, float]
        Shrinkage-adjusted leniency score per rater learned from training data.
    grand_mean_ : float
        Grand mean of the target across all training visits.
    n_features_in_ : int
        Number of features seen during fit (set for sklearn compatibility).
    """

    def __init__(self, rater_col: str = "rater_id", shrinkage: float = 10.0) -> None:
        self.rater_col = rater_col
        self.shrinkage = shrinkage

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> RaterCalibrator:
        """Learn per-rater leniency scores from the training fold.

        Parameters
        ----------
        X:
            Training features. Must contain ``self.rater_col``.
        y:
            Training target values (e.g., MDS-UPDRS motor score).

        Returns
        -------
        self
        """
        if self.rater_col not in X.columns:
            raise ValueError(
                f"Column '{self.rater_col}' not found in X. "
                f"Available columns: {list(X.columns)}"
            )

        y_arr = np.asarray(y, dtype=float)
        self.grand_mean_ = float(np.nanmean(y_arr))
        self.n_features_in_ = X.shape[1]

        raters = X[self.rater_col].values
        rater_means: dict[str, float] = {}
        for rater in np.unique(raters):
            mask = raters == rater
            n_r = int(mask.sum())
            mu_r = float(np.nanmean(y_arr[mask]))
            rater_means[rater] = (n_r * mu_r + self.shrinkage * self.grand_mean_) / (
                n_r + self.shrinkage
            )
        self.rater_means_ = rater_means
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add the ``rater_leniency`` column to ``X``.

        Parameters
        ----------
        X:
            Features DataFrame. Must contain ``self.rater_col``.

        Returns
        -------
        pd.DataFrame
            Copy of ``X`` with an additional ``rater_leniency`` float column.
            Raters unseen during training receive ``grand_mean_``.
        """
        check_is_fitted(self, ["rater_means_", "grand_mean_"])

        if self.rater_col not in X.columns:
            raise ValueError(
                f"Column '{self.rater_col}' not found in X. "
                f"Available columns: {list(X.columns)}"
            )

        out = X.copy()
        out["rater_leniency"] = (
            X[self.rater_col]
            .map(self.rater_means_)
            .fillna(self.grand_mean_)
            .astype(float)
        )
        return out
