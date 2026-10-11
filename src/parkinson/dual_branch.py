"""Dual-branch LightGBM regressor for Parkinson's motor score prediction.

Routes visits through two independent LGBMRegressor estimators at fit and
predict time, based on whether the OFF clinic measurement was recorded:

* Branch A (off observed): leverages the ON/OFF delta, direct OFF score, and
  Bateman concentration features.
* Branch B (off missing): relies on the ON score, Bateman dynamics, and
  longitudinal trajectory priors.

Routing uses the pre-computed ``off_is_missing`` indicator column (int 0/1),
which is always available after ``build_features`` is applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class DualBranchGBDT(BaseEstimator, RegressorMixin):
    """Two-branch LightGBM regressor routing on OFF-measurement availability.

    Visits where ``off`` was recorded (``off_is_missing == 0``) are routed to
    Branch A; visits where it is missing (``off_is_missing == 1``) are routed
    to Branch B. Each branch is an independent ``LGBMRegressor`` fitted only
    on its own subset of the training data.

    Parameters
    ----------
    off_missing_col:
        Name of the routing indicator column in ``X``. Must be an integer
        column with values 0 (off observed) and 1 (off missing). Produced by
        ``build_features`` as ``"off_is_missing"``.
    n_estimators:
        Number of boosting rounds for both branches. Default 500.
    learning_rate:
        Step-size shrinkage for both branches. Default 0.05.
    num_leaves:
        Maximum number of leaves per tree for both branches. Default 31.
    min_child_samples:
        Minimum number of data points in a leaf. Default 20.
    reg_lambda:
        L2 regularisation term for both branches. Default 0.1.
    random_state:
        Random seed for reproducibility. Default 42.
    n_jobs:
        Number of parallel threads for LightGBM. Default -1 (all cores).
        Set to 1 to avoid OpenMP/pthreads conflicts when PyTorch is loaded
        in the same process (required on macOS with Apple Silicon).
    verbose:
        LightGBM verbosity. Default -1 (silent).

    Attributes
    ----------
    branch_a_ : LGBMRegressor
        Fitted estimator for visits where OFF was observed.
    branch_b_ : LGBMRegressor
        Fitted estimator for visits where OFF was missing.
    n_branch_a_ : int
        Number of training rows routed to Branch A.
    n_branch_b_ : int
        Number of training rows routed to Branch B.
    feature_names_in_ : list[str]
        Column names of the training DataFrame (excluding the routing column).
    n_features_in_ : int
        Number of feature columns seen during fit (excluding routing column).
    """

    def __init__(
        self,
        off_missing_col: str = "off_is_missing",
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        reg_lambda: float = 0.1,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = -1,
    ) -> None:
        self.off_missing_col = off_missing_col
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _make_lgbm(self) -> LGBMRegressor:
        """Construct a fresh LGBMRegressor with the current hyperparameters."""
        return LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
        )

    def _get_feature_matrix(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Return (X_features, routing_mask) after validating the routing column."""
        if self.off_missing_col not in X.columns:
            raise ValueError(
                f"Routing column '{self.off_missing_col}' not found in X. "
                f"Apply build_features before fitting DualBranchGBDT. "
                f"Available columns: {list(X.columns)}"
            )
        routing = X[self.off_missing_col].values.astype(int)
        X_feat = X.drop(columns=[self.off_missing_col])
        return X_feat, routing

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> DualBranchGBDT:
        """Fit Branch A and Branch B on their respective visit subsets.

        Parameters
        ----------
        X:
            Feature matrix. Must include ``off_missing_col``.
        y:
            Target values.

        Returns
        -------
        self
        """
        X_feat, routing = self._get_feature_matrix(X)
        y_arr = np.asarray(y, dtype=float)

        mask_a = routing == 0  # off observed
        mask_b = routing == 1  # off missing

        self.n_branch_a_ = int(mask_a.sum())
        self.n_branch_b_ = int(mask_b.sum())

        if self.n_branch_a_ == 0:
            raise ValueError(
                "Branch A has zero training samples (no visits with observed OFF)."
            )
        if self.n_branch_b_ == 0:
            raise ValueError(
                "Branch B has zero training samples (no visits with missing OFF)."
            )

        self.feature_names_in_ = list(X_feat.columns)
        self.n_features_in_ = len(self.feature_names_in_)

        self.branch_a_ = self._make_lgbm()
        self.branch_b_ = self._make_lgbm()

        self.branch_a_.fit(X_feat.iloc[mask_a], y_arr[mask_a])
        self.branch_b_.fit(X_feat.iloc[mask_b], y_arr[mask_b])

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict by routing each row to the appropriate branch.

        Parameters
        ----------
        X:
            Feature matrix. Must include ``off_missing_col``.

        Returns
        -------
        np.ndarray
            Predictions in the original row order of ``X``.
        """
        check_is_fitted(self, ["branch_a_", "branch_b_"])

        X_feat, routing = self._get_feature_matrix(X)

        mask_a = routing == 0
        mask_b = routing == 1

        predictions = np.empty(len(X), dtype=float)

        if mask_a.any():
            predictions[mask_a] = self.branch_a_.predict(X_feat.iloc[mask_a])
        if mask_b.any():
            predictions[mask_b] = self.branch_b_.predict(X_feat.iloc[mask_b])

        return predictions
