"""Retrospective smoothing ensemble for Parkinson's motor score prediction.

Combines the `DualBranchGBDT` tabular model (08_clinical_gbdt) with the
`SeqNet` bi-directional Transformer (09_sequence_transformer) under a
calibrated dual-regime convex blend:

    ŷ = α * ŷ_SeqNet + (1 − α) * ŷ_GBDT

where α = α_obs when the OFF clinic measurement is observed, and
          α_miss when the OFF clinic measurement is missing.

The blend weights are expected to be pre-optimized on pooled out-of-fold
predictions by the experiment script and passed as constructor arguments.

Cross-platform threading note
------------------------------
LightGBM uses OpenMP for parallelism; PyTorch uses pthreads (and on
macOS ARM, the Accelerate framework).  When both are loaded in the same
Python process, OpenMP and pthreads can race on the macOS Grand Central
Dispatch queue, causing SIGSEGV.  The fix applied here is unconditional:
``DualBranchGBDT`` is always constructed with ``n_jobs=1`` inside
``RetrospectiveSmoothingRegressor``.  Callers that want multi-threaded
GBDT (e.g. standalone experiment scripts without a SeqNet arm) can pass
``lgbm_n_jobs=-1`` explicitly.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
from skrub import TableVectorizer
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from parkinson.dataset import (
    CAT_COLS,
    CONT_COLS,
    CategoricalEncoder,
    VisitNormalizer,
    make_dataloader,
    prepare_dataframe,
)
from parkinson.dual_branch import DualBranchGBDT
from parkinson.features import build_features
from parkinson.mixed_effects import PatientTrajectoryTransformer
from parkinson.rater import RaterCalibrator
from parkinson.seqnet import build_seqnet

# ---------------------------------------------------------------------------
# Default LightGBM hyperparameters (matching experiment 08)
# ---------------------------------------------------------------------------

_DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "reg_lambda": 0.1,
    "random_state": 42,
    # Always single-threaded inside RSR: LightGBM's OpenMP conflicts with
    # PyTorch's pthreads on macOS ARM (and can cause hangs on Windows too).
    # Experiment scripts that run GBDT alone should override this.
    "n_jobs": 1,
}

# Columns dropped before the sklearn GBDT pipeline (not before transformers).
_PIPELINE_DROP: list[str] = ["Index", "target"]
# Columns retained for transformer steps but dropped before the GBDT model.
_MODEL_DROP: list[str] = ["patient_id"]


# ---------------------------------------------------------------------------
# RetrospectiveSmoothingRegressor
# ---------------------------------------------------------------------------


class RetrospectiveSmoothingRegressor(BaseEstimator, RegressorMixin):
    """Dual-arm ensemble blending DualBranchGBDT and SeqNet predictions.

    Regime-conditioned blending:
    - When ``off`` is observed  : ŷ = α_obs  * ŷ_SeqNet + (1 − α_obs)  * ŷ_GBDT
    - When ``off`` is missing   : ŷ = α_miss * ŷ_SeqNet + (1 − α_miss) * ŷ_GBDT

    ``alpha_obs`` and ``alpha_miss`` should be optimized on pooled OOF predictions
    before calling ``fit`` for the final production model.

    Parameters
    ----------
    alpha_obs:
        SeqNet weight for visits where ``off`` is observed. Default 0.78.
    alpha_miss:
        SeqNet weight for visits where ``off`` is missing. Default 0.76.
    lgbm_params:
        Hyperparameter dict forwarded to ``DualBranchGBDT``. Defaults to
        the experiment-08 tuned configuration if ``None``.
    epochs:
        Number of SeqNet training epochs. Default 30.
    batch_size:
        Patients per mini-batch for SeqNet. Default 32.
    lr:
        Peak learning rate for AdamW + OneCycleLR. Default 1e-3.
    weight_decay:
        AdamW weight decay. Default 1e-4.
    mask_prob:
        Probability of stochastic OFF masking during training. Default 0.35.
    residual_scale:
        Scale factor in anchored prediction ŷ = base + scale × r̂. Default 20.
    d_model:
        SeqNet Transformer model dimension. Default 128.
    d_time:
        Sinusoidal temporal embedding dimension. Default 16.
    nhead:
        Number of attention heads. Default 4.
    num_layers:
        Number of Transformer encoder layers. Default 2.
    dim_feedforward:
        Feed-forward sublayer width. Default 256.
    dropout:
        Dropout probability in the Transformer. Default 0.1.
    random_state:
        Seed for reproducibility (applied to numpy and torch). Default 42.

    Attributes
    ----------
    gbdt_pipeline_ : sklearn.pipeline.Pipeline
        Fitted ``TableVectorizer → DualBranchGBDT`` pipeline.
    rater_calibrator_ : RaterCalibrator
        Fitted rater leniency encoder.
    trajectory_transformer_ : PatientTrajectoryTransformer
        Fitted per-patient OLS trajectory transformer.
    seqnet_model_ : SeqNet
        Trained sequence model (CPU, eval mode).
    seqnet_norm_ : VisitNormalizer
        Fitted continuous feature normalizer for SeqNet.
    seqnet_enc_ : CategoricalEncoder
        Fitted categorical encoder for SeqNet.
    device_ : torch.device
        PyTorch device used for training.
    n_features_in_ : int
        Number of input columns in ``X`` at fit time.
    """

    def __init__(
        self,
        alpha_obs: float = 0.78,
        alpha_miss: float = 0.76,
        lgbm_params: dict[str, Any] | None = None,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        mask_prob: float = 0.35,
        residual_scale: float = 20.0,
        d_model: int = 128,
        d_time: int = 16,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.alpha_obs = alpha_obs
        self.alpha_miss = alpha_miss
        self.lgbm_params = lgbm_params
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.mask_prob = mask_prob
        self.residual_scale = residual_scale
        self.d_model = d_model
        self.d_time = d_time
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_lgbm_params(self) -> dict[str, Any]:
        base = dict(_DEFAULT_LGBM_PARAMS)
        base["random_state"] = self.random_state
        if self.lgbm_params is not None:
            base.update(self.lgbm_params)
        return base

    def _resolve_device(self) -> torch.device:
        import os

        # MPS is disabled in test environments to avoid the fork-after-init
        # segfault that occurs when pytest collects modules on macOS ARM.
        if os.environ.get("DISABLE_MPS_FOR_TESTS"):
            return torch.device("cpu")
        return torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    # ------------------------------------------------------------------
    # GBDT arm
    # ------------------------------------------------------------------

    def _fit_gbdt_arm(self, X_raw: pd.DataFrame, y: pd.Series) -> None:
        """Fit the GBDT preprocessing chain and DualBranchGBDT.

        Parameters
        ----------
        X_raw:
            Raw X DataFrame as received by ``fit()``. Must contain
            ``patient_id``, ``Index``, and all clinical columns.
        y:
            Target series.
        """
        # 1. Stateless feature engineering (needs target column present).
        visits_raw = X_raw.copy()
        visits_raw["target"] = y.values
        visits = build_features(visits_raw)

        X_fe = visits.drop(columns=_PIPELINE_DROP)

        # 2. RaterCalibrator (target-encoded rater leniency).
        rc = RaterCalibrator(shrinkage=10.0)
        rc.fit(X_fe, y)
        X_rc = rc.transform(X_fe)

        # 3. PatientTrajectoryTransformer (per-patient OLS slopes).
        ptt = PatientTrajectoryTransformer(use_mixedlm=False)
        ptt.fit(X_rc)
        X_t = ptt.transform(X_rc)

        # 4. Drop patient_id; fit TableVectorizer + DualBranchGBDT.
        X_model = X_t.drop(columns=_MODEL_DROP, errors="ignore")
        lgbm_p = self._resolve_lgbm_params()
        pipeline = make_pipeline(TableVectorizer(), DualBranchGBDT(**lgbm_p))
        pipeline.fit(X_model, y)

        self.rater_calibrator_ = rc
        self.trajectory_transformer_ = ptt
        self.gbdt_pipeline_ = pipeline

    def _predict_gbdt(self, X_raw: pd.DataFrame) -> np.ndarray:
        """GBDT arm inference; returns array aligned to ``X_raw`` row order."""
        check_is_fitted(self, ["gbdt_pipeline_"])

        # Stateless feature engineering (no target column on test data).
        visits = build_features(X_raw)
        X_fe = visits.drop(columns=["Index"], errors="ignore")

        X_rc = self.rater_calibrator_.transform(X_fe)
        X_t = self.trajectory_transformer_.transform(X_rc)
        X_model = X_t.drop(columns=_MODEL_DROP, errors="ignore")

        return self.gbdt_pipeline_.predict(X_model)

    # ------------------------------------------------------------------
    # SeqNet arm
    # ------------------------------------------------------------------

    def _fit_seqnet_arm(self, X_raw: pd.DataFrame, y: pd.Series) -> None:
        """Train the SeqNet model on the full training data.

        Parameters
        ----------
        X_raw:
            Raw X DataFrame.
        y:
            Target series.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        device = self._resolve_device()
        self.device_ = device

        # Prepare, normalize, encode.
        df = prepare_dataframe(X_raw, y)
        norm = VisitNormalizer()
        norm.fit(df)
        df = norm.transform(df)
        enc = CategoricalEncoder()
        enc.fit(df)
        df = enc.transform(df)

        loader = make_dataloader(
            df,
            augment=True,
            batch_size=self.batch_size,
            shuffle=True,
            mask_prob=self.mask_prob,
        )

        cat_vocab = [enc.vocab_sizes_[c] for c in CAT_COLS]
        model = build_seqnet(
            d_cont=len(CONT_COLS),
            cat_vocab_sizes=cat_vocab,
            d_time=self.d_time,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            residual_scale=self.residual_scale,
        ).to(device)

        optimizer = AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_steps = self.epochs * len(loader)
        scheduler = OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=0.1
        )

        for _epoch in range(self.epochs):
            model.train()
            for batch in loader:
                features = batch["features"].to(device)
                cat_ids = batch["cat_ids"].to(device)
                delta_age = batch["delta_age"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                base = batch["base"].to(device)
                target = batch["target"].to(device)
                valid = batch["valid_target"].to(device)

                preds = model(features, cat_ids, delta_age, attention_mask, base)
                loss_mask = attention_mask & valid
                if not loss_mask.any():
                    continue
                loss = ((preds - target) ** 2)[loss_mask].mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

        # Store on CPU in eval mode.
        model.eval().cpu()
        self.seqnet_model_ = model
        self.seqnet_norm_ = norm
        self.seqnet_enc_ = enc

    @torch.no_grad()
    def _predict_seqnet(self, X_raw: pd.DataFrame) -> np.ndarray:
        """SeqNet arm inference; returns array aligned to ``X_raw`` row order.

        The DataLoader sorts visits within each patient by age. Predictions are
        collected in that order then re-sorted back to match ``X_raw.index``.
        """
        check_is_fitted(self, ["seqnet_model_"])

        df = prepare_dataframe(X_raw, y=None)
        df = self.seqnet_norm_.transform(df)
        df = self.seqnet_enc_.transform(df)

        # Sort by patient_id + age to match DataLoader order; track Index.
        df_sorted = df.sort_values(["patient_id", "age"]).copy()

        loader = make_dataloader(
            df_sorted, augment=False, batch_size=self.batch_size, shuffle=False
        )

        model = self.seqnet_model_

        all_preds: list[float] = []
        for batch in loader:
            features = batch["features"]
            cat_ids = batch["cat_ids"]
            delta_age = batch["delta_age"]
            attention_mask = batch["attention_mask"]
            base = batch["base"]

            preds = model(features, cat_ids, delta_age, attention_mask, base)
            # Collect valid (non-padding) predictions in patient-age order.
            for i in range(preds.shape[0]):
                seq_len = int(attention_mask[i].sum())
                all_preds.extend(preds[i, :seq_len].tolist())

        preds_arr = np.array(all_preds)
        df_sorted["_pred"] = preds_arr

        # Re-align to original X_raw index order via Index column.
        aligned = df_sorted.sort_values("Index")["_pred"].values
        return aligned

    # ------------------------------------------------------------------
    # Public sklearn interface
    # ------------------------------------------------------------------

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> RetrospectiveSmoothingRegressor:
        """Train both GBDT and SeqNet arms on the full training data.

        Parameters
        ----------
        X:
            Raw feature DataFrame. Must contain ``patient_id``, ``Index``, and
            all clinical columns from X_train.csv. The ``target`` column must
            NOT be present (it is provided via ``y``).
        y:
            Target values.

        Returns
        -------
        self
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        self.n_features_in_ = X.shape[1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # SeqNet must be trained BEFORE LightGBM.  On macOS, LightGBM's
            # OpenMP thread pool (libomp) leaves threads in an inconsistent state
            # that causes a SIGSEGV when PyTorch's pthreads-based scheduler
            # tries to initialise afterwards.  Fitting SeqNet first ensures
            # PyTorch's runtime is fully initialised before OpenMP starts.
            self._fit_seqnet_arm(X, y)
            self._fit_gbdt_arm(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Blend GBDT and SeqNet predictions with regime-calibrated weights.

        Parameters
        ----------
        X:
            Raw feature DataFrame (same column layout as training ``X``).

        Returns
        -------
        np.ndarray
            Blended predictions clipped to [0, 132].
        """
        check_is_fitted(self, ["gbdt_pipeline_", "seqnet_model_"])

        preds_gbdt = self._predict_gbdt(X)
        preds_seqnet = self._predict_seqnet(X)

        # Determine blending regime from raw OFF column.
        off_is_missing = X["off"].isna().values

        alpha = np.where(off_is_missing, self.alpha_miss, self.alpha_obs)
        blended = alpha * preds_seqnet + (1.0 - alpha) * preds_gbdt
        return np.clip(blended, 0.0, 132.0)
