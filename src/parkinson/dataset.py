"""Patient visit sequence dataset for Parkinson's motor score prediction.

Team: parKINGson
Members: Damien CATAU, Malo GERVAISE, Esteban GAUBERT, Maximilien STUDER

Groups visits by patient, sorts chronologically by age, normalizes continuous
features, encodes categorical columns, and assembles per-visit tensors for
the SeqNet bi-directional Transformer. Stochastic OFF masking is applied
during training to force reconstruction of missing OFF states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

# Continuous features that are z-score normalized.
_NORM_COLS: list[str] = [
    "age",
    "age_at_diagnosis",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on_filled",
    "off_filled",
    "base",
]

# Binary / indicator columns included in the feature tensor WITHOUT normalization.
_FLAG_COLS: list[str] = [
    "sexM",
    "off_is_missing",
    "on_is_missing",
    "ledd_is_missing",
]

# Combined feature column list in model input order.
CONT_COLS: list[str] = _NORM_COLS + _FLAG_COLS

# Categorical columns that get integer-encoded for nn.Embedding.
CAT_COLS: list[str] = ["cohort", "gene", "rater_id"]

# Scalar target (the true unbiased OFF MDS-UPDRS score).
TARGET_COL: str = "target"

# The anchoring baseline used for residual prediction.
BASE_COL: str = "base"

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_EPSILON: float = 1e-8


@dataclass
class VisitNormalizer:
    """Z-score normalizer for continuous visit features.

    Parameters
    ----------
    cols:
        Continuous column names to z-score normalize. Defaults to ``_NORM_COLS``
        (excludes binary flag columns, which are passed through unchanged).

    Attributes
    ----------
    mean_ : dict[str, float]
        Per-column training mean.
    std_ : dict[str, float]
        Per-column training std (clamped to epsilon).
    """

    cols: list[str] = field(default_factory=lambda: list(_NORM_COLS))
    mean_: dict[str, float] = field(default_factory=dict, init=False)
    std_: dict[str, float] = field(default_factory=dict, init=False)

    def fit(self, df: pd.DataFrame) -> VisitNormalizer:
        """Compute per-column mean and std from the training DataFrame.

        Parameters
        ----------
        df:
            Training visits DataFrame. Must contain all columns in ``self.cols``.

        Returns
        -------
        self
        """
        for col in self.cols:
            vals = df[col].values.astype(float)
            self.mean_[col] = float(np.nanmean(vals))
            self.std_[col] = max(float(np.nanstd(vals)), _EPSILON)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with z-scored continuous columns.

        NaN values in continuous columns are filled with the column mean
        (= 0 in normalized space) before normalization. Binary flag columns
        (``_FLAG_COLS``) are left untouched.

        Parameters
        ----------
        df:
            DataFrame containing the continuous columns.

        Returns
        -------
        pd.DataFrame
        """
        out = df.copy()
        for col in self.cols:
            filled = out[col].fillna(self.mean_[col])
            out[col] = (filled - self.mean_[col]) / self.std_[col]
        return out


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------


@dataclass
class CategoricalEncoder:
    """Map categorical string columns to 0-based integer indices.

    Unknown categories (including unseen at inference time) map to index 0,
    which is reserved as the ``<unk>`` token.

    Parameters
    ----------
    cols:
        Categorical column names to encode.

    Attributes
    ----------
    vocab_ : dict[str, dict[str, int]]
        Per-column mapping from category string → integer index (1-based;
        0 is reserved for ``<unk>``).
    vocab_sizes_ : dict[str, int]
        Total vocabulary size per column (including the ``<unk>`` token).
    """

    cols: list[str] = field(default_factory=lambda: list(CAT_COLS))
    vocab_: dict[str, dict[str, int]] = field(default_factory=dict, init=False)
    vocab_sizes_: dict[str, int] = field(default_factory=dict, init=False)

    def fit(self, df: pd.DataFrame) -> CategoricalEncoder:
        """Build vocabulary mappings from the training DataFrame.

        Parameters
        ----------
        df:
            Training visits DataFrame.

        Returns
        -------
        self
        """
        for col in self.cols:
            unique_vals = sorted(str(v) for v in df[col].dropna().unique())
            # 0 = <unk>, then 1-based for known categories.
            self.vocab_[col] = {v: i + 1 for i, v in enumerate(unique_vals)}
            self.vocab_sizes_[col] = len(unique_vals) + 1  # +1 for unk
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with categorical columns integer-encoded.

        Parameters
        ----------
        df:
            DataFrame containing the categorical columns.

        Returns
        -------
        pd.DataFrame
            Integer-encoded (unknown / NaN → 0).
        """
        out = df.copy()
        for col in self.cols:
            mapping = self.vocab_.get(col, {})
            out[col] = df[col].astype(str).map(mapping).fillna(0).astype(int)
        return out


# ---------------------------------------------------------------------------
# Feature preparation helper
# ---------------------------------------------------------------------------


def prepare_dataframe(
    X: pd.DataFrame,
    y: pd.Series | None = None,
) -> pd.DataFrame:
    """Merge X and y, compute derived columns needed by the dataset.

    Adds:
    - ``on_filled``: ``on`` with NaN → 0 (filled before normalization).
    - ``off_filled``: ``off`` with NaN → 0.
    - ``base``: anchor = off if observed else on, else 0 (grand-mean fallback).
    - ``off_is_missing``, ``on_is_missing``, ``ledd_is_missing``: binary flags.
    - ``target``: copied from ``y`` if provided, else NaN.

    Parameters
    ----------
    X:
        Raw feature DataFrame (X_train or X_test columns).
    y:
        Target Series aligned with ``X``. Pass ``None`` for test data.

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame with all required columns.
    """
    df = X.copy()
    if y is not None:
        df[TARGET_COL] = y.values
    else:
        df[TARGET_COL] = float("nan")

    df["off_is_missing"] = df["off"].isna().astype(int)
    df["on_is_missing"] = df["on"].isna().astype(int)
    df["ledd_is_missing"] = df["ledd"].isna().astype(int)

    df["on_filled"] = df["on"].fillna(0.0)
    df["off_filled"] = df["off"].fillna(0.0)

    # Anchor: off if observed, else on if observed, else 0.
    df["base"] = df["off"].where(df["off"].notna(), df["on"]).fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ParkinsonDataset(Dataset):
    """PyTorch Dataset grouping visits by patient and sorted by age.

    Each item represents one patient's complete visit sequence. Variable-length
    sequences are padded to the batch maximum in ``collate_visits``.

    Parameters
    ----------
    df:
        Prepared DataFrame (output of ``prepare_dataframe``) with continuous
        features already z-score normalized and categorical columns already
        integer-encoded.
    augment:
        If ``True``, stochastically mask 35% of observed OFF scores during
        ``__getitem__``, replacing ``off_filled`` with 0 and setting
        ``off_is_missing`` to 1. The ``base`` is updated accordingly.
        Set to ``False`` for validation and test.
    mask_prob:
        Probability of masking an observed OFF visit during augmentation.
        Default 0.35.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        augment: bool = False,
        mask_prob: float = 0.35,
    ) -> None:
        self.augment = augment
        self.mask_prob = mask_prob

        # Group by patient_id, sort each group by age.
        self._patients: list[pd.DataFrame] = []
        for _, grp in df.groupby("patient_id"):
            self._patients.append(grp.sort_values("age").reset_index(drop=True))

    def __len__(self) -> int:
        return len(self._patients)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return tensors for patient ``idx``.

        Returns
        -------
        dict with keys:
            features   : FloatTensor (T, d_cont) — z-scored continuous features
            cat_ids    : LongTensor  (T, n_cat)  — integer category IDs
            delta_age  : FloatTensor (T,)         — Δage between consecutive visits
            age        : FloatTensor (T,)         — raw (pre-norm) age for reference
            base       : FloatTensor (T,)         — anchoring baseline
            target     : FloatTensor (T,)         — true target (NaN→0 for test)
            off_is_missing_raw : BoolTensor (T,)  — original off_is_missing flag
            valid_target : BoolTensor (T,)        — True where target is not NaN
        """
        grp = self._patients[idx].copy()
        T = len(grp)

        # Stochastic OFF masking (training augmentation only).
        if self.augment and T > 0:
            # Identify visits with observed OFF.
            observed_off = grp["off_is_missing"].values == 0
            if observed_off.any():
                mask_draw = np.random.random(T) < self.mask_prob
                to_mask = observed_off & mask_draw
                if to_mask.any():
                    grp.loc[to_mask, "off_filled"] = 0.0
                    grp.loc[to_mask, "off_is_missing"] = 1
                    # Update base: masked visits use on_filled as anchor.
                    on_vals = grp.loc[to_mask, "on_filled"].values
                    grp.loc[to_mask, "base"] = on_vals

        # Raw age for Δage computation (before normalization was applied).
        # Note: at this point df columns are already normalized, so we
        # use the normalized age column — Δage on normalized scale is fine
        # since it preserves the ordering and relative spacing.
        age_vals = grp["age"].values.astype(float)
        delta_age = np.zeros(T, dtype=float)
        if T > 1:
            delta_age[1:] = age_vals[1:] - age_vals[:-1]

        # Continuous feature matrix.
        cont_arr = grp[CONT_COLS].values.astype(float)

        # Categorical IDs.
        cat_arr = grp[CAT_COLS].values.astype(int)

        # Target and validity.
        target_vals = grp[TARGET_COL].values.astype(float)
        valid_target = ~np.isnan(target_vals)
        target_filled = np.where(valid_target, target_vals, 0.0)

        # Base anchor.
        base_vals = grp["base"].values.astype(float)

        # Original off_is_missing (before possible augmentation).
        off_missing = grp["off_is_missing"].values.astype(bool)

        return {
            "features": torch.tensor(cont_arr, dtype=torch.float32),
            "cat_ids": torch.tensor(cat_arr, dtype=torch.long),
            "delta_age": torch.tensor(delta_age, dtype=torch.float32),
            "age": torch.tensor(age_vals, dtype=torch.float32),
            "base": torch.tensor(base_vals, dtype=torch.float32),
            "target": torch.tensor(target_filled, dtype=torch.float32),
            "off_is_missing_raw": torch.tensor(off_missing, dtype=torch.bool),
            "valid_target": torch.tensor(valid_target, dtype=torch.bool),
        }


# ---------------------------------------------------------------------------
# Collation (padding)
# ---------------------------------------------------------------------------


def collate_visits(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    """Pad variable-length patient sequences to the batch maximum.

    Parameters
    ----------
    batch:
        List of dicts returned by ``ParkinsonDataset.__getitem__``.

    Returns
    -------
    dict with all keys from the dataset plus:
        attention_mask : BoolTensor (B, T_max) — True = valid token, False = pad
    """
    B = len(batch)
    T_max = max(item["features"].shape[0] for item in batch)
    d_cont = batch[0]["features"].shape[1]
    n_cat = batch[0]["cat_ids"].shape[1]

    features = torch.zeros(B, T_max, d_cont, dtype=torch.float32)
    cat_ids = torch.zeros(B, T_max, n_cat, dtype=torch.long)
    delta_age = torch.zeros(B, T_max, dtype=torch.float32)
    age = torch.zeros(B, T_max, dtype=torch.float32)
    base = torch.zeros(B, T_max, dtype=torch.float32)
    target = torch.zeros(B, T_max, dtype=torch.float32)
    off_missing = torch.zeros(B, T_max, dtype=torch.bool)
    valid_target = torch.zeros(B, T_max, dtype=torch.bool)
    attention_mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i, item in enumerate(batch):
        T = item["features"].shape[0]
        features[i, :T] = item["features"]
        cat_ids[i, :T] = item["cat_ids"]
        delta_age[i, :T] = item["delta_age"]
        age[i, :T] = item["age"]
        base[i, :T] = item["base"]
        target[i, :T] = item["target"]
        off_missing[i, :T] = item["off_is_missing_raw"]
        valid_target[i, :T] = item["valid_target"]
        attention_mask[i, :T] = True

    return {
        "features": features,
        "cat_ids": cat_ids,
        "delta_age": delta_age,
        "age": age,
        "base": base,
        "target": target,
        "off_is_missing": off_missing,
        "valid_target": valid_target,
        "attention_mask": attention_mask,
    }


def make_dataloader(
    df: pd.DataFrame,
    augment: bool,
    batch_size: int = 32,
    shuffle: bool = False,
    mask_prob: float = 0.35,
) -> DataLoader:
    """Construct a DataLoader from a prepared DataFrame.

    Parameters
    ----------
    df:
        Prepared, normalized, encoded DataFrame.
    augment:
        Whether to apply stochastic OFF masking.
    batch_size:
        Number of patients per batch.
    shuffle:
        Whether to shuffle patients between epochs.
    mask_prob:
        OFF masking probability (only used when augment=True).

    Returns
    -------
    DataLoader
    """
    dataset = ParkinsonDataset(df, augment=augment, mask_prob=mask_prob)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_visits,
        num_workers=0,
        drop_last=False,
    )
