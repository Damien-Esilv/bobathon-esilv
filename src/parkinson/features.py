"""Clinical feature engineering for Parkinson's motor score prediction.

All transforms are stateless: `build_features` can be applied identically to
the training DataFrame and the test DataFrame without any fitted state or
access to the target column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bateman pharmacokinetic model
# ---------------------------------------------------------------------------

_KA_DEFAULT: float = 1.5  # levodopa oral absorption rate (h⁻¹)
_KE_DEFAULT: float = 0.4  # levodopa elimination rate (h⁻¹)


def _bateman(
    t: pd.Series, ka: float = _KA_DEFAULT, ke: float = _KE_DEFAULT
) -> pd.Series:
    """Bateman function: effective plasma concentration at time *t* after a dose.

    Parameters
    ----------
    t:
        Time since last levodopa intake (hours). May contain NaN.
    ka:
        First-order absorption rate constant (h⁻¹). Default 1.5.
    ke:
        First-order elimination rate constant (h⁻¹). Default 0.4.

    Returns
    -------
    pd.Series
        Unscaled relative concentration C(t) in [0, 1]. NaN propagated from *t*.
        When ka == ke the limit form ``ka * t * exp(-ka * t)`` is used.
    """
    if abs(ka - ke) < 1e-9:
        # Limit case: C(t) = ka * t * exp(-ka * t)
        c = ka * t * np.exp(-ka * t)
    else:
        c = (ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))
    # Negative concentrations arise only from numerical noise; clamp to zero.
    return c.clip(lower=0)


# ---------------------------------------------------------------------------
# Main feature engineering function
# ---------------------------------------------------------------------------


def build_features(
    df: pd.DataFrame, ka: float = _KA_DEFAULT, ke: float = _KE_DEFAULT
) -> pd.DataFrame:
    """Add clinical and pharmacodynamic features to a visits DataFrame.

    Parameters
    ----------
    df:
        A visits DataFrame containing the raw columns from X_train / X_test.
        The function does not require a ``target`` column to be present.
    ka:
        Levodopa absorption rate constant (h⁻¹) for Bateman PK features.
    ke:
        Levodopa elimination rate constant (h⁻¹) for Bateman PK features.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the following additional columns:

        **Original 9 features**

        disease_duration        : years between diagnosis and current visit
        visit_count             : total visits for each patient in this dataset
        visit_order             : chronological rank within each patient
        time_on_sq              : time_since_intake_on squared (non-linear absorption)
        time_off_sq             : time_since_intake_off squared
        on_off_gap              : off − on (motor fluctuation size; NaN if either missing)
        dose_timing             : ledd / (1 + time_since_intake_on)
        off_is_missing          : 1 when ``off`` is NaN, 0 otherwise
        ledd_is_missing         : 1 when ``ledd`` is NaN, 0 otherwise

        **Bateman PK features (family 2a)**

        bateman_on_raw          : C(time_since_intake_on); unscaled kinetics
        bateman_off_raw         : C(time_since_intake_off); unscaled kinetics
        bateman_on              : bateman_on_raw × ledd; dose-weighted ON concentration
        bateman_off             : bateman_off_raw × ledd; dose-weighted OFF concentration

        **Temporal visit interval features (family 2b)**

        delta_index_prev        : Index − previous visit Index (NaN for first visit)
        delta_index_next        : next visit Index − Index (NaN for last visit)
        delta_age_prev          : age − previous visit age (NaN for first visit)
        delta_age_next          : next visit age − age (NaN for last visit)

        **Patient progression statistics (family 2c)**

        patient_age_std         : std of age across all patient visits (proxy for follow-up span)
        patient_on_mean         : within-patient mean ON score
        patient_on_std          : within-patient std ON score
        patient_off_obs_count   : number of visits with observed OFF for this patient
        on_dev_from_patient_mean: on − patient_on_mean (deviation from personal baseline)

        **Additional missingness flags (family 2d)**

        on_is_missing           : 1 when ``on`` is NaN, 0 otherwise
        both_missing            : 1 when both ``on`` and ``off`` are NaN, 0 otherwise
    """
    out = df.copy()

    # ------------------------------------------------------------------
    # Original features (unchanged)
    # ------------------------------------------------------------------

    # Disease progression: time elapsed since first motor symptom onset.
    out["disease_duration"] = out["age"] - out["age_at_diagnosis"]

    # Patient-level visit statistics.
    visit_counts = out.groupby("patient_id")["Index"].transform("count")
    out["visit_count"] = visit_counts

    # Chronological rank within each patient.
    out["visit_order"] = (
        out.groupby("patient_id")["Index"].rank(method="first").astype(int)
    )

    # Non-linear levodopa absorption / elimination kinetics (simple polynomial).
    out["time_on_sq"] = out["time_since_intake_on"] ** 2
    out["time_off_sq"] = out["time_since_intake_off"] ** 2

    # Motor fluctuation magnitude.
    out["on_off_gap"] = out["off"] - out["on"]

    # Pharmacodynamic interaction: effective dose at assessment time.
    out["dose_timing"] = out["ledd"] / (1.0 + out["time_since_intake_on"])

    # Explicit missingness indicators.
    out["off_is_missing"] = out["off"].isna().astype(int)
    out["ledd_is_missing"] = out["ledd"].isna().astype(int)

    # ------------------------------------------------------------------
    # 2a — Bateman PK features
    # ------------------------------------------------------------------

    bateman_on_raw = _bateman(out["time_since_intake_on"], ka=ka, ke=ke)
    bateman_off_raw = _bateman(out["time_since_intake_off"], ka=ka, ke=ke)

    out["bateman_on_raw"] = bateman_on_raw
    out["bateman_off_raw"] = bateman_off_raw
    # Scale by dose; NaN when ledd or the timing column is missing.
    out["bateman_on"] = bateman_on_raw * out["ledd"]
    out["bateman_off"] = bateman_off_raw * out["ledd"]

    # ------------------------------------------------------------------
    # 2b — Temporal visit interval features
    # ------------------------------------------------------------------

    # Sort within each patient by Index to establish chronological order.
    sorted_df = out.sort_values(["patient_id", "Index"])

    grp = sorted_df.groupby("patient_id")
    index_prev = grp["Index"].shift(1)
    index_next = grp["Index"].shift(-1)
    age_prev = grp["age"].shift(1)
    age_next = grp["age"].shift(-1)

    delta_index_prev = sorted_df["Index"] - index_prev
    delta_index_next = index_next - sorted_df["Index"]
    delta_age_prev = sorted_df["age"] - age_prev
    delta_age_next = age_next - sorted_df["age"]

    # Re-align to the original row order via the Index column.
    out["delta_index_prev"] = delta_index_prev.reindex(out.index)
    out["delta_index_next"] = delta_index_next.reindex(out.index)
    out["delta_age_prev"] = delta_age_prev.reindex(out.index)
    out["delta_age_next"] = delta_age_next.reindex(out.index)

    # ------------------------------------------------------------------
    # 2c — Patient progression statistics (stateless, no target)
    # ------------------------------------------------------------------

    out["patient_age_std"] = out.groupby("patient_id")["age"].transform("std")

    patient_on_mean = out.groupby("patient_id")["on"].transform("mean")
    out["patient_on_mean"] = patient_on_mean
    out["patient_on_std"] = out.groupby("patient_id")["on"].transform("std")

    # Number of visits with observed OFF for this patient.
    out["patient_off_obs_count"] = out.groupby("patient_id")["off"].transform(
        lambda s: s.notna().sum()
    )

    out["on_dev_from_patient_mean"] = out["on"] - patient_on_mean

    # ------------------------------------------------------------------
    # 2d — Additional missingness flags
    # ------------------------------------------------------------------

    out["on_is_missing"] = out["on"].isna().astype(int)
    out["both_missing"] = (out["on"].isna() & out["off"].isna()).astype(int)

    return out
