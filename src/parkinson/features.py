"""Clinical feature engineering for Parkinson's motor score prediction.

All transforms are stateless: `build_features` can be applied identically to
the training DataFrame and the test DataFrame without any fitted state.
"""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinical and pharmacodynamic features to a visits DataFrame.

    Parameters
    ----------
    df:
        A visits DataFrame containing the raw columns from X_train / X_test.
        The function does not require a `target` column to be present.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with the following additional columns:

        disease_duration      : years between diagnosis and current visit
        visit_count           : total number of visits for each patient
        visit_order           : chronological rank of the visit within each patient
        time_on_sq            : time_since_intake_on squared (non-linear absorption)
        time_off_sq           : time_since_intake_off squared
        on_off_gap            : off - on (motor fluctuation size; NaN if either missing)
        dose_timing           : ledd / (1 + time_since_intake_on) (PK interaction)
        off_is_missing        : 1 when `off` is NaN, 0 otherwise
        ledd_is_missing       : 1 when `ledd` is NaN, 0 otherwise
    """
    out = df.copy()

    # Disease progression: time elapsed since first motor symptom onset.
    out["disease_duration"] = out["age"] - out["age_at_diagnosis"]

    # Patient-level visit statistics.  For test patients these reflect their
    # test-set visits; no leakage from training labels.
    visit_counts = out.groupby("patient_id")["Index"].transform("count")
    out["visit_count"] = visit_counts

    # Chronological rank within each patient (Index is monotonically assigned
    # and serves as a reliable temporal proxy).
    out["visit_order"] = (
        out.groupby("patient_id")["Index"]
        .rank(method="first")
        .astype(int)
    )

    # Non-linear levodopa absorption / elimination kinetics.
    out["time_on_sq"] = out["time_since_intake_on"] ** 2
    out["time_off_sq"] = out["time_since_intake_off"] ** 2

    # Motor fluctuation magnitude: large gap signals strong ON/OFF contrast.
    out["on_off_gap"] = out["off"] - out["on"]

    # Pharmacodynamic interaction: effective dose at assessment time.
    out["dose_timing"] = out["ledd"] / (1.0 + out["time_since_intake_on"])

    # Explicit missingness indicators: absence of OFF measurement often means
    # the visit was ON-only (uncomfortable exams skipped → sicker patients).
    out["off_is_missing"] = out["off"].isna().astype(int)
    out["ledd_is_missing"] = out["ledd"].isna().astype(int)

    return out
