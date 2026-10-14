# %% [markdown]
# # 08_clinical_gbdt — Dual-Branch LightGBM with Clinical Feature Engineering
#
# **Team: parKINGson**
# **Members:** Damien CATAU, Malo GERVAISE, Esteban GAUBERT, Maximilien STUDER
#
# Extends the tabular pipeline with four domain-driven feature families
# and a dual-branch architecture that routes visits through two independent
# LGBMRegressor estimators depending on whether the OFF clinic measurement is
# available:
# - Branch A: when OFF is recorded (fits motor delta and direct OFF score).
# - Branch B: when OFF is missing (relies on ON score and trajectory priors).
#
# Feature families added:
#   - Bateman PK curves (levodopa plasma concentration via ka=1.5, ke=0.4 h⁻¹)
#   - Temporal visit intervals (delta_age between consecutive visits)
#   - Patient progression statistics (per-patient OLS slope and residuals)
#   - Rater leniency calibration with Bayesian shrinkage

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from skore import Project, evaluate, login
from skrub import TableVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parkinson.dual_branch import DualBranchGBDT
from parkinson.features import build_features
from parkinson.hub import load_skore_credentials
from parkinson.mixed_effects import PatientTrajectoryTransformer
from parkinson.rater import RaterCalibrator

# %% [markdown]
# ## Hyperparameters for the dual-branch LightGBM estimator

# %%
LGBM_PARAMS: dict = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "reg_lambda": 0.1,
    "random_state": 42,
}

# Reference RMSE from 07_tuning (tuned HGBR blend) — used for comparison.
BASELINE_RMSE_07: float = 8.29  # update after re-checking 07_tuning scores

# %% [markdown]
# ## Data loading and stateless feature engineering

# %%


# Columns to drop only before the sklearn pipeline (not before transformers).
_PIPELINE_DROP = ["Index", "target"]
# Columns present in feature-engineered data that must be dropped before GBDT.
_MODEL_DROP = ["patient_id"]


def _load_data() -> tuple[
    pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame, pd.DataFrame
]:
    """Load raw data, merge, and apply stateless feature engineering.

    ``X_full`` retains ``patient_id`` so that ``PatientTrajectoryTransformer``
    can group visits. Drop it (along with ``rater_id`` after calibration) just
    before the GBDT pipeline via ``_drop_for_model``.

    Returns
    -------
    X_full : pd.DataFrame
        Engineered feature matrix for all training visits. Retains patient_id.
    y : pd.Series
        Training targets.
    groups : np.ndarray
        Patient IDs for GroupKFold.
    X_test_fe : pd.DataFrame
        Engineered feature matrix for test visits. Retains patient_id.
    X_test_raw : pd.DataFrame
        Raw test DataFrame (for Index column in submission).
    """
    X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
    y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
    X_test_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

    visits_raw = X_train.merge(y_train, on="Index")
    visits = build_features(visits_raw)
    X_test_fe = build_features(X_test_raw)

    X_full = visits.drop(columns=_PIPELINE_DROP)
    y = visits["target"]
    groups = visits["patient_id"].values

    X_test_fe = X_test_fe.drop(columns=["Index"], errors="ignore")
    return X_full, y, groups, X_test_fe, X_test_raw


def _drop_for_model(X: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns before TableVectorizer + DualBranchGBDT."""
    return X.drop(columns=_MODEL_DROP, errors="ignore")


# %% [markdown]
# ## Per-fold preprocessing helpers

# %%


def _fit_preprocess_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit RaterCalibrator and PatientTrajectoryTransformer on the training fold.

    Inputs:
        X_tr (pd.DataFrame): Training fold features including rater_id and patient_id.
        y_tr (pd.Series): Training fold target values.
        X_val (pd.DataFrame): Validation fold features.

    Outputs:
        tuple[pd.DataFrame, pd.DataFrame]: (X_tr_t, X_val_t) with fitted calibrations applied.
    """
    rc = RaterCalibrator(shrinkage=10.0)
    rc.fit(X_tr, y_tr)
    X_tr_rc = rc.transform(X_tr)
    X_val_rc = rc.transform(X_val)

    ptt = PatientTrajectoryTransformer(use_mixedlm=False)
    ptt.fit(X_tr_rc)
    X_tr_t = ptt.transform(X_tr_rc)
    X_val_t = ptt.transform(X_val_rc)

    return X_tr_t, X_val_t


def _fit_preprocess_full(
    X_full: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Fit both transformers on the full training set and return transformed X.

    Inputs:
        X_full (pd.DataFrame): All training visits.
        y (pd.Series): All training targets.

    Outputs:
        pd.DataFrame: Transformed training feature matrix.
    """
    rc = RaterCalibrator(shrinkage=10.0)
    rc.fit(X_full, y)
    X_rc = rc.transform(X_full)

    ptt = PatientTrajectoryTransformer(use_mixedlm=False)
    ptt.fit(X_rc)
    return ptt.transform(X_rc)


def _fit_preprocess_test(
    X_full: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Fit transformers on full train, transform the test set.

    Inputs:
        X_full (pd.DataFrame): Full training visits.
        y (pd.Series): Full training targets.
        X_test (pd.DataFrame): Test visits.

    Outputs:
        pd.DataFrame: Transformed test feature matrix.
    """
    rc = RaterCalibrator(shrinkage=10.0)
    rc.fit(X_full, y)
    X_test_rc = rc.transform(X_test)

    ptt = PatientTrajectoryTransformer(use_mixedlm=False)
    ptt.fit(rc.transform(X_full))
    return ptt.transform(X_test_rc)


# %% [markdown]
# ## 5-fold grouped cross-validation with three RMSE metrics

# %%


def _cv_evaluate(
    X_full: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
) -> dict[str, float]:
    """Run 5-fold GroupKFold and compute pooled + per-regime RMSE metrics.

    Returns a dict with keys: pooled_rmse, branch_a_rmse, branch_b_rmse
    (mean and std across folds).
    """
    cv = GroupKFold(n_splits=5)
    cv_splits = list(cv.split(X_full, y, groups=groups))

    pooled_rmses: list[float] = []
    branch_a_rmses: list[float] = []
    branch_b_rmses: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_tr = X_full.iloc[train_idx].reset_index(drop=True)
        X_val = X_full.iloc[val_idx].reset_index(drop=True)
        y_tr = y.iloc[train_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)

        # Per-fold fitting of the two stateful transformers (need patient_id).
        X_tr_t, X_val_t = _fit_preprocess_fold(X_tr, y_tr, X_val)

        # Drop patient_id before the sklearn pipeline.
        X_tr_model = _drop_for_model(X_tr_t)
        X_val_model = _drop_for_model(X_val_t)

        # Build and fit the vectorizer + dual-branch pipeline.
        pipeline = make_pipeline(TableVectorizer(), DualBranchGBDT(**LGBM_PARAMS))
        pipeline.fit(X_tr_model, y_tr)
        preds = pipeline.predict(X_val_model)

        # Pooled RMSE.
        pooled_rmse = float(np.sqrt(np.mean((preds - y_val.values) ** 2)))
        pooled_rmses.append(pooled_rmse)

        # Per-regime RMSE using off_is_missing in the VALIDATION set.
        off_missing_val = X_val["off_is_missing"].values.astype(int)
        mask_a = off_missing_val == 0  # off observed
        mask_b = off_missing_val == 1  # off missing

        if mask_a.any():
            branch_a_rmses.append(
                float(np.sqrt(np.mean((preds[mask_a] - y_val.values[mask_a]) ** 2)))
            )
        if mask_b.any():
            branch_b_rmses.append(
                float(np.sqrt(np.mean((preds[mask_b] - y_val.values[mask_b]) ** 2)))
            )

        print(
            f"  Fold {fold_idx + 1}/5 — pooled: {pooled_rmse:.4f}  "
            f"branch-A: {branch_a_rmses[-1]:.4f}  branch-B: {branch_b_rmses[-1]:.4f}"
        )

    return {
        "pooled_mean": float(np.mean(pooled_rmses)),
        "pooled_std": float(np.std(pooled_rmses)),
        "branch_a_mean": float(np.mean(branch_a_rmses)),
        "branch_a_std": float(np.std(branch_a_rmses)),
        "branch_b_mean": float(np.mean(branch_b_rmses)),
        "branch_b_std": float(np.std(branch_b_rmses)),
    }


# %% [markdown]
# ## run() — called by 00_best_model.py when this experiment is the winner

# %%


def run() -> None:
    """Fit clinical GBDT on full training data.

    Writes submissions/best_submission.csv clipped to [0, 132] and pushes the
    report to Skore Hub as 'best_submission'.
    """
    X_full, y, groups, X_test_fe, X_test_raw = _load_data()
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    X_full_t = _fit_preprocess_full(X_full, y)
    X_test_t = _fit_preprocess_test(X_full, y, X_test_fe)
    X_full_model = _drop_for_model(X_full_t)
    X_test_model = _drop_for_model(X_test_t)

    model = make_pipeline(TableVectorizer(), DualBranchGBDT(**LGBM_PARAMS))
    model.fit(X_full_model, y)
    preds = np.clip(model.predict(X_test_model), 0, 132)
    pd.DataFrame({"Index": X_test_raw["Index"], "target": preds}).to_csv(
        submissions_dir / "best_submission.csv", index=False
    )

    # Hub report: pre-fitted transformers, GroupKFold CV splits.
    cv_splits = list(GroupKFold(n_splits=5).split(X_full_model, y, groups=groups))
    report = evaluate(model, X_full_model, y, splitter=cv_splits)
    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    project.put("best_submission", report)
    print("best_submission.csv written and pushed to Hub (08_clinical_gbdt)")


# %% [markdown]
# ## Main: cross-validate, push report, write submission, compare vs baseline

# %%
if __name__ == "__main__":
    print("=" * 60)
    print("08_clinical_gbdt — Dual-branch LightGBM with clinical features")
    print("=" * 60)

    X_full, y, groups, X_test_fe, X_test_raw = _load_data()
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    # --- 5-fold OOF evaluation with strict per-fold preprocessing ---
    print("\n--- 5-fold GroupKFold evaluation ---")
    metrics = _cv_evaluate(X_full, y, groups)

    print(
        f"\n  Pooled RMSE  : {metrics['pooled_mean']:.4f} ± {metrics['pooled_std']:.4f}"
    )
    print(
        f"  Branch-A RMSE (off observed): {metrics['branch_a_mean']:.4f} ± {metrics['branch_a_std']:.4f}"
    )
    print(
        f"  Branch-B RMSE (off missing) : {metrics['branch_b_mean']:.4f} ± {metrics['branch_b_std']:.4f}"
    )

    delta = BASELINE_RMSE_07 - metrics["pooled_mean"]
    pct = 100.0 * delta / BASELINE_RMSE_07
    print(f"\n  vs 07_tuning ({BASELINE_RMSE_07:.4f}): Δ = {delta:+.4f}  ({pct:+.1f}%)")

    # --- Fit on full training data for Hub report and submission ---
    print("\n--- Fitting on full training data ---")
    X_full_t = _fit_preprocess_full(X_full, y)
    X_test_t = _fit_preprocess_test(X_full, y, X_test_fe)
    X_full_model = _drop_for_model(X_full_t)
    X_test_model = _drop_for_model(X_test_t)

    cv_splits = list(GroupKFold(n_splits=5).split(X_full_model, y, groups=groups))
    model = make_pipeline(TableVectorizer(), DualBranchGBDT(**LGBM_PARAMS))

    report = evaluate(model, X_full_model, y, splitter=cv_splits)
    rmse_df = report.metrics.rmse()
    hub_rmse = float(rmse_df.iloc[0, 0])
    print(f"  skore RMSE (pre-fitted transformers): {hub_rmse:.4f}")

    # Push to Hub.
    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    project.put("08_clinical_gbdt", report)

    # Write per-experiment submission.
    model.fit(X_full_model, y)
    preds = np.clip(model.predict(X_test_model), 0, 132)
    submission = pd.DataFrame({"Index": X_test_raw["Index"], "target": preds})
    submission.to_csv(submissions_dir / "08_clinical_gbdt.csv", index=False)
    print(f"  Wrote submissions/08_clinical_gbdt.csv  ({len(submission)} rows)")

    # --- Conditionally update 00_best_model.py if this experiment wins ---
    if metrics["pooled_mean"] < BASELINE_RMSE_07:
        best_model_path = PROJECT_ROOT / "experiments" / "00_best_model.py"
        content = best_model_path.read_text()
        old_winner_line = next(
            (
                line
                for line in content.splitlines()
                if line.strip().startswith("WINNER =")
            ),
            None,
        )
        if old_winner_line is not None:
            new_content = content.replace(
                old_winner_line, 'WINNER = "08_clinical_gbdt.py"'
            )
            best_model_path.write_text(new_content)
            print(
                f"\n  ✓ 00_best_model.py updated: WINNER = '08_clinical_gbdt.py' "
                f"(pooled RMSE {metrics['pooled_mean']:.4f} < baseline {BASELINE_RMSE_07:.4f})"
            )
            # Write best_submission.csv.
            submission.to_csv(submissions_dir / "best_submission.csv", index=False)
            print(
                f"  ✓ Wrote submissions/best_submission.csv  ({len(submission)} rows)"
            )
        else:
            print(
                "\n  ⚠ Could not locate WINNER line in 00_best_model.py — update manually."
            )
    else:
        print(
            f"\n  07_tuning remains the winner "
            f"(08 pooled RMSE {metrics['pooled_mean']:.4f} ≥ baseline {BASELINE_RMSE_07:.4f})."
        )
