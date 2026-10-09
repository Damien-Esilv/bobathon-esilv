# %% [markdown]
# # 07_tuning — HGBR hyperparameter tuning and blended ensemble
#
# Tunes HistGradientBoostingRegressor on the enriched feature set from
# 06_features. A two-seed blend gives a small RMSE gain over a single model.
#
# Running this file directly performs the full grid search + blend evaluation
# and pushes the 07_tuning report to Skore Hub.
#
# `run()` is self-contained: it uses the hardcoded winning params discovered
# during the search (no re-search needed) and is called by 00_best_model.py.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer

import skore
from skore import Project, evaluate, login

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parkinson.hub import load_skore_credentials  # noqa: E402
from parkinson.features import build_features  # noqa: E402

# %% [markdown]
# ## Hardcoded best params (discovered by grid search below)
#
# Update this dict after re-running the search if a better config is found.

# %%
BEST_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 500,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
    "random_state": 42,
}


def _load_data():
    """Load raw data and apply feature engineering. Returns X_full, y, groups, X_test_full, X_test."""
    X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
    y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
    X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")
    visits_raw = X_train.merge(y_train, on="Index")
    visits = build_features(visits_raw)
    X_test_fe = build_features(X_test)
    DROP_COLS = ["Index", "patient_id", "target"]
    X_full = visits.drop(columns=DROP_COLS)
    y = visits["target"]
    groups = visits["patient_id"].values
    X_test_full = X_test_fe.drop(columns=["Index", "patient_id"], errors="ignore")
    return X_full, y, groups, X_test_full, X_test


# %% [markdown]
# ## run() — called by 00_best_model.py when this experiment is the winner

# %%
def run() -> None:
    """Fit tuned HGBR (BEST_PARAMS) on full training data.

    Writes submissions/best_submission.csv clipped to [0, 132] and pushes
    the report to Skore Hub as 'best_submission'.
    No grid search is performed here — uses the hardcoded BEST_PARAMS above.
    """
    X_full, y, groups, X_test_full, X_test = _load_data()
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    model = make_pipeline(
        TableVectorizer(),
        HistGradientBoostingRegressor(**BEST_PARAMS),
    ).fit(X_full, y)
    preds = np.clip(model.predict(X_test_full), 0, 132)
    pd.DataFrame({"Index": X_test["Index"], "target": preds}).to_csv(
        submissions_dir / "best_submission.csv", index=False
    )

    # Push a fresh report to Hub under the 'best_submission' key.
    cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))
    report = evaluate(model, X_full, y, splitter=cv_splits)
    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    project.put("best_submission", report)
    print("best_submission.csv written and pushed to Hub (07_tuning)")


# %% [markdown]
# ## Grid search + evaluation (runs only when executed directly)

# %%
if __name__ == "__main__":
    X_full, y, groups, X_test_full, X_test = _load_data()
    cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    # --- HGBR grid search (max_iter=200 for speed; winner re-evaluated at 500) ---
    hgbr_grid = [
        {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.1},
        {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 63, "min_samples_leaf": 10, "l2_regularization": 0.0},
        {"learning_rate": 0.10, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.1},
        {"learning_rate": 0.10, "max_iter": 200, "max_leaf_nodes": 63, "min_samples_leaf": 10, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 30, "l2_regularization": 0.1},
    ]
    print("=== HGBR grid search ===")
    hgbr_results = []
    for params in hgbr_grid:
        pipe = make_pipeline(
            TableVectorizer(),
            HistGradientBoostingRegressor(random_state=42, **params),
        )
        scores = cross_val_score(pipe, X_full, y, cv=cv_splits, scoring="neg_root_mean_squared_error")
        mean_rmse = -scores.mean()
        hgbr_results.append((mean_rmse, params))
        print(f"  lr={params['learning_rate']} leaves={params['max_leaf_nodes']} min_leaf={params['min_samples_leaf']}  →  RMSE={mean_rmse:.4f}")

    hgbr_results.sort(key=lambda t: t[0])
    _, best_search_params = hgbr_results[0]
    best_search_params_500 = {**best_search_params, "max_iter": 500}
    print(f"\nBest at max_iter=500: {best_search_params_500}")

    scores_500 = cross_val_score(
        make_pipeline(TableVectorizer(), HistGradientBoostingRegressor(random_state=42, **best_search_params_500)),
        X_full, y, cv=cv_splits, scoring="neg_root_mean_squared_error",
    )
    best_rmse_500 = -scores_500.mean()
    print(f"Best HGBR RMSE (max_iter=500): {best_rmse_500:.4f}")

    # --- Blended ensemble ---
    print("\n=== Blended ensemble (seed=42 + seed=0) ===")
    params2 = {**best_search_params_500, "random_state": 0}
    blend_rmses = []
    for train_idx, val_idx in cv_splits:
        X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        m1 = make_pipeline(TableVectorizer(), HistGradientBoostingRegressor(**best_search_params_500)).fit(X_tr, y_tr)
        m2 = make_pipeline(TableVectorizer(), HistGradientBoostingRegressor(**params2)).fit(X_tr, y_tr)
        blend_rmses.append(float(np.sqrt(np.mean((0.5 * m1.predict(X_val) + 0.5 * m2.predict(X_val) - y_val.values) ** 2))))
    blend_rmse = float(np.mean(blend_rmses))
    print(f"Blended ensemble RMSE: {blend_rmse:.4f}")

    # Update BEST_PARAMS if the search found something better than the hardcoded default.
    if best_rmse_500 < float(BEST_PARAMS.get("_rmse", 9999)):
        print(f"\n→ Update BEST_PARAMS in this file if {best_search_params_500} beats current default.")

    # --- Evaluate for Hub ---
    winning_params = best_search_params_500
    winning_name = "blend" if blend_rmse < best_rmse_500 else "best_hgbr"
    print(f"\nOverall winner: {winning_name}")

    final_model = make_pipeline(
        TableVectorizer(),
        HistGradientBoostingRegressor(random_state=42, **winning_params),
    ).fit(X_full, y)
    report = evaluate(
        make_pipeline(TableVectorizer(), HistGradientBoostingRegressor(random_state=42, **winning_params)),
        X_full, y, splitter=cv_splits,
    )
    rmse_df = report.metrics.rmse()
    print(f"skore RMSE: {float(rmse_df.iloc[0, 0]):.4f}")

    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    project.put("07_tuning", report)

    preds = np.clip(final_model.predict(X_test_full), 0, 132)
    submission = pd.DataFrame({"Index": X_test["Index"], "target": preds})
    submission.to_csv(submissions_dir / "07_tuning.csv", index=False)
    print(f"Wrote submissions/07_tuning.csv  ({len(submission)} rows)")
