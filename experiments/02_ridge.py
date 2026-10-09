# %% [markdown]
# # 02_ridge — Ridge regression with patient-grouped CV
#
# First real linear model. Numeric features only (Ridge cannot ingest NaN so
# median-impute). Alpha sweep over [0.1, 1.0, 10.0, 100.0, 1000.0] with
# GroupKFold(n_splits=5) to prevent patient leakage.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

import skore
from skore import Project, evaluate, login

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parkinson.hub import load_skore_credentials  # noqa: E402

# %% [markdown]
# ## Load data

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

visits = X_train.merge(y_train, on="Index")

FEATURE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

X = visits[FEATURE_COLS]
y = visits["target"]
groups = visits["patient_id"]

cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

# %% [markdown]
# ## Alpha sweep — pick the best regularisation strength

# %%
alpha_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
results = {}

for alpha in alpha_values:
    ridge = make_pipeline(
        SimpleImputer(strategy="median"),
        Ridge(alpha=alpha),
    )
    rep = evaluate(ridge, X, y, splitter=cv_splits)
    rmse_df = rep.metrics.rmse()
    mean_rmse = float(rmse_df.iloc[0, 0])
    results[alpha] = (mean_rmse, rep)
    print(f"alpha={alpha:>8.1f}  RMSE mean={mean_rmse:.4f}")

best_alpha = min(results, key=lambda a: results[a][0])
best_report = results[best_alpha][1]
print(f"\nBest alpha: {best_alpha}  →  RMSE={results[best_alpha][0]:.4f}")

# %% [markdown]
# ## Push best Ridge report to Skore Hub

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("02_ridge", best_report)

# %% [markdown]
# ## Write per-experiment submission

# %%
best_ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=best_alpha),
).fit(X, y)

preds = np.clip(best_ridge.predict(X_test[FEATURE_COLS]), 0, 132)
submission = pd.DataFrame({"Index": X_test["Index"], "target": preds})
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "02_ridge.csv", index=False)
print(f"Wrote submissions/02_ridge.csv  ({len(submission)} rows)")


# %% [markdown]
# ## run() — called by 00_best_model.py when this experiment is the winner

# %%
def run() -> None:
    """Fit best Ridge on full training data, write best_submission.csv, push Hub."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        Ridge(alpha=best_alpha),
    ).fit(X, y)
    preds = np.clip(model.predict(X_test[FEATURE_COLS]), 0, 132)
    pd.DataFrame({"Index": X_test["Index"], "target": preds}).to_csv(
        submissions_dir / "best_submission.csv", index=False
    )
    project.put("best_submission", best_report)
    print("best_submission.csv written and pushed to Hub (02_ridge)")
