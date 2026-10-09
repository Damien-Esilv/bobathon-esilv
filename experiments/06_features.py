# %% [markdown]
# # 06_features — Clinical feature engineering + TableVectorizer + HGBR
#
# Extends the tabular pipeline with domain-knowledge features derived from
# CONTEXT.md: disease progression duration, levodopa pharmacodynamics,
# visit chronology, motor fluctuation size, and explicit missingness flags.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer

import skore
from skore import Project, evaluate, login

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parkinson.hub import load_skore_credentials  # noqa: E402
from parkinson.features import build_features  # noqa: E402

# %% [markdown]
# ## Load data and apply clinical feature engineering

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

visits_raw = X_train.merge(y_train, on="Index")

# Apply feature engineering to train and test (stateless transform).
visits = build_features(visits_raw)
X_test_fe = build_features(X_test)

DROP_COLS = ["Index", "patient_id", "target"]
X_full = visits.drop(columns=DROP_COLS)
y = visits["target"]
groups = visits["patient_id"]

X_test_full = X_test_fe.drop(columns=["Index", "patient_id"], errors="ignore")

cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

# %% [markdown]
# ## Evaluate enriched pipeline

# %%
model = make_pipeline(
    TableVectorizer(),
    HistGradientBoostingRegressor(random_state=42),
)
report = evaluate(model, X_full, y, splitter=cv_splits)

rmse_df = report.metrics.rmse()
mean_rmse = float(rmse_df.iloc[0, 0])
std_rmse = float(rmse_df.iloc[0, 1])
print(f"Feature-engineered pipeline mean RMSE: {mean_rmse:.4f}  std: {std_rmse:.4f}")

# %% [markdown]
# ## Push report to Skore Hub

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("06_features", report)

# %% [markdown]
# ## Write per-experiment submission

# %%
final_model = make_pipeline(
    TableVectorizer(),
    HistGradientBoostingRegressor(random_state=42),
).fit(X_full, y)

preds = np.clip(final_model.predict(X_test_full), 0, 132)
submission = pd.DataFrame({"Index": X_test["Index"], "target": preds})
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "06_features.csv", index=False)
print(f"Wrote submissions/06_features.csv  ({len(submission)} rows)")


# %% [markdown]
# ## run() — called by 00_best_model.py when this experiment is the winner

# %%
def run() -> None:
    """Fit feature-engineered pipeline on full training data, write best_submission.csv."""
    m = make_pipeline(
        TableVectorizer(),
        HistGradientBoostingRegressor(random_state=42),
    ).fit(X_full, y)
    preds = np.clip(m.predict(X_test_full), 0, 132)
    pd.DataFrame({"Index": X_test["Index"], "target": preds}).to_csv(
        submissions_dir / "best_submission.csv", index=False
    )
    project.put("best_submission", report)
    print("best_submission.csv written and pushed to Hub (06_features)")
