# %% [markdown]
# # 04_skrub_tabular — TableVectorizer + HGBR with Mixed-Type Features
#
# **Team: parKINGson**
# **Members:** Damien CATAU, Malo GERVAISE, Esteban GAUBERT, Maximilien STUDER
#
# Incorporates categorical columns (`gene`, `cohort`, `rater_id`) previously ignored.
# `TableVectorizer` automatically selects the appropriate encoder per column:
# - One-hot encoding for low-cardinality categoricals (e.g., `cohort`, `sexM`).
# - `StringEncoder` for high-cardinality text (e.g., `gene`).
# - Passthrough for numeric features, preserving NaNs for native HGBR split routing.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from skore import Project, evaluate, login
from skrub import TableVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parkinson.hub import load_skore_credentials

# %% [markdown]
# ## Load data

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

visits = X_train.merge(y_train, on="Index")

# Drop identifiers and target; keep all clinical + categorical columns.
DROP_COLS = ["Index", "patient_id", "target"]
X_full = visits.drop(columns=DROP_COLS)
y = visits["target"]
groups = visits["patient_id"]

X_test_full = X_test.drop(columns=["Index", "patient_id"], errors="ignore")

cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

# %% [markdown]
# ## Evaluate TableVectorizer + HGBR

# %%
model = make_pipeline(
    TableVectorizer(),
    HistGradientBoostingRegressor(random_state=42),
)
report = evaluate(model, X_full, y, splitter=cv_splits)

rmse_df = report.metrics.rmse()
mean_rmse = float(rmse_df.iloc[0, 0])
std_rmse = float(rmse_df.iloc[0, 1])
print(f"Tabular pipeline mean RMSE: {mean_rmse:.4f}  std: {std_rmse:.4f}")

# %% [markdown]
# ## Push report to Skore Hub

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("04_skrub_tabular", report)

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
submission.to_csv(submissions_dir / "04_skrub_tabular.csv", index=False)
print(f"Wrote submissions/04_skrub_tabular.csv  ({len(submission)} rows)")


# %% [markdown]
# ## run() — called by 00_best_model.py when this experiment is the winner

# %%
def run() -> None:
    """Fit skrub TableVectorizer + HGBR pipeline on full training data and push Hub report.

    Inputs:
        None (uses global training data X_full, y, and X_test_full).

    Outputs:
        None (writes submissions/best_submission.csv and updates Skore Hub).
    """
    m = make_pipeline(
        TableVectorizer(),
        HistGradientBoostingRegressor(random_state=42),
    ).fit(X_full, y)
    # Clip predictions to valid MDS-UPDRS scale [0, 132]
    preds = np.clip(m.predict(X_test_full), 0, 132)
    pd.DataFrame({"Index": X_test["Index"], "target": preds}).to_csv(
        submissions_dir / "best_submission.csv", index=False
    )
    project.put("best_submission", report)
    print("best_submission.csv written and pushed to Hub (04_skrub_tabular)")
