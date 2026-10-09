# %% [markdown]
# # 03_hgbr — HistGradientBoosting with native NaN handling
#
# Replaces Ridge + median imputation. HGBR routes NaN values at each split
# so missingness in `off`, `ledd`, and timing columns becomes a direct signal
# (skipped OFF exams correlate with worse disease state). No imputer step.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

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
# ## Evaluate HGBR — NaN preserved as clinical signal

# %%
hgbr = HistGradientBoostingRegressor(random_state=42)
report = evaluate(hgbr, X, y, splitter=cv_splits)

rmse_df = report.metrics.rmse()
mean_rmse = float(rmse_df.iloc[0, 0])
std_rmse = float(rmse_df.iloc[0, 1])
print(f"HGBR mean RMSE: {mean_rmse:.4f}  std: {std_rmse:.4f}")

# %% [markdown]
# ## Push report to Skore Hub

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("03_hgbr", report)

# %% [markdown]
# ## Write per-experiment submission

# %%
final_hgbr = HistGradientBoostingRegressor(random_state=42).fit(X, y)
preds = np.clip(final_hgbr.predict(X_test[FEATURE_COLS]), 0, 132)
submission = pd.DataFrame({"Index": X_test["Index"], "target": preds})
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "03_hgbr.csv", index=False)
print(f"Wrote submissions/03_hgbr.csv  ({len(submission)} rows)")


# %% [markdown]
# ## run() — called by 00_best_model.py when this experiment is the winner

# %%
def run() -> None:
    """Fit HGBR on full training data, write best_submission.csv, push Hub."""
    model = HistGradientBoostingRegressor(random_state=42).fit(X, y)
    preds = np.clip(model.predict(X_test[FEATURE_COLS]), 0, 132)
    pd.DataFrame({"Index": X_test["Index"], "target": preds}).to_csv(
        submissions_dir / "best_submission.csv", index=False
    )
    project.put("best_submission", report)
    print("best_submission.csv written and pushed to Hub (03_hgbr)")
