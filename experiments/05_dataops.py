# %% [markdown]
# # 05_dataops — skrub DataOps graph with GroupKFold baked in
#
# Demonstrates the DataOps pattern where GroupKFold and patient groups live
# on the graph, not as a side list that can drift. Validates that SkrubLearner
# produces predictions on X_test (which has no `target` column).

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from skrub import TableVectorizer
import skrub

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

# %% [markdown]
# ## Build the DataOps graph
#
# Groups and CV splitter are attached to `mark_as_X` so they cannot be
# accidentally omitted or misaligned with the feature matrix.

# %%
data = skrub.var("visits", visits)
groups_op = data["patient_id"]

# Use apply_func to drop "target" safely: errors="ignore" means the same
# graph works both when the data contains "target" (training) and when it
# does not (test-time prediction via SkrubLearner).
X_op = data.skb.apply_func(
    lambda df: df.drop(columns=["target"], errors="ignore")
).skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups_op},
)
y_op = data["target"].skb.mark_as_y()

pred = (
    X_op.skb.apply(TableVectorizer())
    .skb.apply(HistGradientBoostingRegressor(random_state=42), y=y_op)
)

# %% [markdown]
# ## Evaluate — CV and groups are read from the DataOp

# %%
# Pre-compute cv_splits so skore can consume them (skore.evaluate does not
# accept a DataOp directly; we pass the SkrubLearner + data dict instead).
groups = visits["patient_id"].values
X_feat = visits.drop(columns=["target"])
y_vals = visits["target"]
cv_splits = list(GroupKFold(n_splits=5).split(X_feat, y_vals, groups=groups))

learner = pred.skb.make_learner()
report = evaluate(learner, data={"visits": visits}, splitter=cv_splits)

rmse_df = report.metrics.rmse()
mean_rmse = float(rmse_df.iloc[0, 0])
std_rmse = float(rmse_df.iloc[0, 1])
print(f"DataOps mean RMSE: {mean_rmse:.4f}  std: {std_rmse:.4f}")

# %% [markdown]
# ## Push report to Skore Hub

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("05_dataops", report)

# %% [markdown]
# ## Write per-experiment submission via SkrubLearner

# %%
learner.fit({"visits": visits})
preds = np.clip(learner.predict({"visits": X_test}), 0, 132)
submission = pd.DataFrame({"Index": X_test["Index"], "target": preds})
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "05_dataops.csv", index=False)
print(f"Wrote submissions/05_dataops.csv  ({len(submission)} rows)")


# %% [markdown]
# ## run() — called by 00_best_model.py when this experiment is the winner

# %%
def run() -> None:
    """Fit DataOps learner on full training data, write best_submission.csv, push Hub."""
    l = pred.skb.make_learner()
    l.fit({"visits": visits})
    preds = np.clip(l.predict({"visits": X_test}), 0, 132)
    pd.DataFrame({"Index": X_test["Index"], "target": preds}).to_csv(
        submissions_dir / "best_submission.csv", index=False
    )
    project.put("best_submission", report)
    print("best_submission.csv written and pushed to Hub (05_dataops)")
