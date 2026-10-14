# %% [markdown]
# # 05_dataops — skrub DataOps Graph with Patient-Grouped CV Baked In
#
# **Team: parKINGson**
# **Members:** Damien CATAU, Malo GERVAISE, Esteban GAUBERT, Maximilien STUDER
#
# Demonstrates the declarative DataOps pattern:
# - Patient grouping and GroupKFold are attached directly to the computation graph node.
# - Eliminates the risk of passing mismatched split indices or features during evaluation.
# - Evaluates using `SkrubLearner` to predict directly on `X_test`.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
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
    """Fit DataOps learner on full training visits, write best_submission.csv, and push Hub report.

    Inputs:
        None (uses global training data visits and test data X_test).

    Outputs:
        None (writes submissions/best_submission.csv and updates Skore Hub).
    """
    l = pred.skb.make_learner()
    l.fit({"visits": visits})
    # Clip predictions to valid MDS-UPDRS scale [0, 132]
    preds = np.clip(l.predict({"visits": X_test}), 0, 132)
    pd.DataFrame({"Index": X_test["Index"], "target": preds}).to_csv(
        submissions_dir / "best_submission.csv", index=False
    )
    project.put("best_submission", report)
    print("best_submission.csv written and pushed to Hub (05_dataops)")
