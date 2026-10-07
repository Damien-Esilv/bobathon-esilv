# %% [markdown]
# # 01_dummy — Dummy mean baseline
#
# Establishes the error floor: a DummyRegressor that always predicts the
# mean target over training visits. Every subsequent model must beat this RMSE.

# %%
import sys
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor

import skore
from skore import Project, evaluate, login

# Locate repo root from this file's position (experiments/ is one level down).
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

X.shape, y.shape  # noqa: B018

# %% [markdown]
# ## Evaluate dummy baseline

# %%
dummy = DummyRegressor(strategy="mean")
report = evaluate(dummy, X, y)
report.metrics.rmse()

# %% [markdown]
# ## Write submission file

# %%
# Fit on all training data to predict test visits.
dummy.fit(X, y)
mean_pred = dummy.predict(X_test[FEATURE_COLS])

submission = pd.DataFrame({"Index": X_test["Index"], "target": mean_pred})
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "01_dummy.csv", index=False)

submission.shape  # noqa: B018

# %% [markdown]
# ## Push report to Skore Hub

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)
