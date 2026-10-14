# %% [markdown]
# # 01_dummy — Dummy Mean Baseline
#
# **Team: parKINGson**
# **Members:** Damien CATAU, Malo GERVAISE, Esteban GAUBERT, Maximilien STUDER
#
# Establishes the error floor: a DummyRegressor that always predicts the
# empirical mean target over training visits. Any valid predictive model
# must achieve an RMSE lower than this baseline (RMSE ≈ 16.48, matching target std).

# %%
import sys
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from skore import Project, evaluate, login

# Locate repo root from this file's position (experiments/ is one level down).
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
# ## run() — Standalone execution function
#
# Called by 00_best_model.py or standalone runner.

# %%
def run() -> None:
    """Fit the dummy mean model on all training visits and write submission.

    Inputs:
        None (reads data from project data/ directory).

    Outputs:
        None (writes submissions/01_dummy.csv and updates Skore Hub).
    """
    # Baseline model: predicts mean target regardless of feature values.
    model = DummyRegressor(strategy="mean")
    model.fit(X, y)
    preds = model.predict(X_test[FEATURE_COLS])
    
    out_path = submissions_dir / "01_dummy.csv"
    pd.DataFrame({"Index": X_test["Index"], "target": preds}).to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(preds)} rows)")
    
    # Push report to Skore Hub
    project.put("01_dummy", report)
    print("Report pushed to Skore Hub as '01_dummy'")


if __name__ == "__main__":
    run()

