# %% [markdown]
# # EDA: Parkinson's Motor Score (bobathon-esilv)
#
# Exploratory data analysis of the Parkinson's motor score dataset,
# run before designing a model.
#
# - **Raw data** is read-only — `X_train.csv`, `y_train.csv`, `X_test.csv`
#   live in `data/`. This file never cleans or modifies them.
# - **Outputs** go under `EDA_DIR` (the repo's `data/`): one
#   `eda_<table>.html` report per table, summarised in `eda.md`.

# %%
import json
from pathlib import Path

import pandas as pd
import skrub

# This file lives in data/, so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# EDA outputs always land here (created if missing); the raw data may
# live elsewhere.
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# Merge X_train + y_train on Index to get a single visits table with the
# target. X_test is profiled separately (no target, patient holdout).

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

RAW = X_train.merge(y_train, on="Index")
RAW.shape  # noqa: B018

# %% [markdown]
# ## Table overview — training visits
#
# Per-column summary: dtype, fraction missing, number of unique values.
# Interactive report saved to `data/eda_train.html`.

# %%
report_train = skrub.TableReport(RAW, title="train visits", verbose=0)
report_train.write_html(EDA_DIR / "eda_train.html")

summary = json.loads(report_train.json())
n_rows = summary.get("n_rows")
overview = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary.get("columns", [])
]
{"n_rows": n_rows, "n_columns": len(overview), "columns": overview}  # noqa: B018

# %% [markdown]
# ## Table overview — test visits
#
# Same profile on X_test (no target). Saved to `data/eda_test.html`.

# %%
report_test = skrub.TableReport(X_test, title="test visits", verbose=0)
report_test.write_html(EDA_DIR / "eda_test.html")

summary_test = json.loads(report_test.json())
n_rows_test = summary_test.get("n_rows")
overview_test = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_test.get("columns", [])
]
{"n_rows": n_rows_test, "n_columns": len(overview_test), "columns": overview_test}  # noqa: B018

# %% [markdown]
# ## Patient grouping
#
# Count visits per patient in train and test — key for GroupKFold later.

# %%
train_patients = RAW["patient_id"].nunique()
test_patients = X_test["patient_id"].nunique()
visits_per_patient = RAW.groupby("patient_id").size()
{
    "train_visits": int(RAW.shape[0]),
    "train_patients": int(train_patients),
    "test_visits": int(X_test.shape[0]),
    "test_patients": int(test_patients),
    "visits_per_patient_median": float(visits_per_patient.median()),
    "visits_per_patient_max": int(visits_per_patient.max()),
}  # noqa: B018

# %% [markdown]
# ## Target
#
# Distribution of `target` (true unbiased OFF MDS-UPDRS score).
# This shapes the metric choice and regression baseline.

# %%
TARGET = "target"
target_col = next(
    (col for col in summary.get("columns", []) if col.get("name") == TARGET), None
)
target_col  # noqa: B018

# %% [markdown]
# ## Structure signals
#
# Datetime columns (time-based splits) and high-cardinality id/group-like
# columns (GroupKFold candidates).

# %%
datetime_cols = [
    col.get("name")
    for col in summary.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (col.get("n_unique") or 0) / n_rows if n_rows else None,
        }
        for col in summary.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:10]}  # noqa: B018

# %% [markdown]
# ## Associations
#
# Strongest pairwise column associations. Strong feature↔target links are
# candidate predictors; implausibly perfect ones flag possible leakage.

# %%
assoc = skrub.column_associations(RAW)
rows = assoc.to_dicts() if hasattr(assoc, "to_dicts") else assoc.to_dict(orient="records")
target_links = [
    row
    for row in rows
    if row["left_column_name"] == TARGET or row["right_column_name"] == TARGET
]
{"with_target": target_links[:15], "strongest": rows[:10]}  # noqa: B018

# %% [markdown]
# ## Summary
#
# The findings and their modelling implications are written up in
# `data/eda.md`.
