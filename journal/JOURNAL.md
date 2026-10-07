# JOURNAL

## Status

- **Project / dataset:** Parkinson's motor score regression (bobathon-esilv)
- **Goal:** Predict the unbiased true OFF MDS-UPDRS motor score for each patient visit (RMSE minimisation, Kaggle holdout by patient)
- **Last experiment:** 01_dummy — done
- **Last result:** RMSE ≈ 16.5 (dummy mean floor)

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2025-10-07
  - env manager: pip+venv - recorded: 2025-10-07
  - agent feature: installed - recorded: 2025-10-07
  - optional features: none - recorded: 2025-10-07
  - package name (`src/<pkg>/`): parkinson - recorded: 2025-10-07
  - skore mode: hub - recorded: 2025-10-07
  - skore hub workspace: parKINGson - recorded: 2025-10-07
  - skore mlflow tracking uri: n/a - recorded: 2025-10-07
  - student prior: some-sklearn - recorded: 2025-10-07
  - CV splitter family: GroupKFold - recorded: 2025-10-07

## Data understanding (EDA)

- **Status:** done - 2025-10-07
- **Summary:** 44 590 train visits / 5 576 patients (median 7 visits, max 12); test 11 013 visits / 1 395 patients (no overlap). Target `target` is true OFF MDS-UPDRS (range 0–109.5, mean 37.5, std 16.5, roughly bell-shaped). `off` is the dominant feature (Pearson r = 0.87) but 42 % missing. Holdout is by patient → GroupKFold required on Day 2.
- **Report:** [data/eda.md](../data/eda.md)

## History

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_dummy` | DummyRegressor mean — establishes the error floor | done | See 01_dummy.md | [design note](01_dummy.md) |

## Backlog

| # | Item | Source |
|---|---|---|
| B1 | Ridge regression on numeric features — first real model (Day 2) | user |
| B2 | HistGradientBoosting with native NaN handling — captures missingness as signal | user |
| B3 | GroupKFold CV by patient_id — correct evaluation matching Kaggle holdout | user |
