# JOURNAL

## Status

- **Project / dataset:** Parkinson's motor score regression (bobathon-esilv)
- **Goal:** Predict the unbiased true OFF MDS-UPDRS motor score for each patient visit (RMSE minimisation, Kaggle holdout by patient)
- **Last experiment:** 13_super_moe_final_sota — done
- **Last result:** Bernstein OOF RMSE **3.0007** (exp-13, Super-MoE: SeqNet v1 + TD-SeqNet + 6-regime gating + Bernstein)

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
| `01_dummy` | DummyRegressor mean — establishes the error floor | done | RMSE ≈ 16.48 | [design note](01_dummy.md) |
| `02_ridge` | Ridge regression with imputation — first linear model | done | RMSE ≈ 10.60 | [design note](02_ridge.md) |
| `03_hgbr` | HGBR with native NaN — beats ridge | done | RMSE ≈ 7.75 | [design note](03_hgbr.md) |
| `04_skrub_tabular` | TableVectorizer + HGBR adds categorical columns | done | RMSE ≈ 7.53 | [design note](04_skrub_tabular.md) |
| `05_dataops` | skrub DataOps graph — baked GroupKFold | done | RMSE ≈ 7.53 (DataOps graph) | [design note](05_dataops.md) |
| `06_features` | Clinical feature engineering (9 engineered features) | done | RMSE ≈ 5.68 | [design note](06_features.md) |
| `07_tuning` | Tuned HGBR grid search + blended ensemble | done | RMSE ≈ 5.57 | [design note](07_tuning.md) |
| `08_clinical_gbdt` | Dual-branch LightGBM + Bateman PK + MixedLM priors + rater calibration | done | Pooled 6.040 / Branch-A 5.626 / Branch-B 6.561 | [design note](08_clinical_gbdt.md) |
| `09_sequence_transformer` | Bi-directional SeqNet — anchored residual + stochastic OFF masking | done | Pooled 3.308 / Branch-A 2.938 / Branch-B 3.751 | [design note](09_sequence_transformer.md) |
| `10_retrospective_smoothing` | Calibrated dual-regime ensemble (GBDT + SeqNet blend) | done | Pooled 3.303 / Branch-A 2.943 / Branch-B 3.737 | [design note](10_retrospective_smoothing.md) |
| `11_monotonic_bernstein_trajectory` | Monotone Bernstein post-processor on RSR blend — 6-regime evaluation | done | Bernstein OOF 3.2317 | [design note](11_monotonic_bernstein_trajectory.md) |
| `12_multi_paradigm_bernstein_sota` | TD-SeqNet + Anchored GBDT + 6-regime SLSQP gating + Bernstein | done | Bernstein OOF 3.0135 | [design note](12_multi_paradigm_bernstein_sota.md) |
| `13_super_moe_final_sota` | Super-MoE: SeqNet v1 + TD-SeqNet + 6-regime gating + Bernstein | done | Bernstein OOF 3.0007 (SeqNet-v1 3.2976 / TD-SeqNet 3.0547 / Blend 3.0293) | [design note](13_super_moe_final_sota.md) |

## Backlog

| # | Item | Source |
|---|---|---|
| B1 | Ridge regression on numeric features — first real model (Day 2) | user |
| B2 | HistGradientBoosting with native NaN handling — captures missingness as signal | user |
| B3 | GroupKFold CV by patient_id — correct evaluation matching Kaggle holdout | user |
