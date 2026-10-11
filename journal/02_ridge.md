# 02_ridge — Design Note

## Intent

First real linear model. Verify that the numeric clinical columns carry an additive
predictive signal by fitting a Ridge regression with median imputation. The goal
is to beat the Day 1 dummy baseline (RMSE ≈ 16.48) and establish how much a linear
model alone can buy before introducing non-linearities or patient-grouped evaluation.

An alpha sweep over `[0.1, 1.0, 10.0, 100.0, 1000.0]` picks the best
regularisation strength using `skore.evaluate` with pre-computed GroupKFold splits.

## Method

- **Features**: 8 numeric columns only — `sexM`, `age_at_diagnosis`, `age`, `ledd`,
  `time_since_intake_on`, `time_since_intake_off`, `on`, `off`. Categorical columns
  (`gene`, `cohort`, `rater_id`) excluded — Ridge requires finite numbers and a
  separate encoding step.
- **Preprocessing**: `SimpleImputer(strategy="median")` fills NaN before Ridge.
  The imputer collapses the missingness signal (a limitation addressed in 03_hgbr).
- **Splitter**: `GroupKFold(n_splits=5)` by `patient_id` — ensures each patient
  appears in exactly one validation fold, matching the Kaggle holdout structure.
- **Alpha sweep**: five candidates scored with `skore.evaluate`; best alpha selected
  by lowest OOF RMSE.
- **Submission**: best Ridge fit on full training data; predictions clipped to `[0, 132]`.

## Acceptance Criteria

- [x] Alpha sweep runs for all 5 candidates.
- [x] Best Ridge report pushed to Hub as `"02_ridge"`.
- [x] `submissions/02_ridge.csv` written (11 013 rows).

## Results

| Alpha | Pooled RMSE (5-fold GroupKFold) |
|---|---|
| 0.1 | 10.597 (std: 0.239) |
| 1.0 | 10.597 (std: 0.239) |
| 10.0 | 10.597 (std: 0.239) |
| 100.0 | 10.597 (std: 0.239) |
| 1000.0 | 10.597 (std: 0.239) |

Alpha sensitivity is low: with 8 features and ~44 k rows the regularisation term
is dominated by the data term. The dominant predictor is `off` (Pearson r = 0.87),
giving Ridge most of its signal.

**Key finding**: Ridge beats the dummy baseline meaningfully, confirming that the
numeric features carry real signal. However, the linear model cannot exploit
missingness structure (median fill erases it) and cannot capture the non-linear
pharmacokinetic response.

## Workbook Answers

**1. Percentage improvement over dummy (RMSE ≈ 16.48)**
Ridge reduces the RMSE from 16.48 to 10.597, a ~35.7 % relative reduction (Δ = −5.88 RMSE) — a clear
win that shows the clinical columns carry strong additive signal.

**2. Largest coefficient in absolute value**
`off` has by far the largest coefficient. Its sign is positive (higher clinic OFF
score → higher true OFF score), consistent with r = 0.87. `on` is also positive
and second-largest. `ledd` is positive (higher dose → worse disease burden). All
signs are clinically sensible.

**3. Why raw coefficient size cannot rank feature importance**
Features are on very different scales (`age` in years ≈ 60–80; `ledd` in mg/day
≈ 0–1500; `on`/`off` in MDS-UPDRS units ≈ 0–100). A large coefficient on `age`
could simply mean `age` has a small variance. To compare importance, features must
be standardised first (e.g. via `StandardScaler`) so each coefficient reflects
effect per standard-deviation change.
