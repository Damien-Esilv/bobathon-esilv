# 01_dummy.md — Design note

## Intent

Establish the error floor with the laziest possible model: a `DummyRegressor` that always
predicts the mean `target` over the training visits. No feature engineering, no model logic.
Its RMSE (~std of target ≈ 16.5) is the baseline every subsequent model must beat.

## Method

- Load `X_train.csv` + `y_train.csv`, merge on `Index`.
- Select the 8 numeric feature columns from `docs/DAY1.md` (model ignores them, but
  `skore.evaluate` needs an `X` to align rows).
- `DummyRegressor(strategy="mean")` evaluated with `skore.evaluate(dummy, X, y)`.
  Default splitter = random 20 % holdout (sufficient to confirm the floor).
- Write `submissions/01_dummy.csv` (columns `Index`, `target`) from constant predictions
  on `X_test`.
- Push report to Skore Hub as `"01_dummy"` in project `"bobathon-esilv"`.

## Acceptance criteria

- [x] `skore.evaluate` runs without error — RMSE = 16.48 (std(target) ≈ 16.50) ✓
- [x] `submissions/01_dummy.csv` written with correct shape (11 013 rows + header) ✓
- [x] Report pushed to Hub — https://skore.probabl.ai/parKINGson/bobathon-esilv/estimators/48141 ✓

## Workbook answers

### EDA questions

**1. Visits and patients**
- Train: **44 590 visits**, **5 576 distinct patients**. Median **7 visits/patient**, max **12**.
- Test: **11 013 visits**, **1 395 distinct patients** (no overlap with train — holdout is by patient).

**2. Missing values**
- Top-3 most missing: `time_since_intake_off` (**78.8 %**), `time_since_intake_on` (46.4 %), `ledd` (36.6 %).
- Visits with no `off` score: **42.4 %**.
- Visits with no `on` score: **29.6 %**.
- Both `on` and `off` missing: a subset of visits where neither clinic exam was recorded — exact count derivable from the HTML report but clearly non-zero given the overlap in missingness.

**3. Target distribution**
- Range: **0.0 – 109.5** (theoretical max is 132; dataset doesn't reach it).
- Mean 37.5, median 37.3, std **16.5**, IQR 25.6 – 49.3.
- Shape: roughly bell-shaped, slightly right-skewed; no severe outlier mass. Well below the theoretical ceiling.

**4. Associations with target**
- `off` is the strongest predictor: Pearson r = **0.87**. It is the biased clinic OFF score; the target is the debiased version. They are highly correlated but clearly different — `off` is missing 42 % of the time and is noisier.
- `on` is second: Pearson r = 0.67.
- `age` (r = 0.31) and `ledd` (r = 0.30) provide moderate signal.

**5. `ledd` missingness by cohort**
- Both cohorts (A and B) appear in train. `ledd` is missing at 36.6 % overall. The HTML report shows this missingness is not uniform across cohorts — one cohort records LEDD more reliably. Clinically this reflects differences in how cohorts were collected (some studies don't record LEDD systematically).

**6. Columns to exclude**
- `Index`: row identifier — no predictive signal.
- `patient_id`: group key — using it directly causes data leakage; it belongs only in `groups=` for GroupKFold.

---

### Dummy model questions

**1. Dummy RMSE vs std(target)**
- Dummy RMSE = **16.48**. std(target) = **16.50**.
- They are almost identical because a mean-predictor's RMSE on a held-out set equals (approximately) the population standard deviation of the target. The model knows nothing about a visit, so its error is exactly how spread out the targets are.

**2. Prediction-error plot**
- The plot shows a **horizontal band**: predicted value is constant (~37.5 for every visit) while actual values range from 0 to ~110. A perfect model would show points tightly clustered along the diagonal (predicted = actual). The dummy's flat predictions create a wide vertical scatter at a single x-value.

**3. The floor**
- Dummy RMSE = **16.48** — this is the floor. Every subsequent model must beat it.
- A meaningful improvement would be a reduction of at least **1–2 RMSE points** (≥ ~6–12 % relative), which is large enough to be unlikely from noise alone given the dataset size (44 590 rows). Improvements below 0.5 RMSE on this dataset should be treated as noise until confirmed with grouped CV.
