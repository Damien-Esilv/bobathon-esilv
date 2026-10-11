# 06_features — Design Note

## Intent

Extend the tabular gradient boosting pipeline (`04_skrub_tabular` / `05_dataops`) by incorporating
clinical domain knowledge and pharmacodynamic feature engineering. The hypothesis is that raw
measurements (`age`, `age_at_diagnosis`, `ledd`, `on`, `off`, intake timings) do not directly expose
the underlying biophysical and clinical dynamics of Parkinson's disease. By engineering explicit
disease duration, levodopa intake interactions, intra-patient visit order, motor fluctuation gaps,
and indicator flags for missing values, we provide tree splits with direct proxies for disease state.

## Method

### Feature Families in `build_features` (`src/parkinson/features.py`)

1. **Disease duration & patient progression:**
   - `disease_duration = age - age_at_diagnosis`: total time lived with Parkinson's, a primary driver of motor deterioration and levodopa-induced complications.
   - `visit_count`: total recorded clinic visits for the patient across the study.
   - `visit_order`: chronological visit index (1 to N) per patient, ordering visits along the disease timeline.

2. **Levodopa pharmacodynamics & non-linear timing:**
   - `time_on_sq`: `time_since_intake_on ** 2`, capturing the non-linear absorption/clearance curve of levodopa.
   - `time_off_sq`: `time_since_intake_off ** 2`, capturing the non-linear washout decay curve.
   - `dose_timing = ledd / (1 + time_since_intake_on)`: effective circulating dose proxy accounting for oral intake clearance over time.

3. **Motor fluctuation gap:**
   - `on_off_gap = off - on`: difference between OFF and ON motor scores, directly quantifying the patient's levodopa response amplitude (therapeutic window). NaN when either score is missing.

4. **Explicit missingness flags:**
   - `off_is_missing`: binary indicator (1 if `off` is NaN, 0 otherwise).
   - `ledd_is_missing`: binary indicator (1 if `ledd` is NaN, 0 otherwise).

### Model & Preprocessing Pipeline

- **Features**: all engineered features + raw clinical columns (excluding `Index`, `patient_id`, and `target`).
- **Pipeline**: `TableVectorizer() → HistGradientBoostingRegressor(random_state=42)`.
  - `TableVectorizer` encodes string/categorical columns (`cohort`, `sexM`, `gene`, `rater_id`).
  - HGBR routes NaNs natively at tree splits.
- **Cross-Validation**: `GroupKFold(n_splits=5)` grouped strictly by `patient_id`.
- **Submission**: Full training set fit, predictions clipped to `[0, 132]`.

## Acceptance Criteria

- [x] Feature engineering implemented in `src/parkinson/features.py` via stateless `build_features`.
- [x] Evaluated with 5-fold GroupKFold by `patient_id`.
- [x] Report pushed to Hub as `"06_features"`.
- [x] `submissions/06_features.csv` written (11 013 rows).
- [x] Meaningful reduction in CV RMSE vs tabular baseline (`04_skrub_tabular` / `05_dataops`).

## Results

| Metric | Value |
|---|---|
| Pooled RMSE (5-fold GroupKFold) | **5.680** (std: 0.120) |
| vs Tabular Baseline (`04_skrub_tabular` / `05_dataops` 7.525) | **−1.845 (−24.5%)** |
| vs Ridge (`02_ridge` 10.597) | **−4.917 (−46.4%)** |
| vs Dummy (`01_dummy` 16.48) | **−10.800 (−65.5%)** |

Skore Hub: `"06_features"` (workspace: `parKINGson`, project: `bobathon-esilv`)

## Analysis & Insights

1. **Massive performance jump (−1.845 RMSE)**:
   This represents the single largest tabular performance gain in the progression. Explicitly giving the model `disease_duration`, `on_off_gap`, and `dose_timing` enables tree nodes to split immediately on clinical concepts that previously required deep, noisy combinations of raw features.

2. **Missingness flags as regime separators**:
   The `off_is_missing` flag allows tree estimators to immediately partition visits into the high-confidence regime (where `off` is observed and `on_off_gap` is available) versus the latent reconstruction regime (where `off` is absent).

3. **Pharmacodynamic non-linearities**:
   Quadratic timing terms (`time_on_sq`, `time_off_sq`) and `dose_timing` approximate the exponential pharmacokinetic clearance curve of levodopa, providing critical signal for predicting motor fluctuation states.
