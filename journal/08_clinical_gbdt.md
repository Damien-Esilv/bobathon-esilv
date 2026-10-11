# 08_clinical_gbdt — Design Note

## Intent

Extend the tabular IID baseline (`07_tuning.py`) with four clinically motivated
feature families and a dual-branch LightGBM architecture that explicitly models
the two data-generating regimes created by the 42 % missingness rate in the OFF
clinic score.

The core insight is that visits with and without an observed OFF score are
fundamentally different:
- **OFF observed (Branch A, ~58 % of visits):** the model has direct access to
  the biased clinic OFF value, the ON/OFF motor fluctuation gap, and both drug
  timing measurements. The task reduces to debiasing a noisy-but-informative
  observation.
- **OFF missing (Branch B, ~42 % of visits):** the model must reconstruct the
  latent OFF severity from ON scores, pharmacokinetic priors, and longitudinal
  patient trajectory. A generic model must average these two regimes, which
  compromises both.

## Method

### Feature Families Added to `build_features`

**Bateman PK features (ka = 1.5 h⁻¹, ke = 0.4 h⁻¹):**
- `bateman_on_raw`, `bateman_off_raw`: unscaled plasma concentration C(t) at
  assessment time, encoding the full non-linear absorption/clearance curve.
- `bateman_on`, `bateman_off`: dose-weighted concentrations (× ledd), capturing
  the joint pharmacological signal of dose magnitude and timing.

**Temporal visit intervals:**
- `delta_index_prev / _next`: raw Index gap to previous/next visit (visit
  frequency proxy).
- `delta_age_prev / _next`: age-based time interval to adjacent visits (disease
  progression rate proxy).

**Patient progression statistics:**
- `patient_age_std`, `patient_on_mean`, `patient_on_std`: patient-level
  summaries of motor severity across all visits in the dataset.
- `patient_off_obs_count`: how many OFF measurements this patient has — a
  signal for disease severity and clinical care intensity.
- `on_dev_from_patient_mean`: visit-level deviation from the patient's own ON
  baseline.

**Additional missingness flags:**
- `on_is_missing`, `both_missing`: complete the missingness feature set.

### Fitted Transformers (per CV fold)

**`RaterCalibrator`** (`src/parkinson/rater.py`): learns per-rater mean target
with Bayesian shrinkage (m = 10) from the training fold. Adds `rater_leniency`
column. Unseen raters receive the grand mean.

**`PatientTrajectoryTransformer`** (`src/parkinson/mixed_effects.py`): learns
per-patient OLS slope of ON ~ age (`numpy.polyfit`). Adds `oof_slope_age`,
`patient_random_intercept`, `patient_residual`. Patients with < 2 valid ON
visits receive slope = 0.

### Dual-Branch GBDT (`src/parkinson/dual_branch.py`)

Routes the feature matrix by `off_is_missing` (int 0/1):
- **Branch A** (off observed): independent `LGBMRegressor(n_estimators=500,
  lr=0.05, num_leaves=31, min_child_samples=20, reg_lambda=0.1)`.
- **Branch B** (off missing): same hyperparameters, different training subset.

Both branches receive the full feature matrix (all engineered columns); the
routing is a training-time split, not a feature restriction.

### Evaluation Strategy

- **OOF metrics**: strict 5-fold patient-grouped CV where `RaterCalibrator` and
  `PatientTrajectoryTransformer` are re-fitted on each training fold. This
  avoids rater-leniency leakage.
- **Hub report**: transformers pre-fitted on full training set (pragmatic for
  competition reporting; same GroupKFold split structure).
- **Three RMSE metrics**: pooled, Branch-A (off observed), Branch-B (off missing).

## Acceptance Criteria

- [x] `experiments/08_clinical_gbdt.py` runs end-to-end.
- [x] Three RMSE metrics (pooled, branch-A, branch-B) are printed per fold.
- [x] Report pushed to Hub as `"08_clinical_gbdt"`.
- [x] `submissions/08_clinical_gbdt.csv` written (11 013 rows).
- [x] `00_best_model.py` automatically updated if pooled RMSE < 07_tuning baseline.
- [x] All new modules pass `ruff check` with zero errors.

## Results

| Metric | Mean (5-fold) | Std |
|---|---|---|
| Pooled RMSE | **6.040** | 0.124 |
| Branch-A RMSE (off observed) | **5.626** | 0.087 |
| Branch-B RMSE (off missing) | **6.561** | 0.194 |
| vs 07_tuning (8.290) | **−2.250 (−27.1 %)** | — |

Skore Hub report: https://skore.probabl.ai/parKINGson/bobathon-esilv/cross-validations/60827

**Winner update:** `00_best_model.py` automatically updated to point to `08_clinical_gbdt.py`.
`submissions/best_submission.csv` regenerated.
