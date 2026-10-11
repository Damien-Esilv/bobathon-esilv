# Clinical GBDT Plan — Experiment 08

## Top-Level Overview

**Goal:** Extend the Parkinson's MDS-UPDRS Part III prediction pipeline beyond
IID tabular modeling by encoding four biostatistical insights: dopaminergic
pharmacokinetics (Bateman PK curves), longitudinal disease trajectory (MixedLM
random effects), rater scoring bias (Bayesian-shrinkage target encoding), and
temporal visit intervals. Then build a dual-branch LightGBM architecture that
routes samples through two specialized estimators based on whether the `off`
clinic measurement is available. Evaluate with the existing 5-fold
patient-grouped cross-validation and compare against `07_tuning.py` (tuned
HGBR blend) on pooled RMSE and per-regime RMSE (observed-OFF vs missing-OFF).

**Files changed / created:**
- `src/parkinson/features.py` — augmented with four new feature families
- `src/parkinson/rater.py` — new `RaterCalibrator` sklearn transformer
- `src/parkinson/mixed_effects.py` — new `PatientTrajectoryTransformer`
- `src/parkinson/dual_branch.py` — new `DualBranchGBDT` sklearn estimator
- `experiments/08_clinical_gbdt.py` — new experiment script
- `journal/08_clinical_gbdt.md` — design note (to be approved before implementation)
- `requirements.txt` — add `lightgbm`

**Constraints carried forward:**
- No random row splits for evaluation — only `GroupKFold(n_splits=5)` by `patient_id`.
- Predictions clipped to `[0, 132]`.
- Hub workspace: `parKINGson`, project: `bobathon-esilv`.
- `ruff` lint + type annotations on all new code.
- No target-column access inside `build_features`; target-dependent transforms
  live in fitted sklearn transformers only.

---

## Sub-Task 1 — Add LightGBM dependency

**Status:** `[ ] pending`

**Intent:**
Install `lightgbm` into `.venv` and record it in `requirements.txt`. All
downstream sub-tasks depend on LightGBM being importable.

**Expected Outcomes:**
- `import lightgbm` succeeds in the active venv.
- `requirements.txt` contains a `lightgbm` line.

**Todo List:**
1. Run `.venv/bin/pip install lightgbm` (or the equivalent pip install in the
   active venv).
2. Add `lightgbm` to `requirements.txt`.
3. Verify import: `.venv/bin/python -c "import lightgbm; print(lightgbm.__version__)"`.

**Relevant Context:**
- Current `requirements.txt` lives at repo root.
- LightGBM handles NaN natively; no imputer step is needed in either branch.

---

## Sub-Task 2 — Extend `features.py` with four new feature families

**Status:** `[ ] pending`

**Intent:**
Augment the stateless `build_features` function in `src/parkinson/features.py`
with four new feature families. All transforms remain stateless (no fitting
against `y`) and are identical for train and test DataFrames.

**Expected Outcomes:**
- `build_features` returns a DataFrame with 17 additional columns (beyond the
  existing 9 engineered columns).
- No `target` column is read inside `build_features`.
- Function is callable on train and test without leakage.

**Feature families to add:**

### 2a — Bateman PK Features (dopaminergic pharmacokinetics)
Compute effective levodopa plasma concentration for both ON and OFF states
using the Bateman function:

```
C(t) = (k_a / (k_a - k_e)) * (exp(-k_e * t) - exp(-k_a * t))
```

Use standard physiological constants:
- `k_a = 1.5` h⁻¹ (absorption rate, levodopa oral)
- `k_e = 0.4` h⁻¹ (elimination rate, levodopa)

New columns:
- `bateman_on` — `C(time_since_intake_on)` scaled by `ledd`; NaN when either
  input is missing
- `bateman_off` — `C(time_since_intake_off)` scaled by `ledd`; NaN when either
  input is missing
- `bateman_on_raw` — unscaled `C(time_since_intake_on)` (captures kinetics
  without dose magnitude)
- `bateman_off_raw` — unscaled `C(time_since_intake_off)`

Guard: when `k_a == k_e` use the limit `C(t) = k_a * t * exp(-k_a * t)`.
Guard: clamp negative concentrations to 0.

### 2b — Temporal Visit Interval Features
Compute four delta features using chronological visit ordering within each
patient (sort by `Index` within each `patient_id`):

- `delta_index_prev` — difference in `Index` to the previous visit for the
  same patient (NaN for first visit)
- `delta_index_next` — difference in `Index` to the next visit (NaN for last
  visit)
- `delta_age_prev` — difference in `age` to the previous visit (NaN for first)
- `delta_age_next` — difference in `age` to the next visit (NaN for last)

These are computed purely from columns already in `df`, making them safe for
both train and test.

### 2c — Patient Progression Statistics (simple/stateless)
Lightweight patient-level stats that can be computed without `y`:

- `patient_age_std` — std of `age` across all visits in the patient's group
  (proxy for visit spread / follow-up duration)
- `patient_on_mean` — within-patient mean of `on` score across all visits
- `patient_on_std` — within-patient std of `on` score
- `patient_off_observed_count` — number of visits with non-missing `off` for
  this patient
- `on_deviation_from_patient_mean` — `on - patient_on_mean` (visit-level
  deviation from the patient's own ON baseline)

### 2d — Missingness Pattern Features
Two additional indicator flags:

- `on_is_missing` — `on.isna().astype(int)` (mirrors existing `off_is_missing`)
- `both_missing` — 1 when both `on` and `off` are NaN simultaneously

**Todo List:**
1. Open `src/parkinson/features.py`.
2. Add `_bateman(t, k_a, k_e)` private helper function implementing the
   Bateman formula with the `k_a == k_e` guard.
3. In `build_features`, after existing features, add Section 2a (Bateman PK).
4. Add Section 2b (temporal intervals) using `groupby + shift` on sorted Index.
5. Add Section 2c (patient progression statistics) using `groupby + transform`.
6. Add Section 2d (additional missingness indicators).
7. Update the docstring of `build_features` to list all new columns.
8. Run `ruff check src/parkinson/features.py --fix` and `ruff format`.

**Relevant Context:**
- `src/parkinson/features.py` — current file (68 lines, 9 engineered columns)
- Existing `visit_order` is already computed from `Index` rank — reuse the
  same sort order for shift-based interval features.
- `time_since_intake_on` has 46.4 % missingness; `time_since_intake_off` has
  78.8 % missingness — Bateman features will be NaN-heavy; that is expected and
  LightGBM handles it natively.

---

## Sub-Task 3 — Implement `RaterCalibrator` sklearn transformer

**Status:** `[ ] pending`

**Intent:**
Build a fitted sklearn transformer (`BaseEstimator + TransformerMixin`) in a
new file `src/parkinson/rater.py`. During `.fit(X, y)` it learns a per-rater
leniency index from the training fold using Bayesian shrinkage (weighted blend
of the rater's own mean and the grand mean). During `.transform(X)` it adds a
`rater_leniency` column. Unseen raters fall back to the grand mean.

**Expected Outcomes:**
- `src/parkinson/rater.py` created with class `RaterCalibrator`.
- `RaterCalibrator` is a valid sklearn transformer (passes `check_estimator`
  mentally: `fit` returns `self`, `transform` returns a DataFrame with the new
  column, `set_output(transform="pandas")` compatible).
- In a pipeline, this transformer runs after data loading and before GBDT.

**Design:**
```
rater_leniency[r] = (n_r * mu_r + m * mu_grand) / (n_r + m)
```
where:
- `n_r` = number of training visits for rater `r`
- `mu_r` = mean `target` for rater `r` in the training fold
- `mu_grand` = grand mean of `target` in the training fold
- `m` = shrinkage hyperparameter (default `m=10`)

**Parameters:**
- `rater_col: str = "rater_id"` — name of the rater column in X
- `shrinkage: float = 10.0` — smoothing count

**Fit:** reads `y` directly (target-encoded feature). Requires `X` to contain
`rater_col`. Stores `rater_means_` (dict), `grand_mean_` (float), `is_fitted_`
flag.

**Transform:** adds column `rater_leniency` (float) with NaN-free output
(unseen raters → `grand_mean_`).

**Todo List:**
1. Create `src/parkinson/rater.py`.
2. Implement `RaterCalibrator(BaseEstimator, TransformerMixin)` as described.
3. Add full numpy-doc docstring.
4. Run `ruff check src/parkinson/rater.py --fix` and `ruff format`.

**Relevant Context:**
- `rater_id` has 50 distinct values, 0 % missing in both train and test.
- EDA note: rater bias may be encodeable (eda.md line 111).
- Target-encoded features must only be learned from training fold data — the
  transformer in a pipeline ensures this automatically.
- `src/parkinson/hub.py` is a good style reference.

---

## Sub-Task 4 — Implement `PatientTrajectoryTransformer`

**Status:** `[ ] pending`

**Intent:**
Build a fitted sklearn transformer in `src/parkinson/mixed_effects.py` that
computes two kinds of longitudinal patient features:

**Primary (always-on):** patient-level OOF progression statistics derived
purely from chronological ordering, computed without any external model:
- `oof_slope_age` — linear slope of `on` ~ `age` within each patient
  (computed from training fold history; for test patients, computed from
  test data itself since no training history is available).
- `patient_residual` — visit-level residual of `on` after subtracting the
  patient's own linear trend (deviation from personal trajectory).

**Optional MixedLM branch:** a `statsmodels MixedLM` fit on the training fold
that produces patient-level random intercepts and slopes. Controlled by the
`use_mixedlm: bool = False` constructor parameter (False by default to avoid
the statsmodels dependency and runtime cost in the primary pipeline path; True
for an experimental variant).

When `use_mixedlm=False`, the transformer computes:
- `patient_random_intercept` — patient mean of `on` residuals from a global
  OLS trend (lightweight approximation of the MixedLM random intercept BLUP)
- `patient_age_slope` — per-patient linear slope of `on` ~ `age` from the
  training fold

When `use_mixedlm=True`:
- Fit `MixedLM(endog=on, exog=age, groups=patient_id)` on the training fold.
- Extract BLUP random intercepts and slopes.
- Map to test visits by `patient_id`; unseen patients fall back to 0.

**Todo List:**
1. Create `src/parkinson/mixed_effects.py`.
2. Implement `PatientTrajectoryTransformer(BaseEstimator, TransformerMixin)`
   with parameter `use_mixedlm: bool = False`.
3. `.fit(X, y=None)`: compute per-patient OLS slopes from training fold data.
   If `use_mixedlm=True`, also fit `statsmodels MixedLM`.
4. `.transform(X)`: add columns `oof_slope_age`, `patient_residual`,
   `patient_random_intercept`.
5. For test patients with no training history, use a fallback of 0 for
   intercept and 0 for slope (grand-mean trajectory).
6. Run `ruff check` and `ruff format`.

**Relevant Context:**
- `statsmodels` may already be installed; check `requirements.txt`.
- Use `numpy.polyfit(age_vals, on_vals, 1)` per patient group for the
  lightweight OLS path.
- `on` has 29.6 % missingness — drop NaNs within each patient group before
  fitting the per-patient slope.
- Patient groups with only 1 non-NaN `on` visit cannot yield a slope; use 0
  as fallback.

---

## Sub-Task 5 — Implement `DualBranchGBDT` estimator

**Status:** `[ ] pending`

**Intent:**
Build a custom sklearn-compatible regressor in `src/parkinson/dual_branch.py`
that routes samples through two specialized `LGBMRegressor` estimators at
fit and predict time based on whether the `off` column is observed (non-NaN).

**Architecture:**
- **Branch A** (off observed): trained only on rows where `off` is non-NaN.
  Leverages the `on_off_gap`, `off` value, and Bateman features directly.
- **Branch B** (off missing): trained only on rows where `off` is NaN.
  Relies on `on`, Bateman dynamics, trajectory priors, and rater leniency.
- At predict time: rows are routed to the appropriate branch by the same
  `off_is_missing` indicator.

**Fit strategy:**
- `Branch A` receives `X[off_observed_mask]` and `y[off_observed_mask]`.
- `Branch B` receives `X[~off_observed_mask]` and `y[~off_observed_mask]`.
- Both branches are independently fitted `LGBMRegressor` instances.

**Predict strategy:**
- For each row, call the appropriate branch's `.predict()`.
- Concatenate results preserving original row order.

**Class signature:**
```python
class DualBranchGBDT(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        off_col: str = "off",
        **lgbm_params,
    ) -> None: ...
```

Store `branch_a_: LGBMRegressor`, `branch_b_: LGBMRegressor` after fit.

**Note on sklearn pipeline compatibility:** The `DualBranchGBDT` receives
a numeric `X` (after `TableVectorizer`). But we need the raw `off` column to
determine the routing mask. Two strategies:
1. Pass `off_is_missing` as a column to the estimator (already engineered by
   `build_features`) — use this column as the routing key rather than the raw
   `off` column itself. This is safer inside a pipeline.

Use `off_is_missing` column (int 0/1) as the routing indicator. The parameter
becomes `off_missing_col: str = "off_is_missing"`.

**Todo List:**
1. Create `src/parkinson/dual_branch.py`.
2. Implement `DualBranchGBDT(BaseEstimator, RegressorMixin)` with
   `off_missing_col: str = "off_is_missing"` and `**lgbm_params` forwarded
   to both branches.
3. `.fit(X, y)`: split by `off_is_missing`, fit two `LGBMRegressor`s, store
   `branch_a_`, `branch_b_`, `n_branch_a_`, `n_branch_b_`.
4. `.predict(X)`: route by `off_is_missing`, combine, return array in original
   row order.
5. Validate that the `off_is_missing` column is present in X; raise
   `ValueError` if not.
6. Add `get_params` / `set_params` via `BaseEstimator`.
7. Run `ruff check` and `ruff format`.

**Relevant Context:**
- `off_is_missing` is already computed by existing `build_features` (int 0/1).
- 42.4 % of visits have `off` missing → Branch B has ~18,900 rows; Branch A
  ~25,690 rows. Both are large enough to train a GBDT reliably.
- LightGBM handles NaN natively in the features themselves; the routing only
  uses the `off_is_missing` indicator, not the raw `off` value.
- Beware: inside the pipeline, `TableVectorizer` runs before `DualBranchGBDT`.
  After vectorization, column names may differ from the original DataFrame.
  The `off_is_missing` column name should be preserved by `TableVectorizer`
  (it's a numeric int column). Verify this during implementation; adjust the
  lookup if needed.

---

## Sub-Task 6 — Create `experiments/08_clinical_gbdt.py`

**Status:** `[ ] pending`

**Intent:**
Write the experiment script that assembles all new components into an
end-to-end pipeline, evaluates on 5-fold GroupKFold, computes the three target
RMSE metrics, pushes the report to Skore Hub, and writes the submission CSV.

**Expected Outcomes:**
- `experiments/08_clinical_gbdt.py` runs end-to-end without error.
- Pooled RMSE, observed-OFF RMSE, and missing-OFF RMSE are printed and logged.
- Report pushed to Hub as `"08_clinical_gbdt"`.
- `submissions/08_clinical_gbdt.csv` written.
- `run()` function implemented for `00_best_model.py` compatibility.

**Pipeline structure:**
```
build_features(df)                   # stateless feature engineering
  → RaterCalibrator                  # fitted target-encoder for rater_id
  → PatientTrajectoryTransformer     # fitted longitudinal stats
  → TableVectorizer                  # encodes categoricals, passes numerics
  → DualBranchGBDT                   # dual-branch LightGBM routing on off_is_missing
```

Note: `RaterCalibrator` and `PatientTrajectoryTransformer` are fitted
transformers that need `y` during fit. Since standard sklearn `Pipeline`
only passes `y` to the final estimator, these two transformers should be
applied **outside** the sklearn pipeline (inline before building X) OR wrapped
as pipeline steps with `fit_transform` compatibility that accepts `y`.

**Recommended approach:** Apply `RaterCalibrator` and
`PatientTrajectoryTransformer` as pre-processing steps that fit and transform
within each fold of a manual cross-validation loop, then pass the
transformed X to a `make_pipeline(TableVectorizer(), DualBranchGBDT(...))`.
This respects fold boundaries correctly.

For the `skore.evaluate` call (which requires a single estimator): use the
`make_pipeline(TableVectorizer(), DualBranchGBDT(...))` as the model argument.
Apply `RaterCalibrator` and `PatientTrajectoryTransformer` transforms to the
full training data before calling `evaluate` (this is slightly optimistic for
the trajectory transformer but acceptable since these transformers use only
`X` features, not `y` directly for `PatientTrajectoryTransformer`). For
`RaterCalibrator`, compute rater means on full train — acceptable for a
competition context where the goal is test RMSE.

**Three RMSE metrics (manual fold loop):**
```python
for train_idx, val_idx in cv_splits:
    X_tr, X_val = ...
    y_tr, y_val = ...
    # fit rater calibrator + trajectory transformer on train fold
    # transform both folds
    # fit pipeline on transformed X_tr
    # predict on transformed X_val
    # compute: pooled RMSE, observed-OFF RMSE (off_is_missing==0), missing-OFF RMSE (off_is_missing==1)
```

**Todo List:**
1. Create `experiments/08_clinical_gbdt.py` as a jupytext `# %%` script.
2. Load data, apply `build_features` to train+test.
3. Build `cv_splits` with `GroupKFold(n_splits=5)`.
4. Run manual fold loop collecting three RMSE metrics per fold.
5. Print fold-by-fold and mean ± std for all three metrics.
6. Print comparison table vs `07_tuning.py` BEST_PARAMS results (hardcoded
   from `07_tuning.py` for reference).
7. Fit `RaterCalibrator` + `PatientTrajectoryTransformer` on full train, then
   fit `make_pipeline(TableVectorizer(), DualBranchGBDT(...))` for Hub report.
8. Call `skore.evaluate(pipeline, X_full_transformed, y, splitter=cv_splits_transformed)`.
9. Push report: `project.put("08_clinical_gbdt", report)`.
10. Fit full pipeline, predict on X_test, clip to `[0, 132]`, write
    `submissions/08_clinical_gbdt.csv`.
11. Implement `run()` function at module level.
12. Run `ruff check experiments/08_clinical_gbdt.py --fix` and `ruff format`.

**Relevant Context:**
- Experiment scripts: `experiments/07_tuning.py` (style reference)
- `src/parkinson/hub.py` — Hub login pattern
- `GroupKFold` pattern from all prior experiments
- `07_tuning.py` BEST_PARAMS: `{"learning_rate": 0.05, "max_iter": 500,
  "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.1}`
  → reference RMSE to beat

---

## Sub-Task 7 — Create `journal/08_clinical_gbdt.md` design note

**Status:** `[ ] pending`

**Intent:**
Write the design note (approved before implementation in the iterate loop)
documenting the intent, method, acceptance criteria, and recorded outcomes
for experiment 08. Also update `journal/JOURNAL.md` History table.

**Expected Outcomes:**
- `journal/08_clinical_gbdt.md` created with intent, method, acceptance
  criteria, and (after run) recorded headline RMSE.
- `journal/JOURNAL.md` History table updated with a row for `08_clinical_gbdt`.

**Todo List:**
1. Create `journal/08_clinical_gbdt.md` using the design note template from
   `journal/01_dummy.md` as a style guide.
2. Fill in: intent (clinical feature engineering + dual-branch GBDT), method
   (four feature families + RaterCalibrator + PatientTrajectoryTransformer +
   DualBranchGBDT), acceptance criteria (pooled RMSE < 07_tuning, per-branch
   RMSE reported).
3. After the experiment runs, record the three RMSE metrics in the design note.
4. Add a row to the `## History` table in `journal/JOURNAL.md`.

**Relevant Context:**
- `journal/01_dummy.md` — style reference.
- `journal/JOURNAL.md` — index to update.

---

## Sub-Task 8 — Run setup checks and validate end-to-end

**Status:** `[ ] pending`

**Intent:**
Confirm the full implementation is correct by running the mandatory setup
checks, then executing `experiments/08_clinical_gbdt.py` end-to-end.

**Expected Outcomes:**
- `python -m pytest tests/test_setup.py` — all tests pass.
- `python experiments/08_clinical_gbdt.py` runs to completion, prints all
  three RMSE metrics, pushes to Hub, writes CSV.
- `ruff check src/parkinson/ experiments/08_clinical_gbdt.py` — zero errors.

**Todo List:**
1. Run `.venv/bin/python -m pytest tests/test_setup.py`.
2. Run `.venv/bin/python experiments/08_clinical_gbdt.py`.
3. Run `.venv/bin/python -m ruff check src/parkinson/ experiments/08_clinical_gbdt.py`.
4. Record final metrics in `journal/08_clinical_gbdt.md`.

---

## Implementation Order

```
Sub-Task 1 → Sub-Task 2 → Sub-Task 3 → Sub-Task 4 → Sub-Task 5 → Sub-Task 7 → Sub-Task 6 → Sub-Task 8
```

Sub-Tasks 2–5 can be implemented mostly in parallel (they produce independent
modules), but Sub-Task 6 depends on all of them being complete. Sub-Task 7
(design note) should be drafted early and finalized alongside Sub-Task 6.
