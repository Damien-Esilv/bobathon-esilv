# Day 2 Plan — Parkinson's RMSE Minimisation

## Top-Level Overview

**Goal:** Build and iterate on real regression models to beat the dummy baseline
(RMSE ≈ 16.48) on the Kaggle Parkinson's motor-score challenge.

**Approach:** Seven consecutive experiment scripts (`experiments/02_*.py` –
`experiments/07_*.py`) follow the DAY2.md protocol. Each experiment:
1. Evaluates with `GroupKFold(n_splits=5)` grouped by `patient_id` (no patient
   leakage, matches Kaggle holdout).
2. Pushes a Skore Hub report with a unique key.
3. Writes `submissions/<key>.csv` in `Index,target` format.
4. **Exposes a `run()` function** at module level so `00_best_model.py` can
   delegate to it with a single import.

`experiments/00_best_model.py` is the stable entry point for the final
submission. It contains **no model code** — only one import line pointing to
the current winner, and a call to `run()`. To swap the winner, change that one
line. The file is numbered `00` so it sorts first and is always easy to find.

**Scope constraints:**
- No random row splits for evaluation – only GroupKFold cv_splits.
- Missingness (especially `off`) is a clinically meaningful signal; impute only
  where the model architecture requires it.
- Do not touch `X_train`, `y_train`, `X_test` in place; always derive `visits`
  by merging and building a fresh `X / y` from it.
- Predictions clipped to `[0, 132]` (valid MDS-UPDRS range).
- Hub workspace: `parKINGson`, project: `bobathon-esilv`.

**Starting point:** `experiments/01_dummy.py` is complete (RMSE = 16.48). The
FEATURE_COLS pattern, Hub login pattern, and submission-writing pattern from
that file are reused throughout.

---

## Sub-Task 1 — Ridge Baseline with GroupKFold Alpha Sweep

**Status:** `[ ] pending`

**Intent:**
Establish the first real linear model. Median-impute numeric features (Ridge
cannot ingest NaN), sweep alpha values, push the best model to Hub, and
write the submission CSV.

**Expected Outcomes:**
- `experiments/02_ridge.py` created and runnable.
- GroupKFold RMSE printed per fold and as mean ± std for each alpha.
- Best alpha selected; report pushed to Hub as `"02_ridge"`.
- `submissions/02_ridge.csv` written with `Index,target` columns.
- RMSE meaningfully below 16.48 (expect ~8% improvement per DAY1 workbook hint).

**Todo List:**
1. Load `data/X_train.csv`, `data/y_train.csv`, `data/X_test.csv`.
2. Merge train on `Index` → `visits`.
3. Define `FEATURE_COLS` (same 8 numeric cols as 01_dummy).
4. Define `groups = visits["patient_id"]`.
5. Pre-compute `cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))`.
6. Loop over `alpha_values = [0.1, 1.0, 10.0, 100.0, 1000.0]`:
   - Build `make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=alpha))`.
   - Call `evaluate(ridge, X, y, splitter=cv_splits)`.
   - Record `report.metrics.rmse()`.
7. Select best alpha; push final report with `project.put("02_ridge", report)`.
8. `clone(best_ridge).fit(X, y)`, predict on `X_test[FEATURE_COLS]`, clip to
   `[0, 132]`, write `submissions/02_ridge.csv`.

**Relevant Context:**
- Pattern: [`experiments/01_dummy.py`](experiments/01_dummy.py)
- Hub login: [`src/parkinson/hub.py`](src/parkinson/hub.py)
- DAY2.md §2 code snippet for submission writing.
- `skore.evaluate` with `splitter=cv_splits` (pre-computed list) → `CrossValidationReport`.

---

## Sub-Task 2 — HistGradientBoosting with Native NaN Handling

**Status:** `[ ] pending`

**Intent:**
Replace median imputation with HGBR's native NaN routing. Missingness in
`off`, `ledd`, and timing columns is a direct pathological signal (skipped OFF
exams → worse disease state). Keeping NaN preserves that signal.

**Expected Outcomes:**
- `experiments/03_hgbr.py` created and runnable.
- GroupKFold RMSE printed fold-by-fold with explicit comparison to Ridge (same
  `cv_splits`).
- Report pushed as `"03_hgbr"`.
- `submissions/03_hgbr.csv` written.
- Expect measurable RMSE improvement over Ridge.

**Todo List:**
1. Re-use same data loading and `cv_splits` pattern (or import from shared util).
2. Build `HistGradientBoostingRegressor(random_state=42)` — **no imputer step**.
3. Evaluate with `cv_splits` via `evaluate(hgbr, X, y, splitter=cv_splits)`.
4. Compare fold-by-fold RMSE with Ridge results stored in Sub-Task 1.
5. Push report: `project.put("03_hgbr", report)`.
6. Fit on full train, predict on `X_test[FEATURE_COLS]`, clip `[0, 132]`, write
   `submissions/03_hgbr.csv`.

**Relevant Context:**
- DAY2.md §4 explains NaN-as-signal rationale.
- `HistGradientBoostingRegressor` natively handles `float` NaN; no need to
  convert dtypes unless adding `category` columns.
- `categorical_features="from_dtype"` is the default; numeric columns pass as-is.

---

## Sub-Task 3 — skrub tabular_pipeline with Categorical Features

**Status:** `[ ] pending`

**Intent:**
Add the string columns `gene` and `cohort` that Ridge and numeric-only HGBR
ignored. `skrub.tabular_pipeline("regressor")` or `TableVectorizer` + HGBR
encodes them without manual one-hot / label-encoding.

**Expected Outcomes:**
- `experiments/04_skrub_tabular.py` created and runnable.
- `X_full = visits.drop(columns=["Index", "patient_id", "target"])` used.
- GroupKFold RMSE reported; comparison with Sub-Task 2.
- Report pushed as `"04_skrub_tabular"`.
- `submissions/04_skrub_tabular.csv` written.

**Todo List:**
1. Build `X_full` by dropping `Index`, `patient_id`, `target` from `visits`.
2. Re-compute `cv_splits` with `X_full` shape (same `groups` array).
3. Build model: `tabular_pipeline("regressor")` (or `make_pipeline(TableVectorizer(), HistGradientBoostingRegressor(random_state=42))`).
4. Evaluate: `evaluate(model, X_full, y, splitter=cv_splits)`.
5. Push: `project.put("04_skrub_tabular", report)`.
6. Fit on full `X_full`, predict on `X_test.drop(columns=["Index","patient_id"], errors="ignore")`,
   clip `[0, 132]`, write `submissions/04_skrub_tabular.csv`.

**Relevant Context:**
- DAY2.md §5 code snippet and cardinality strategy.
- `TableVectorizer` picks per-column: one-hot for low cardinality (e.g. `cohort`,
  `sexM`), `StringEncoder` for high-cardinality strings (e.g. `gene` if many
  variants), passthrough for numerics.
- Alignment note: test columns must match train columns exactly (same drops,
  same dtypes).

---

## Sub-Task 4 — skrub DataOps Graph with Baked GroupKFold

**Status:** `[ ] pending`

**Intent:**
Demonstrate the DataOps pattern where `groups` cannot drift away from the data.
The split lives on the graph, not as a side list. Also validates that the
`SkrubLearner` can predict on `X_test` (no `target` column).

**Expected Outcomes:**
- `experiments/05_dataops.py` created and runnable.
- Report pushed as `"05_dataops"`.
- Scores should match Sub-Task 3 (same model, same CV; validate consistency).
- `submissions/05_dataops.csv` written using `learner.predict({"visits": X_test})`.

**Todo List:**
1. Build the DataOp graph:
   ```
   data = skrub.var("visits", visits)
   groups = data["patient_id"]
   X_op = data.drop("target", axis=1).skb.mark_as_X(
       cv=GroupKFold(n_splits=5),
       split_kwargs={"groups": groups},
   )
   y_op = data["target"].skb.mark_as_y()
   pred = X_op.skb.apply(TableVectorizer()).skb.apply(
       HistGradientBoostingRegressor(random_state=42), y=y_op
   )
   ```
2. Call `evaluate(pred)` — no `splitter=` argument; CV is baked in.
3. Push: `project.put("05_dataops", report)`.
4. Freeze: `learner = pred.skb.make_learner()`.
5. Fit: `learner.fit({"visits": visits})`.
6. Predict: `pred_test = learner.predict({"visits": X_test})`, clip `[0, 132]`.
7. Write `submissions/05_dataops.csv`.

**Relevant Context:**
- DAY2.md §6 full code snippet.
- Key difference from Sub-Task 3: `X_op` is derived from `data` which still
  contains `patient_id`; the `drop("target")` happens inside the graph. Ensure
  `Index` and `patient_id` are also dropped before the vectorizer (the
  `TableVectorizer` should receive them as IDs or they should be dropped
  explicitly from the graph).
- Test table `X_test` has no `target` column; `SkrubLearner` handles this
  because `y_op` is only used during fit.

---

## Sub-Task 5 — Clinical Feature Engineering

**Status:** `[ ] pending`

**Intent:**
Encode domain knowledge from CONTEXT.md: disease progression timeline,
levodopa pharmacodynamics, and explicit missingness indicators. These
features can unlock non-linear signal that HGBR and Ridge both miss.

**Expected Outcomes:**
- `src/parkinson/features.py` created with a `build_features(df)` function.
- `experiments/06_features.py` created applying these features.
- GroupKFold RMSE compared to Sub-Task 3/4 baseline.
- Report pushed as `"06_features"`.
- `submissions/06_features.csv` written.

**Feature Engineering Steps (inside `build_features`):**
1. `disease_duration = age - age_at_diagnosis`.
2. `visit_count` per patient (number of visits by `patient_id`).
3. `visit_order` (rank of visit within each patient, sorted by chronological
   index or a date column if available).
4. `time_since_intake_on_sq = time_since_intake_on ** 2` (non-linear absorption).
5. `time_since_intake_off_sq = time_since_intake_off ** 2`.
6. `on_off_gap = off - on` (motor fluctuation size; NaN when either is missing).
7. `dose_timing = ledd / (1.0 + time_since_intake_on)` (pharmacodynamic
   interaction; NaN when `ledd` or `time_since_intake_on` missing).
8. `off_is_missing = off.isna().astype(int)`.
9. `ledd_is_missing = ledd.isna().astype(int)`.

**Todo List:**
1. Create `src/parkinson/features.py` with `build_features(df: pd.DataFrame) → pd.DataFrame`.
2. Apply `build_features` to `visits` and to `X_test` (same function, same
   transformations, no fitting required).
3. Build enriched `X_full` including new features (keep original + engineered).
4. Recompute `cv_splits` on new `X_full`.
5. Evaluate `tabular_pipeline("regressor")` or `TableVectorizer` + HGBR on
   enriched features.
6. Push: `project.put("06_features", report)`.
7. Write `submissions/06_features.csv`.

**Relevant Context:**
- CONTEXT.md pharmacokinetic section: levodopa timing, ON/OFF fluctuation,
  disease progression.
- `visit_order` may require sorting by `Index` within each `patient_id` group
  as a proxy for chronological order (no explicit date column).
- `build_features` must be callable on both train and test DataFrames and
  must not use target information.

---

## Sub-Task 6 — Hyperparameter Tuning and Algorithm Comparison

**Status:** `[ ] pending`

**Intent:**
Systematically tune HGBR and test LightGBM and CatBoost (if installed).
Produce a stacked or blended ensemble from the best heterogeneous models.

**Expected Outcomes:**
- `experiments/07_tuning.py` created.
- HGBR tuned; LightGBM and/or CatBoost compared on same `cv_splits`.
- Best individual model and an ensemble (stacking or weighted blending) evaluated.
- Comparison report pushed as `"07_tuning"`.
- Best single model and ensemble RMSE recorded in JOURNAL.md.

**Todo List:**
1. Tune HGBR over: `learning_rate` [0.05, 0.1], `max_iter` [200, 500],
   `max_leaf_nodes` [15, 31], `min_samples_leaf` [10, 20],
   `l2_regularization` [0, 0.1]. Use `RandomizedSearchCV` or manual loop
   with `cv_splits`.
2. Try `LGBMRegressor` if `lightgbm` importable; skip gracefully if not.
3. Try `CatBoostRegressor` if `catboost` importable; skip gracefully if not.
4. Collect all models with GroupKFold RMSE; print comparison table.
5. Build an ensemble: `StackingRegressor` with the 2–3 best heterogeneous
   models as base estimators and `Ridge` as final estimator (or weighted
   average blending using held-out fold predictions).
6. Evaluate ensemble with `cv_splits`.
7. Push: `project.put("07_tuning", report)`.
8. Record results in `journal/JOURNAL.md`.

**Relevant Context:**
- `StackingRegressor` from `sklearn.ensemble` uses `cv` parameter for base-
  estimator fold generation; ensure it also uses `GroupKFold`-compatible
  approach or pre-split.
- LightGBM and CatBoost both handle NaN natively; no imputer needed.
- `catboost` requires `verbose=0` to suppress training output.
- Check imports with `try/except ImportError` to keep the script portable.

---

## Sub-Task 7 — Stable Best-Submission Entry Point

**Status:** `[ ] pending`

**Intent:**
Create `experiments/00_best_model.py` — a **thin dispatcher** that imports
`run` from whichever numbered experiment currently holds the lowest GroupKFold
RMSE and calls it. No model code lives here; the file's only job is to express
"this experiment is the winner right now." Swapping the winner later means
editing exactly one import line.

**Expected Outcomes:**
- `experiments/00_best_model.py` created (≤ 15 lines of code).
- Every numbered experiment (`02_ridge.py`, `03_hgbr.py`, …) exposes a
  `run()` function that: fits the pipeline on full training data, predicts on
  `X_test`, clips to `[0, 132]`, writes `submissions/best_submission.csv`,
  and pushes to Hub as `"best_submission"`.
- Running `python experiments/00_best_model.py` produces `submissions/best_submission.csv`.
- Changing the winner = changing one import line in `00_best_model.py`.

**Todo List:**
1. In **each** numbered experiment file (`02_ridge.py` through `07_tuning.py`),
   wrap the "fit full train → predict test → write CSV → push Hub" block in a
   `def run() -> None:` function. The per-experiment evaluation and Hub report
   sections remain at module level (so running the file standalone still works).
2. Create `experiments/00_best_model.py` with exactly this structure:
   ```python
   # Change this import to point at the current winner.
   from experiments.XX_name import run

   if __name__ == "__main__":
       run()
   ```
3. After Sub-Tasks 1–6 are complete, identify the experiment with the lowest
   GroupKFold RMSE mean and set the import in `00_best_model.py` accordingly.
4. Run `python experiments/00_best_model.py` to produce the final
   `submissions/best_submission.csv`.
5. Push: the `run()` function of the winning experiment calls
   `project.put("best_submission", report)`.
6. Update `journal/JOURNAL.md` with final RMSE and the winning experiment key.

**Relevant Context:**
- `00_best_model.py` sorts alphabetically before `01_dummy.py` — it is
  immediately visible at the top of the `experiments/` directory.
- The `run()` convention must be respected in every experiment file; the module
  level still runs evaluation + Hub push under `if __name__ == "__main__"` (or
  unconditionally, matching the style of `01_dummy.py`).
- Submission template: `data/sample_submission.csv` (header: `Index,target`,
  11,014 data rows).
- Clip range: MDS-UPDRS Part III motor score maximum = 132 (18 items × 4).

---

## Implementation Notes

### File Naming Convention

Each numbered experiment matches its own Skore report key and CSV.
`00_best_model.py` is the stable dispatcher — its Hub key and CSV are always
`best_submission`, regardless of which experiment it delegates to.

| File | Report Key | Submission |
|---|---|---|
| `experiments/00_best_model.py` | `best_submission` | `submissions/best_submission.csv` |
| `experiments/02_ridge.py` | `02_ridge` | `submissions/02_ridge.csv` |
| `experiments/03_hgbr.py` | `03_hgbr` | `submissions/03_hgbr.csv` |
| `experiments/04_skrub_tabular.py` | `04_skrub_tabular` | `submissions/04_skrub_tabular.csv` |
| `experiments/05_dataops.py` | `05_dataops` | `submissions/05_dataops.csv` |
| `experiments/06_features.py` | `06_features` | `submissions/06_features.csv` |
| `experiments/07_tuning.py` | `07_tuning` | `submissions/07_tuning.csv` |

### `run()` Function Contract

Every numbered experiment must expose:

```python
def run() -> None:
    """Fit on full training data, predict X_test, write best_submission.csv,
    push Hub report as 'best_submission'."""
    ...
```

`00_best_model.py` structure (change only the import line to swap winner):

```python
# %%
# Point this import at the experiment with the lowest GroupKFold RMSE.
from experiments.XX_name import run  # ← change this line only

if __name__ == "__main__":
    run()
```

### GroupKFold Pattern (reused in every experiment)
```python
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
report = evaluate(model, X, y, splitter=cv_splits)
```

### Hub Push Pattern (from 01_dummy.py)
```python
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("<key>", report)
```

### Submission Write Pattern
```python
preds = np.clip(model.predict(X_test[feature_cols]), 0, 132)
pd.DataFrame({"Index": X_test["Index"], "target": preds}).to_csv(
    PROJECT_ROOT / "submissions" / "<key>.csv", index=False
)
```
