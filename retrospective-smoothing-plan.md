# Retrospective Smoothing Ensemble Plan — Experiment 10

## Top-Level Overview

**Goal:** Combine the complementary strengths of `08_clinical_gbdt` (sharp
tabular non-linearities, pharmacokinetic features) and `09_sequence_transformer`
(retrospective trajectory smoothing via bi-directional self-attention) under a
calibrated dual-regime ensemble: `RetrospectiveSmoothingRegressor`.

**Baseline to beat:** `09_sequence_transformer` pooled 5-fold RMSE = **3.308**.

**Design:** Per-regime convex combination of the two model predictions, with
blend weights `(α_obs, α_miss)` optimized on pooled OOF predictions:

```
ŷ_final = α * ŷ_SeqNet + (1 − α) * ŷ_GBDT
  where α = α_obs  if off was observed
            α_miss if off was missing
```

**Files created:**
- `src/parkinson/pipeline.py` — `RetrospectiveSmoothingRegressor`
- `experiments/10_retrospective_smoothing.py` — CV + blending + bootstrap
- `tests/test_pipeline.py` — unit tests
- `journal/10_retrospective_smoothing.md` — design note
- `journal/JOURNAL.md` — updated History table

---

## Architecture

```
fit(X_raw, y):
  ┌─────────────────────────────────┐
  │  GBDT arm                       │
  │  build_features(X) →            │
  │  RaterCalibrator →              │
  │  PatientTrajectory →            │
  │  TableVectorizer →              │
  │  DualBranchGBDT.fit()           │
  └─────────────────────────────────┘
  ┌─────────────────────────────────┐
  │  SeqNet arm                     │
  │  prepare_dataframe(X) →         │
  │  VisitNormalizer →              │
  │  CategoricalEncoder →           │
  │  SeqNet.train()                 │
  └─────────────────────────────────┘
  Both arms store their fitted state.

predict(X_raw):
  ŷ_gbdt  = GBDT arm pipeline
  ŷ_seqnet = SeqNet arm (eval mode)
  off_is_missing = X_raw["off"].isna()
  α = α_obs where ~off_is_missing
  α = α_miss where off_is_missing
  return clip(α * ŷ_seqnet + (1−α) * ŷ_gbdt, 0, 132)
```

**Key design decision:** `RetrospectiveSmoothingRegressor` takes `alpha_obs`
and `alpha_miss` as constructor arguments (defaulting to pre-optimized values).
The experiment script optimizes these weights from OOF predictions and then
passes them when constructing the final fitted ensemble.

---

## Sub-Task 1 — Implement `src/parkinson/pipeline.py`

**Status:** `[ ] pending`

**Intent:** A single sklearn-compatible class that owns both model arms and the
blending logic. All internal preprocessing is encapsulated so `fit(X, y)` and
`predict(X)` receive the raw `X_train` / `X_test` DataFrames.

**Expected Outcomes:**
- `RetrospectiveSmoothingRegressor` imports without error.
- `fit(X, y)` trains both arms end-to-end.
- `predict(X)` returns a numpy array clipped to [0, 132].
- `alpha_obs` and `alpha_miss` are constructor parameters with sensible defaults.

**Class signature:**
```python
class RetrospectiveSmoothingRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        alpha_obs: float = 0.78,
        alpha_miss: float = 0.76,
        # GBDT hyperparameters
        lgbm_params: dict | None = None,
        # SeqNet hyperparameters
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        mask_prob: float = 0.35,
        residual_scale: float = 20.0,
        d_model: int = 128,
        d_time: int = 16,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        random_state: int = 42,
    ) -> None: ...
```

**Stored fitted state:**
- `gbdt_pipeline_`: `sklearn.pipeline.Pipeline` (TableVectorizer + DualBranchGBDT)
- `rater_calibrator_`: fitted `RaterCalibrator`
- `trajectory_transformer_`: fitted `PatientTrajectoryTransformer`
- `seqnet_model_`: trained `SeqNet` (kept on CPU after training)
- `seqnet_norm_`: fitted `VisitNormalizer`
- `seqnet_enc_`: fitted `CategoricalEncoder`
- `device_`: torch.device (auto-selected)

**Internal flow for `fit(X, y)`:**

```
1. GBDT arm:
   a. Apply build_features(X.merge(y)) — stateless
   b. Fit RaterCalibrator on training data (stores rater_calibrator_)
   c. Fit PatientTrajectoryTransformer on RC-transformed data
   d. Drop patient_id, fit TableVectorizer + DualBranchGBDT (stores gbdt_pipeline_)

2. SeqNet arm:
   a. Fit VisitNormalizer and CategoricalEncoder on prepare_dataframe(X, y)
   b. Build DataLoader and train SeqNet (30 epochs, OneCycleLR)
   c. Store seqnet_model_ in eval() mode on CPU
```

**Internal flow for `predict(X)`:**

```
1. GBDT arm: build_features → RC.transform → PTT.transform → drop patient_id
   → gbdt_pipeline_.predict()

2. SeqNet arm: prepare_dataframe(X, None) → norm.transform → enc.transform
   → make_dataloader(augment=False) → seqnet_model_.predict (no_grad)

3. Blend: determine off_is_missing from X["off"].isna()
   → per-visit α selection → convex combination → clip [0, 132]
```

**Todo List:**
1. Create `src/parkinson/pipeline.py`.
2. Import all required components from existing modules.
3. Implement `RetrospectiveSmoothingRegressor.__init__`.
4. Implement `_fit_gbdt_arm(X, y)` private method.
5. Implement `_fit_seqnet_arm(X, y)` private method.
6. Implement `fit(X, y)` calling both arms.
7. Implement `_predict_gbdt(X)` private method.
8. Implement `_predict_seqnet(X)` private method.
9. Implement `predict(X)` with regime blending and clipping.
10. Run `ruff check src/parkinson/pipeline.py --fix` and `ruff format`.

**Relevant Context:**
- GBDT preprocessing pattern: `experiments/08_clinical_gbdt.py` lines 121–169
- SeqNet training pattern: `experiments/09_sequence_transformer.py` lines 95–330
- Both arms need `patient_id` during preprocessing — `X` must retain it
- `predict()` reads `X["off"].isna()` to determine the blending regime

---

## Sub-Task 2 — Implement `experiments/10_retrospective_smoothing.py`

**Status:** `[ ] pending`

**Intent:** The experiment script that drives the full CV-based weight
optimization and final production submission. This is the orchestrator; it does
NOT train `RetrospectiveSmoothingRegressor` directly during CV — instead it
trains the two arms independently per fold (reusing helpers from experiments 08
and 09), collects pooled OOF predictions from each, optimizes weights, and then
trains the final ensemble on full data with the optimized weights.

**Expected Outcomes:**
- Pooled OOF RMSE of the blend, both arms printed per fold.
- Optimal `(α_obs, α_miss)` printed.
- Paired bootstrap probability of superiority vs SeqNet printed.
- Report pushed to Hub as `"10_retrospective_smoothing"`.
- `submissions/10_retrospective_smoothing.csv` written (11,013 rows).
- `00_best_model.py` updated and `best_submission.csv` regenerated if blend
  RMSE < 3.308 (SeqNet baseline).

**CV Loop Design:**

```python
# For each fold:
#   1. Train GBDT arm on training fold, collect val OOF predictions.
#   2. Train SeqNet arm on training fold, collect val OOF predictions.
#   Both are the SAME splits (same GroupKFold seed) so arrays align.
#
# After all folds:
#   3. Stack pooled OOF arrays: preds_gbdt, preds_seqnet, targets, off_missing
#   4. Optimize α_obs, α_miss on the pooled arrays via grid search.
#   5. Compute blend RMSE with optimal weights.
#   6. Paired bootstrap test vs SeqNet pooled RMSE.
```

**Weight Optimization:**

```python
def _optimize_weights(
    preds_gbdt: np.ndarray,
    preds_seqnet: np.ndarray,
    targets: np.ndarray,
    off_missing: np.ndarray,
) -> tuple[float, float]:
    """Grid search α_obs, α_miss ∈ [0, 1] in steps of 0.01."""
    best_rmse = np.inf
    best_obs, best_miss = 0.5, 0.5
    for a_obs in np.arange(0.0, 1.01, 0.01):
        for a_miss in np.arange(0.0, 1.01, 0.01):
            alpha = np.where(off_missing, a_miss, a_obs)
            blend = alpha * preds_seqnet + (1 - alpha) * preds_gbdt
            blend = np.clip(blend, 0, 132)
            rmse = np.sqrt(np.mean((blend - targets) ** 2))
            if rmse < best_rmse:
                best_rmse, best_obs, best_miss = rmse, a_obs, a_miss
    return float(best_obs), float(best_miss)
```

**Bootstrap Test:**

```python
def _bootstrap_superiority(
    preds_blend: np.ndarray,
    preds_seqnet: np.ndarray,
    targets: np.ndarray,
    patient_ids: np.ndarray,   # aligned with preds arrays
    n_bootstrap: int = 10_000,
    rng_seed: int = 42,
) -> float:
    """Clustered bootstrap: resample patients (not visits)."""
    rng = np.random.default_rng(rng_seed)
    unique_patients = np.unique(patient_ids)
    n_wins = 0
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        mask = np.isin(patient_ids, sampled)
        rmse_blend = np.sqrt(np.mean((preds_blend[mask] - targets[mask])**2))
        rmse_seq   = np.sqrt(np.mean((preds_seqnet[mask] - targets[mask])**2))
        n_wins += rmse_blend < rmse_seq
    return n_wins / n_bootstrap
```

**`run()` function:**
- Trains `RetrospectiveSmoothingRegressor` (with optimized weights) on full data.
- Predicts on `X_test`, clips to [0, 132], writes submission.
- Pushes metrics dict to Hub.
- Updates `00_best_model.py` if blend RMSE < 3.308.

**Todo List:**
1. Create `experiments/10_retrospective_smoothing.py` as a jupytext `# %%` script.
2. Implement `_cv_collect_oof()` — runs 5-fold CV collecting OOF arrays from
   both arms. Reuses the GBDT preprocessing helpers from experiment 08 and
   SeqNet training helpers from experiment 09 (import directly from those files).
3. Implement `_optimize_weights()` — grid search.
4. Implement `_blend_predict()` — apply optimal weights, clip.
5. Implement `_bootstrap_superiority()` — clustered bootstrap test.
6. Implement main `if __name__ == "__main__"` block.
7. Implement `run()` — fit full ensemble, write submission, push Hub.
8. Run `ruff check` and `ruff format`.

**Relevant Context:**
- Import `_fit_preprocess_fold`, `_fit_preprocess_full`, `_fit_preprocess_test`,
  `_drop_for_model`, `_load_data` from `experiments/08_clinical_gbdt.py` via
  `importlib` (same pattern as `00_best_model.py`) OR copy the helper functions
  inline — prefer inline copy to avoid import-time execution of module-level
  code in 08.
- Import `_build_encoders`, `_apply_encoders`, `_train_epoch`, `_eval_fold`,
  `_train_full` from `experiments/09_sequence_transformer.py` similarly — or
  inline. **Prefer inline** to avoid import-time side effects.
- SeqNet baseline RMSE: `BASELINE_RMSE_09 = 3.308`.

---

## Sub-Task 3 — Implement `tests/test_pipeline.py`

**Status:** `[ ] pending`

**Intent:** Four unit tests verifying the ensemble's correctness without
triggering full training (use very small SeqNet, 2 training epochs, tiny data).

**Tests:**
1. `test_rsr_fits_and_predicts`: `fit` runs without error on a 20-row synthetic
   DataFrame; `predict` returns a numpy array of the correct shape.
2. `test_rsr_predictions_clipped`: All predictions lie within [0, 132] on
   synthetic test data including out-of-range raw values.
3. `test_rsr_regime_weighting`: With `alpha_obs=1.0` and `alpha_miss=0.0`, the
   blend for off-observed visits equals the SeqNet prediction (within float32
   tolerance), and for off-missing visits equals the GBDT prediction.
4. `test_rsr_predict_shape_matches_input`: Shape of `predict(X_test)` equals the
   number of rows in `X_test`.

**Todo List:**
1. Create `tests/test_pipeline.py`.
2. Add `_make_tiny_df(n_patients, visits_per_patient)` helper (can reuse from
   `test_seqnet.py` or create anew with target column included).
3. Implement the four tests with a fast constructor override:
   `RetrospectiveSmoothingRegressor(epochs=2, d_model=32, dim_feedforward=32)`.
4. Run `pytest tests/test_pipeline.py` to confirm all pass.
5. Run `ruff check tests/test_pipeline.py`.

---

## Sub-Task 4 — Write `journal/10_retrospective_smoothing.md` + update `JOURNAL.md`

**Status:** `[ ] pending`

**Todo List:**
1. Create `journal/10_retrospective_smoothing.md` with full design note.
2. Add `10_retrospective_smoothing` row to `journal/JOURNAL.md` History table.
3. After experiment runs, fill in RMSE metrics, optimal weights, bootstrap P.

---

## Sub-Task 5 — End-to-end validation

**Status:** `[ ] pending`

**Todo List:**
1. Run `.venv/bin/python -m pytest tests/test_setup.py` — all pass.
2. Run `.venv/bin/python -m pytest tests/test_pipeline.py` — all pass.
3. Run `.venv/bin/python experiments/10_retrospective_smoothing.py`.
4. Run `ruff check src/parkinson/pipeline.py experiments/10_retrospective_smoothing.py tests/test_pipeline.py`.
5. Record final metrics in `journal/10_retrospective_smoothing.md`.

---

## Implementation Order

```
ST1 (pipeline.py) → ST3 (tests) → ST4 (journal draft) → ST2 (experiment) → ST5 (e2e)
```

ST1 and ST4 are independent and can run in parallel.
ST2 depends on ST1 (imports `RetrospectiveSmoothingRegressor`).
ST3 depends on ST1.
ST5 depends on ST2 and ST3.

---

## Open Design Notes

### Why inline helpers rather than importing from 08/09?
Importing `experiments/08_clinical_gbdt.py` via `importlib` is fragile:
module-level code in that file (data loading, etc.) may execute at import time.
The cleaner approach is to copy the preprocessing helpers inline in the new
experiment, or factor them out into `src/parkinson/` utilities.

### Prediction alignment issue in `RetrospectiveSmoothingRegressor.predict`
The SeqNet predicts in patient-sorted order (by age); the GBDT predicts in
row-index order. The `predict()` method must ensure both output arrays are
aligned to the same original row order as the input `X`. The blending step
reads `X["off"].isna()` using the original index, so both prediction arrays
must be re-sorted to match `X.index` order before blending.

### OOF patient_id tracking
The bootstrap test requires `patient_ids` aligned with the pooled OOF
predictions. During the CV loop, collect `X_val["patient_id"].values` alongside
the prediction arrays.

### SeqNet `predict()` in `RetrospectiveSmoothingRegressor`
The SeqNet DataLoader sorts patients by age within each patient group. To map
predictions back to the original DataFrame row order, `_predict_seqnet()` must
track the original Index column through the DataLoader. Use the same approach
as `_predict_test()` in `09_sequence_transformer.py`: sort `df_test` by
`patient_id + age`, predict, then re-sort by `Index`.
