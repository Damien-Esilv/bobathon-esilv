# 10_retrospective_smoothing — Design Note

## Intent

Combine the complementary strengths of the two best-performing models:
- **08_clinical_gbdt** (DualBranchGBDT, RMSE 6.040): sharp tabular non-linearities, Bateman pharmacokinetic features, per-patient OLS trajectory priors, rater leniency encoding.
- **09_sequence_transformer** (SeqNet, RMSE 3.308): retrospective bi-directional self-attention smoothing the full patient visit timeline simultaneously.

These models have orthogonal inductive biases:
- The GBDT captures rich feature interactions at the visit level.
- The SeqNet propagates denoised signal across a patient's trajectory, excelling at reconstructing missing OFF states from neighboring visits.

A calibrated dual-regime ensemble blends both under regime-specific weights, giving more weight to whichever model has lower error in each missingness regime.

## Method

### Ensemble formula

```
ŷ = α * ŷ_SeqNet + (1 − α) * ŷ_GBDT
```

where:
- `α = α_obs` when the OFF clinic measurement is **observed**
- `α = α_miss` when the OFF clinic measurement is **missing**

Predictions are clipped to `[0, 132]` (physiological MDS-UPDRS Part III range).

### Weight optimization

Blend weights `(α_obs, α_miss)` are optimized on pooled out-of-fold predictions
via a fine grid search over `[0, 1] × [0, 1]` in steps of 0.01, minimizing
pooled RMSE. This respects CV fold boundaries — weights are never fit on the
same data used to generate predictions.

### Bootstrap test

A clustered bootstrap test (10,000 resamples, clustered at patient level)
computes:

```
P(RMSE_blend < RMSE_SeqNet)
```

Patient-level resampling respects within-patient correlation.

### Architecture (`RetrospectiveSmoothingRegressor`, `src/parkinson/pipeline.py`)

```
fit(X, y):
  GBDT arm: build_features → RaterCalibrator → PatientTrajectory → TableVectorizer → DualBranchGBDT
  SeqNet arm: prepare_dataframe → VisitNormalizer → CategoricalEncoder → SeqNet.train()

predict(X):
  ŷ_gbdt   = GBDT arm
  ŷ_seqnet = SeqNet arm (eval mode)
  α = α_obs where off observed, α_miss where off missing
  return clip(α * ŷ_seqnet + (1−α) * ŷ_gbdt, 0, 132)
```

## Acceptance Criteria

- [x] `experiments/10_retrospective_smoothing.py` runs end-to-end.
- [x] Pooled blend RMSE, Branch-A RMSE, Branch-B RMSE printed.
- [x] Optimal `(α_obs, α_miss)` printed.
- [x] Bootstrap P(superiority) printed.
- [x] Report pushed to Hub as `"10_retrospective_smoothing"`.
- [x] `submissions/10_retrospective_smoothing.csv` written (11 013 rows).
- [x] `00_best_model.py` updated if blend RMSE < 3.308.
- [x] `tests/test_pipeline.py` passes (4 tests).
- [x] All new modules pass `ruff check` with zero errors.

## Results

| Metric | Value |
|---|---|
| GBDT OOF RMSE | 6.0229 |
| SeqNet OOF RMSE | 3.3087 |
| Blend OOF RMSE | 3.3032 |
| Branch-A (OFF observed) | 2.9429 |
| Branch-B (OFF missing) | 3.7372 |
| α_obs | 1.00 |
| α_miss | 0.95 |
| P(blend < SeqNet) bootstrap | 1.0000 |
| vs 09_sequence_transformer (3.308) | Δ = +0.0048 (+0.1%) |

Skore Hub: `"10_retrospective_smoothing"` (workspace: `parKINGson`, project: `bobathon-esilv`)
