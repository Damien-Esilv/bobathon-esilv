# 09_sequence_transformer — Design Note

## Intent

Exploit the retrospective smoothing structure of the competition: for every
patient in both train and test, the model has simultaneous access to the full
visit timeline. When an OFF score is missing at visit t, its true value is
constrained by visits at t−1 and t+1. A bi-directional self-attention model
(`SeqNet`) captures these temporal constraints explicitly, where the IID GBDT
baseline cannot.

The key design choices:
- **Anchored residual formulation**: predict the deviation from a medication
  baseline (`off` if observed, else `on`) rather than the raw score. This
  leverages the strongest single predictor (off, r=0.87) as a prior and asks
  the model only to debias it.
- **Stochastic OFF masking (p=0.35)** during training forces the model to
  reconstruct masked OFF values from neighboring visits, directly targeting the
  42 % missing-OFF regime.
- **No weight EMA**: AdamW + OneCycleLR cosine annealing is sufficient.

## Method

### Feature set

Raw columns only (not `build_features` aggregates):

- **Continuous (z-scored, 12 dims)**: age, age_at_diagnosis, ledd,
  time_since_intake_on, time_since_intake_off, on_filled, off_filled, base,
  sexM, off_is_missing, on_is_missing, ledd_is_missing.
- **Categorical (embedded)**: cohort (dim=4), gene (dim=8), rater_id (dim=16).
- **Temporal**: Δage sinusoidal embedding (d_time=16).

### Architecture (`SeqNet`, `src/parkinson/seqnet.py`)

```
Continuous → Linear(12, 84)                         → (B, T, 84)
Categorical → Embeddings(cohort/gene/rater) concat  → (B, T, 28)
Δage → SinusoidalTemporalEmbedding                  → (B, T, 16)
─────────────────────────────────────────────────────
Token concat                                         → (B, T, 128)
TransformerEncoder (2 layers, 4 heads, pre-norm)     → (B, T, 128)
Linear(128, 1) → r̂                                  → (B, T)
ŷ = base + 20 × r̂                                   → (B, T)
```

### Training

- 5-fold GroupKFold by patient_id.
- AdamW(lr=1e-3, weight_decay=1e-4).
- OneCycleLR(max_lr=1e-3, epochs=30).
- MSE loss on non-padding, valid-target positions.
- Stochastic OFF masking (p=0.35) active during training only.
- Device: MPS → CUDA → CPU (auto-selected).

### Evaluation metrics

Three RMSE values per fold:
1. Pooled RMSE (all non-padding visits).
2. Branch-A RMSE (visits where OFF was originally observed).
3. Branch-B RMSE (visits where OFF was originally missing).

Comparison baseline: experiment 08 pooled RMSE = **6.040**.

## Acceptance Criteria

- [x] `experiments/09_sequence_transformer.py` runs end-to-end.
- [x] Three RMSE metrics printed per fold.
- [x] Report pushed to Hub as `"09_sequence_transformer"`.
- [x] `submissions/09_sequence_transformer.csv` written (11 013 rows).
- [x] `00_best_model.py` updated — WINNER = `09_sequence_transformer.py`.
- [x] `tests/test_seqnet.py` passes (5 unit tests).
- [x] All new modules pass `ruff check` with zero errors.

## Results

| Metric | Mean (5-fold) | Std |
|---|---|---|
| Pooled RMSE | **3.308** | 0.052 |
| Branch-A RMSE (off observed) | **2.938** | 0.077 |
| Branch-B RMSE (off missing) | **3.751** | 0.078 |
| vs 08_clinical_gbdt (6.040) | **−2.732 (−45.2%)** | — |

**Per-fold breakdown:**

| Fold | Pooled | Branch-A | Branch-B |
|---|---|---|---|
| 1 | 3.407 | 3.065 | 3.828 |
| 2 | 3.264 | 2.890 | 3.718 |
| 3 | 3.271 | 2.989 | 3.618 |
| 4 | 3.294 | 2.884 | 3.767 |
| 5 | 3.303 | 2.864 | 3.823 |

**Winner update:** `00_best_model.py` updated to `WINNER = "09_sequence_transformer.py"`.
`submissions/best_submission.csv` regenerated from `09_sequence_transformer.csv`.

Skore Hub: `"09_sequence_transformer"` key pushed during full-train run.
