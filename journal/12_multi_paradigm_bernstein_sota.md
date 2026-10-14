# 12_multi_paradigm_bernstein_sota — Design Note

## Intent

Improve on experiment 11 (3.2317 OOF RMSE) by replacing the single SeqNet +
weak GBDT blend with a two-engine ensemble whose second engine (Anchored GBDT)
is properly calibrated via residual anchoring — predicted residual is added back
to a physiologically meaningful baseline rather than predicting the absolute score.

## Engines

### Engine 1 — Time-Decay SeqNet (`src/parkinson/seqnet_td.py`)

Architecture: 2 TimeDecayEncoderLayers + Bidirectional GRU + residual head.

**Key innovation — time-decay self-attention:**

```
attn_ij = (q_i · k_j) / sqrt(d_k) - softplus(log_gamma) * |t_i - t_j|
```

One learnable `log_gamma` per attention head softly penalises attending across
large age gaps without hard masking.

**Bidirectional GRU:** After the Transformer layers, a `nn.GRU(hidden_size=64,
bidirectional=True)` produces a 128-dim contextual representation per visit
by integrating both past and future trajectory context.

**Loss:** `Huber(delta=4) + 0.3 * MSE(consecutive diffs)` — the second term
penalises discontinuous jumps in consecutive visit predictions.

Hyperparameters: d_model=128, nhead=4, 2 layers, dim_ff=256, gru_hidden=64,
30 epochs, AdamW + OneCycleLR.

### Engine 2 — Anchored Dual-Branch GBDT

**Root-cause fix for exp-08/10 GBDT RMSE ≈ 6.13:** The unanchored GBDT
predicted absolute scores, which is hard because the target range spans 0–110.
Anchoring to a per-visit physiological baseline shrinks the residual range to
roughly ±15, making gradient boosting much more accurate.

**Branch A (OFF observed):**
```
y_residual = y - off
GBDT predicts Delta_A
y_hat = off + Delta_A
```

**Branch B (OFF missing):**
```
patient_anchor = mean(observed off)  [or mean(on) + 10 if no off observed]
y_residual = y - patient_anchor
GBDT predicts Delta_B
y_hat = patient_anchor + Delta_B
```

### Gating — 6-Regime SLSQP Convex Weights

Visits are assigned to 6 clinical regimes (see `assign_regimes`). Per-regime
convex mixture weights `w = [w_E1, w_E2], w >= 0, sum = 1` are fitted on
pooled OOF predictions via `scipy.optimize.minimize(method="SLSQP")` with mild
Tikhonov regularization (`lambda_reg=0.001`).

### Post-processing — Monotonic Bernstein Projector

`MonotonicBernsteinTrajectoryProjector` (exp 11 / `src/parkinson/trajectory.py`)
is applied after gating to enforce per-patient non-decreasing predictions.

## Pipeline

```
5-fold GroupKFold
  for each fold:
    Engine 1: TD-SeqNet.fit(X_tr) → OOF preds_e1
    Engine 2: AnchoredGBDT.fit(X_tr) → OOF preds_e2
  ↓
  6-regime SLSQP gating weights (on pooled OOF)
  ↓
  preds_blend = sum_r(w_r1 * preds_e1 + w_r2 * preds_e2)
  ↓
  MonotonicBernsteinTrajectoryProjector.transform_predictions(df_oof, preds_blend)
  ↓
  Bernstein OOF RMSE
```

## Alignment contract

`df_oof = X_train.iloc[oof["val_indices"]]` — constructed from the concatenated
positional `val_idx` arrays (same order as OOF prediction arrays).  No global
pre-sort of `X_train`.  This is the pattern proven correct in experiment 11.

## Acceptance Criteria

- [ ] `src/parkinson/seqnet_td.py` importable; passes `ruff check`; 5 unit tests.
- [ ] `experiments/12_multi_paradigm_bernstein_sota.py` passes `ruff check`.
- [ ] Script runs end-to-end (5-fold CV + full train + submission).
- [ ] Engine 2 (Anchored GBDT) OOF RMSE printed and substantially below 6.13.
- [ ] 6-regime gating weights printed.
- [ ] Global OOF RMSE, 6-regime breakdown, and bootstrap test printed.
- [ ] Monotonicity violation rate asserted to be 0.00%.
- [ ] `submissions/12_multi_paradigm_bernstein_sota.csv` written (11 013 rows,
      zero NaN, values in [0.0, 132.0]).
- [ ] Hub metrics pushed as `"12_multi_paradigm_bernstein_sota"`.
- [ ] `00_best_model.py` updated if Bernstein RMSE < 3.2317 (exp-11 record).
- [ ] Full `pytest` suite passes (0 regressions).

## Results

*To be filled after running the script on the full dataset.*

| Metric | Exp 11 | Exp 12 |
|---|---|---|
| Engine 1 (TD-SeqNet) OOF RMSE | — | TBD |
| Engine 2 (Anchored GBDT) OOF RMSE | — | TBD |
| Gated Blend OOF RMSE | — | TBD |
| Bernstein OOF RMSE | 3.2317 | TBD |
| Observed-OFF RMSE | TBD | TBD |
| Missing-OFF RMSE | TBD | TBD |
| Monotonicity violation rate | 0.00% | 0.00% |
| P(Bernstein < GatedBlend) | — | TBD |

## Links

- Script: [`experiments/12_multi_paradigm_bernstein_sota.py`](../experiments/12_multi_paradigm_bernstein_sota.py)
- TD-SeqNet: [`src/parkinson/seqnet_td.py`](../src/parkinson/seqnet_td.py)
- Tests: [`tests/test_seqnet_td.py`](../tests/test_seqnet_td.py)
- Projector: [`src/parkinson/trajectory.py`](../src/parkinson/trajectory.py)
- Submission: [`submissions/12_multi_paradigm_bernstein_sota.csv`](../submissions/12_multi_paradigm_bernstein_sota.csv)
