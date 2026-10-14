# 13_super_moe_final_sota — Design Note

## Intent

Combine the two best sequence architectures — SeqNet v1 (exp 11, standard
bi-directional Transformer, MSE loss) and TD-SeqNet (exp 12, time-decay
attention + BiGRU, Huber+diff loss) — via 6-regime convex gating, then apply
the Monotonic Bernstein Trajectory Projector.

Target: break below **3.0000** OOF RMSE (exp-12 record: 3.0135).

## Key insight

Exp 12 showed that TD-SeqNet alone scored 3.0526 and that the ensemble with
the Anchored GBDT (which scored 10.01) converged to near-100% TD-SeqNet weight.
The correct next step is therefore to ensemble TD-SeqNet with the other strong
neural model (SeqNet v1, which scored ~3.2–3.3), not with the weak GBDT.
Two complementary architectures with different inductive biases (sinusoidal
temporal embedding vs time-decay bias; MSE loss vs Huber+consecutive-diff loss)
should produce orthogonal errors that cancel under convex gating.

## Architecture

```
X_train, y_train
  → 5-fold GroupKFold (single joint CV pass, trains both models per fold)
     for each fold:
       SeqNet v1  : 2-layer Transformer + MSE loss       → preds_v1
       TD-SeqNet  : 2-layer TimeDecay Transformer + BiGRU + Huber+diff loss → preds_td
  → 6-regime SLSQP convex gating on pooled OOF
      [w_v1, w_td] per regime, w >= 0, sum = 1
  → MonotonicBernsteinTrajectoryProjector.transform_predictions(df_oof, blend)
  → Bernstein OOF RMSE (target < 3.0000)
```

## Alignment contract

`df_oof = X_train.iloc[oof["val_indices"]].copy().reset_index(drop=True)`

This ensures row `i` of `df_oof` corresponds to `preds_v1[i]`, `preds_td[i]`,
and `targets[i]`. No global pre-sort of `X_train` is performed (the bug fixed
in exp 11).

## Why one joint CV loop

There are no OOF CSV files on disk from previous experiments. Running two
independent 5-fold CV passes would be equivalent to 10 folds of training.
Training both architectures in a single 5-fold loop (10 models total, 5 pairs)
halves compute while producing OOF predictions on identical fold splits,
which is required for the gating optimization to be meaningful.

## Acceptance Criteria

- [ ] `experiments/13_super_moe_final_sota.py` passes `ruff check`.
- [ ] Script runs end-to-end (5-fold CV + full train + submission).
- [ ] Per-fold and pooled OOF RMSEs for both models printed.
- [ ] 6-regime gating weights printed per regime.
- [ ] Gated blend OOF RMSE printed.
- [ ] Bernstein OOF RMSE printed; asserted < 3.0000 if hypothesis holds.
- [ ] Monotonicity violation rate asserted to be 0.00%.
- [ ] `submissions/13_super_moe_final_sota.csv` written (11 013 rows,
      zero NaN, values in [0.0, 132.0]).
- [ ] Hub metrics pushed as `"13_super_moe_final_sota"`.
- [ ] `00_best_model.py` updated if Bernstein RMSE < 3.0135 (exp-12).
- [ ] Full `pytest` suite passes (0 regressions).

## Results

*To be filled after running the script on the full dataset.*

| Metric | Exp 12 | Exp 13 |
|---|---|---|
| SeqNet v1 OOF RMSE | — | TBD |
| TD-SeqNet OOF RMSE | 3.0526 | TBD |
| Gated Blend OOF RMSE | — | TBD |
| Bernstein OOF RMSE | 3.0135 | TBD |
| Observed-OFF RMSE | TBD | TBD |
| Missing-OFF RMSE | TBD | TBD |
| Monotonicity violation rate | 0.00% | 0.00% |
| Sub-3.0000 achieved | — | TBD |
| P(Bernstein < GatedBlend) | — | TBD |

## Links

- Script: [`experiments/13_super_moe_final_sota.py`](../experiments/13_super_moe_final_sota.py)
- TD-SeqNet: [`src/parkinson/seqnet_td.py`](../src/parkinson/seqnet_td.py)
- SeqNet v1: [`src/parkinson/seqnet.py`](../src/parkinson/seqnet.py)
- Projector: [`src/parkinson/trajectory.py`](../src/parkinson/trajectory.py)
- Submission: [`submissions/13_super_moe_final_sota.csv`](../submissions/13_super_moe_final_sota.csv)
