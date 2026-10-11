# 07_tuning — Design Note

## Intent

Systematically tune the hyperparameters of `HistGradientBoostingRegressor` on top of the
enriched clinical feature set from `06_features`, exploring depth, regularisation, and learning rate.
Additionally, evaluate a two-seed blended ensemble (`random_state=42` and `random_state=0`) to
smooth tree-partition boundaries and reduce model variance.

## Method

### Hyperparameter Search Space

A grid search was executed over the key tree-building parameters using 5-fold GroupKFold by `patient_id`:

- `learning_rate`: `{0.05, 0.10}`
- `max_leaf_nodes`: `{31, 63}`
- `min_samples_leaf`: `{10, 20, 30}`
- `l2_regularization`: `{0.0, 0.1}`
- `max_iter`: 200 during exploration, extended to 500 for the winning configuration.

### Winning Configuration (`BEST_PARAMS`)

```python
BEST_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 500,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
    "random_state": 42,
}
```

### Multi-Seed Blending

To reduce stochastic tree variance, an ensemble combining two random seeds (`seed=42` and `seed=0`)
was evaluated on out-of-fold validation:
```python
ŷ_blend = 0.5 * ŷ_seed42 + 0.5 * ŷ_seed0
```

### Evaluation & Submissions

- **Cross-Validation**: 5-fold `GroupKFold` grouped strictly by `patient_id`.
- **Predictions**: Clipped to `[0, 132]`.
- **Submissions**: Wrote `submissions/07_tuning.csv` (11 013 rows).
- **Hub tracking**: Pushed report to Skore Hub as `"07_tuning"`.

## Acceptance Criteria

- [x] Hyperparameter grid search evaluated across candidate configurations with GroupKFold.
- [x] Best configuration identified and scaled to `max_iter=500`.
- [x] Blended ensemble evaluated on OOF validation.
- [x] Report pushed to Hub as `"07_tuning"`.
- [x] `submissions/07_tuning.csv` written (11 013 rows).
- [x] `00_best_model.py` and `best_submission.csv` updated (established Day 2 tabular baseline).

## Results

| Configuration | Pooled RMSE (5-fold GroupKFold) |
|---|---|
| Default HGBR (`06_features`) | 5.680 (std: 0.120) |
| lr=0.10, leaves=63, min_leaf=10 (max_iter=200) | 5.823 |
| lr=0.10, leaves=31, min_leaf=20 (max_iter=200) | 5.761 |
| lr=0.05, leaves=63, min_leaf=10 (max_iter=200) | 5.698 |
| lr=0.05, leaves=31, min_leaf=30 (max_iter=200) | 5.684 |
| lr=0.05, leaves=31, min_leaf=20, l2=0.1 (max_iter=200) | 5.652 |
| **Best Tuned HGBR (max_iter=500)** | **5.572** (std: 0.136) |
| Two-Seed Blend (`seed=42` + `seed=0`) | **5.568** |

Skore Hub: `"07_tuning"` (workspace: `parKINGson`, project: `bobathon-esilv`)

## Analysis & Insights

1. **Smaller steps + more boosting rounds win**:
   Reducing `learning_rate` from 0.10 to 0.05 and expanding `max_iter` to 500 improved RMSE from 5.680 to 5.572. Slower learning allows the gradient booster to fit subtle interactions between clinical timing and rater leniency without overfitting early folds.

2. **Conservative tree depth resists patient-level overfitting**:
   `max_leaf_nodes=31` with `min_samples_leaf=20` consistently outperformed deeper trees (`max_leaf_nodes=63`, `min_samples_leaf=10`). Because validation is grouped by patient, deep trees overfit individual patient quirks rather than learning general disease progression dynamics.

3. **Multi-seed ensemble gain**:
   A simple average of two models trained with different seeds yielded an additional modest drop (~0.004 RMSE), confirming that variance reduction across tree structures provides free regularization.
