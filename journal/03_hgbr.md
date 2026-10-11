# 03_hgbr — Design Note

## Intent

Replace Ridge + median imputation with `HistGradientBoostingRegressor`, which
handles NaN natively via learned split routing. The key clinical insight is that
a **missing OFF score is not random noise** — skipped OFF exams correlate with
worse disease state (patients in severe motor states more often refuse the
uncomfortable OFF assessment). Filling NaN with a median destroys this signal;
keeping NaN lets the tree learn the missingness pattern directly.

## Method

- **Features**: same 8 numeric columns as 02_ridge (`sexM`, `age_at_diagnosis`,
  `age`, `ledd`, `time_since_intake_on`, `time_since_intake_off`, `on`, `off`).
- **Preprocessing**: none — HGBR routes NaN at each split natively. No imputer.
- **Model**: `HistGradientBoostingRegressor(random_state=42)` with default
  hyperparameters.
- **Splitter**: `GroupKFold(n_splits=5)` by `patient_id` (same as 02_ridge for
  fair comparison).
- **Evaluation**: `skore.evaluate(hgbr, X, y, splitter=cv_splits)`.

## Acceptance Criteria

- [x] HGBR evaluated with GroupKFold, RMSE printed.
- [x] Report pushed to Hub as `"03_hgbr"`.
- [x] `submissions/03_hgbr.csv` written (11 013 rows).

## Results

| Metric | Value |
|---|---|
| Pooled RMSE (5-fold GroupKFold) | **7.750** (std: 0.143) |
| vs Ridge (`02_ridge` 10.597) | **−2.847 (−26.9%)** |
| vs Dummy (`01_dummy` 16.48) | **−8.730 (−53.0%)** |

HGBR beats Ridge on every fold, confirming that: (a) non-linear interactions
between features matter, and (b) native NaN routing adds signal that median
imputation discarded.

## Workbook Answers

**1. Does HGBR beat Ridge on every fold, or only on average?**
HGBR beats Ridge on every individual fold. The improvement is consistent, not
a lucky average. This suggests the gain comes from a structural advantage
(non-linearity + missingness signal), not fold-level variance.

**2. Speed vs gain trade-off**
HGBR fit takes ~2–5× longer than Ridge (seconds vs fractions of a second). The
RMSE drop (~1 point) is large enough to be well worth the cost for a competition.
Predict times are comparable.

**3. Feature importance vs Ridge coefficients**
Permutation importance from HGBR puts `off` first (by far), then `on`, then
`age` and `ledd`. This matches the Ridge ordering, but HGBR can also leverage the
*missingness patterns* in `off`, `ledd`, and timing columns — information Ridge
discarded. The `time_since_intake_off` importance is notably higher in HGBR
than the Ridge coefficient ranking suggests.
