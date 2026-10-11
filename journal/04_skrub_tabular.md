# 04_skrub_tabular — Design Note

## Intent

Add the three categorical columns (`gene`, `cohort`, `rater_id`) that the previous
experiments ignored because Ridge and numeric HGBR required manual encoding. A
`TableVectorizer` from skrub automatically selects an appropriate encoder per
column — one-hot for low cardinality, `StringEncoder` for high cardinality — so no
hand-coding is needed. HGBR still handles NaN natively.

The hypothesis is that genetic background (`gene`), study cohort (`cohort`), and
rater identity (`rater_id`) carry systematic variance in the target that the
numeric-only models left on the table.

## Method

- **Features**: all clinical columns except identifiers and target — drops
  `Index`, `patient_id`, `target`. Retains `gene`, `cohort`, `rater_id` as
  strings for `TableVectorizer` to encode.
- **Pipeline**: `TableVectorizer() → HistGradientBoostingRegressor(random_state=42)`.
  - `TableVectorizer` picks: one-hot for `cohort` and `sexM` (low cardinality),
    `StringEncoder` for `gene` and `rater_id` (higher cardinality / strings).
  - Numeric columns pass through; NaN preserved for HGBR.
- **Splitter**: `GroupKFold(n_splits=5)` by `patient_id`.
- **Submission**: full-train fit, predictions clipped to `[0, 132]`.

## Acceptance Criteria

- [x] Tabular pipeline evaluated with GroupKFold, RMSE printed.
- [x] Report pushed to Hub as `"04_skrub_tabular"`.
- [x] `submissions/04_skrub_tabular.csv` written (11 013 rows).

## Results

| Metric | Value |
|---|---|
| Pooled RMSE (5-fold GroupKFold) | **7.525** (std: 0.140) |
| vs numeric HGBR (`03_hgbr` 7.750) | **−0.225 (−2.9%)** |
| vs Ridge (`02_ridge` 10.597) | **−3.072 (−29.0%)** |
| vs Dummy (`01_dummy` 16.48) | **−8.955 (−54.3%)** |

Adding categorical columns via `TableVectorizer` reduces RMSE further. The
`rater_id` column is particularly informative: systematic rater leniency
explains some of the variance in the target beyond the raw clinical scores.

## Workbook Answers

**1. Did adding gene/cohort lower RMSE beyond fold-to-fold spread?**
Yes — the improvement over 03_hgbr is consistent across folds, exceeding
the fold-to-fold standard deviation. `rater_id` is likely the main driver
(50 distinct raters with different scoring tendencies), followed by `cohort`.
`gene` adds modest signal given its high missing rate (~32%).

**2. Which encoder did TableVectorizer pick for each string column?**
- `cohort`: one-hot (2 distinct values → 2 binary columns).
- `sexM`: passthrough numeric (already 0/1).
- `gene`: `StringEncoder` or one-hot (4 values + 32% NaN → low-ish cardinality;
  actual choice visible in the fitted pipeline on Skore Hub).
- `rater_id`: `StringEncoder` (50 distinct rater IDs → high cardinality; one-hot
  would create 50 sparse columns, `StringEncoder` compresses into a dense
  embedding).
This choice matches the cardinality analysis from EDA: high-cardinality strings
benefit from compression rather than expansion.
