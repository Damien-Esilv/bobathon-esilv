# 05_dataops — Design Note

## Intent

Demonstrate the skrub DataOps graph pattern where GroupKFold and the patient
groups are **baked into the computation graph**, not held in a separate
`cv_splits` list that can drift. The model itself is identical to 04_skrub_tabular
(TableVectorizer + HGBR); the goal is architectural correctness:
- The groups cannot be accidentally omitted after a copy-paste.
- The same graph works for both training (where `visits` contains `target`) and
  test-time prediction (where it does not), because the `drop("target")` step
  uses `errors="ignore"`.

## Method

- **DataOps graph**:
  ```
  skrub.var("visits", visits)        # named input
    → drop "target" (errors="ignore") # safe for train and test
    → mark_as_X(cv=GroupKFold(5),    # groups baked in
                split_kwargs={"groups": patient_id})
  data["target"] → mark_as_y()
  X_op.apply(TableVectorizer())
      .apply(HGBR(random_state=42), y=y_op)
  ```
- **Evaluation**: `skore.evaluate(learner, data={"visits": visits}, splitter=cv_splits)`.
  (skore's sklearn path requires pre-computed index pairs even for a DataOp learner.)
- **Submission**: `learner.fit({"visits": visits})` then
  `learner.predict({"visits": X_test})` — the named-input API handles column
  alignment automatically.

## Acceptance Criteria

- [x] DataOps graph built and evaluated with GroupKFold.
- [x] RMSE matches 04_skrub_tabular (same model, same data).
- [x] Report pushed to Hub as `"05_dataops"`.
- [x] `submissions/05_dataops.csv` written (11 013 rows).

## Results

| Metric | Value |
|---|---|
| Pooled RMSE (5-fold GroupKFold) | **7.525** (std: 0.140, exactly matches 04_skrub_tabular) |

The score is identical to 04_skrub_tabular because the model and data are the same.
The value of this step is the architecture: the DataOps pattern eliminates the class
of bugs where `groups` or `cv_splits` goes out of sync with the feature matrix.

## Workbook Answers

**1. Are the scores the same as step 5?**
Yes — the model and data are identical. Any small numerical difference (~0.0001
RMSE) comes from floating-point ordering differences in how the learner packages
the data, not from different splits or different models.

**2. Which columns does X_op actually contain?**
`X_op` contains all columns from `visits` except `target` (dropped via
`apply_func`) and, after TableVectorizer, the vectorized representations of
`gene`, `cohort`, `rater_id`. Crucially, `Index` and `patient_id` are **not**
dropped automatically — they remain in `X_op`. This means the model can
potentially memorize patient identifiers, a form of leakage. Experiments 06–10
explicitly drop them before building the feature matrix.
