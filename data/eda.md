# EDA: Parkinson's Motor Score

> Interactive HTML reports: [data/eda_train.html](eda_train.html) · [data/eda_test.html](eda_test.html)

---

## Dataset at a glance

| | Train | Test |
|---|---|---|
| Rows (visits) | 44 590 | 11 013 |
| Columns | 14 (incl. target) | 13 (no target) |
| Distinct patients | 5 576 | 1 395 |
| Median visits / patient | 7 | — |
| Max visits / patient | 12 | — |

The holdout is **by patient**: no `patient_id` appears in both train and test. This is the single most important structural fact — it mandates **patient-grouped cross-validation** (`GroupKFold`) on Day 2 to avoid leaking a patient's history across folds.

---

## Per-column findings

| Column | Dtype | Missing % | Notes |
|---|---|---|---|
| `Index` | int | 0 % | Row identifier — exclude from features |
| `patient_id` | string | 0 % | Group key — exclude from features |
| `cohort` | string | 0 % | 2 cohorts (A / B) |
| `sexM` | int | 0 % | Binary (0/1) |
| `gene` | string | **32.4 %** | 4 unique values; high missingness |
| `age_at_diagnosis` | float | 5.2 % | Correlated with `age` (r = 0.94) |
| `age` | float | 0 % | Current age at visit |
| `ledd` | float | **36.6 %** | Levodopa equivalent daily dose |
| `time_since_intake_on` | float | **46.4 %** | Hours since last ON intake |
| `time_since_intake_off` | float | **78.8 %** | Hours since last OFF intake — most-missing column |
| `rater_id` | string | 0 % | 50 distinct raters — possible scorer bias |
| `on` | float | 29.6 % | Measured ON motor score |
| `off` | float | **42.4 %** | Measured OFF motor score |
| `target` | float | 0 % | **True unbiased OFF score** (regression target) |

**Top-3 most-missing columns:** `time_since_intake_off` (78.8 %), `time_since_intake_on` (46.4 %), `ledd` (36.6 %).

**Visits with no `off` score:** 42.4 % — nearly half the training visits have no measured OFF. This is the core clinical motivation: the model must recover the true OFF even when the clinic didn't measure it.

**Visits with no `on` score:** 29.6 %.

**Visits with both missing:** a meaningful fraction (exact count derivable; the two missingness patterns partially overlap). Missingness is informative signal, not noise to discard — treat it as a feature on Day 2.

---

## Target

- **Range:** 0.0 – 109.5 (theoretical MDS-UPDRS motor max is 132; the dataset does not reach it)
- **Mean:** 37.5 · **Median:** 37.3 · **Std:** 16.5
- **IQR:** 25.6 – 49.3
- **Shape:** roughly bell-shaped, slightly right-skewed, centred around 37. No severe outlier mass.
- **No missing values** in train (0 %).

The std of 16.5 will be the approximate RMSE of the dummy mean baseline — that is the floor every real model must beat.

---

## Structure signals

**No datetime columns** detected. Progression over time is implicit in `age` and `age - age_at_diagnosis` (years since diagnosis); there is no explicit visit timestamp column.

**High unique-ratio columns:**

| Column | Unique ratio |
|---|---|
| `Index` | 1.0 (row ID) |
| `patient_id` | 0.125 — 5 576 distinct patients across 44 590 visits |
| `ledd`, `age`, `target` | 0.02–0.03 (continuous) |

`patient_id` with unique ratio 0.125 and median 7 visits/patient is the group column for `GroupKFold`. No datetime column means `TimeSeriesSplit` is not warranted.

---

## Associations

**Strongest feature↔target links (Pearson r):**

| Feature | Pearson r with `target` | Cramér V |
|---|---|---|
| `off` | **0.871** | 0.406 |
| `on` | 0.669 | 0.368 |
| `age` | 0.310 | 0.114 |
| `ledd` | 0.298 | 0.236 |
| `age_at_diagnosis` | 0.133 | 0.051 |
| `time_since_intake_off` | 0.008 (weak linear) | 0.092 |
| `time_since_intake_on` | 0.0002 (flat linear) | 0.177 |

`off` has a Pearson r of **0.87** with `target` — the strongest single predictor. However, `off` is missing for 42 % of visits, so the model cannot rely on it alone. `on` is the second-best predictor (r = 0.67). `ledd` and `age` provide moderate signal.

`time_since_intake_on/off` have near-zero Pearson r but moderate Cramér V — non-linear relationship with target. Drug timing matters but not in a simple linear way.

⚠️ **Potential leakage flag:** `off` and `target` both measure OFF motor severity. `off` is the *biased clinic measurement*; `target` is the *debiased true OFF*. Using `off` as a feature is valid and intended (the task is to debias it), but a model that overfits to `off` will be brittle when `off` is missing. Day 2 models should handle missing `off` gracefully.

**Notable feature–feature correlation:** `age_at_diagnosis` ↔ `age` (r = 0.94) — nearly collinear. Using both provides little extra information; consider using `age - age_at_diagnosis` (disease duration) instead.

---

## Modelling implications

| Finding | Implication |
|---|---|
| Patient holdout split | **`GroupKFold` with `groups=patient_id`** on Day 2 — random row splits leak patients across folds |
| Heavy missingness in `off`, `ledd`, `gene`, `time_since_*` | Use an estimator that handles NaN natively (`HistGradientBoostingRegressor`) or impute; treat missingness as signal (add indicator columns) |
| `off` ↔ `target` r = 0.87 | `off` is the dominant feature when present; model must degrade gracefully when it is absent |
| No datetime column | `TimeSeriesSplit` not needed; `GroupKFold` is the right default |
| `rater_id` (50 raters, no missingness) | Rater bias may be encodeable — worth including as a categorical feature on Day 2 |
| `cohort` (2 values) | Cohort may encode systematic differences; include as a categorical |
| `Index` and `patient_id` | Row identifiers — **exclude from features** |
| Target std ≈ 16.5 | Dummy mean baseline RMSE ≈ 16.5; any real model must beat this floor |

---

## Open questions

1. Does `off` in `X_test` carry the same biased-measurement meaning as in train? (It should — it's a raw clinic value.)
2. Is `rater_id` stable across train and test (same 50 raters)? If raters appear only in one split, it cannot be used directly.
3. `time_since_intake_off` is 78.8 % missing — is that because most visits don't do an OFF exam, or because the timing wasn't recorded? Clinically it's probably the former (OFF exams are uncomfortable and often skipped).
