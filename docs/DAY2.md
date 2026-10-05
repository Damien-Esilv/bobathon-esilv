# Guided lab – Day 2

On Day 1 you loaded the competition data, explored it (EDA) to see what is missing and how visits group by patient, and pushed a dummy mean baseline to Skore Hub as the score every model must beat. Today you build real models, learn to evaluate them honestly with patient-grouped cross-validation, and submit to Kaggle.

## 1. A simple linear model

Numeric clinical variables (`age`, `on`, `off`, `ledd`, delays) may carry a simple additive signal.

- **What it is**: Ridge draws a straight line: predicted true OFF ≈ intercept + (a weight × each column).
- **How it works**: it chooses weights so predictions match `y`, then `alpha` pulls those weights toward zero so one noisy column cannot dominate.
- **Why here**: a first check that the numbers matter at all. Ridge only eats numbers with no holes, so median-impute first and leave strings like `gene` for later.

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from skore import evaluate

ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)
report = evaluate(ridge, X, y)
```

`make_pipeline(*steps)` stitches transformers then the regressor. Compare dummy vs ridge in one report:

```python
report = evaluate(
    {"dummy": dummy, "ridge": ridge},
    X,
    y,
)
```

`put` with a new key. The default row holdout can still leak the same patient into both sides - grouped CV is step 3.

### Example prompt:

```
Run a linear model and push the report to the hub. Please compare the results to the dummy model.
```



### Workbook questions

Open the comparison report on Skore Hub.

1. By what **percentage** does Ridge lower the RMSE compared with the dummy? Is that a big win or a small one?
2. Look at the Ridge coefficients. Which feature has the largest weight in absolute value? Is the sign of the `off` and `on` coefficients what you expected clinically?
3. The features were not scaled. Why does that mean you **cannot** read "`age` matters more than `ledd`" straight off the raw coefficient sizes?



## 2. Tweak Ridge and submit on Kaggle

`Ridge(alpha=…)` is the knob: larger `alpha` shrinks coefficients more. Change it, re-run `evaluate`, keep the one that beats dummy by the most.

```python
ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=10.0),  # try 0.1, 1.0, 10.0, …
)
report = evaluate(ridge, X, y)
report.metrics.rmse()
project.put("02_ridge", report)
```

`evaluate` only scores on train. The file you upload is that pipeline **fit on all training visits**, then `predict` on `X_test` (same columns as `X`):

```python
from pathlib import Path
from sklearn.base import clone

final = clone(ridge).fit(X, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])

submissions_dir = Path("submissions")
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "02_ridge.csv", index=False)
```

Upload `submissions/02_ridge.csv` on Kaggle. The file is named after the report key, so each experiment keeps its own CSV - a later model never overwrites an earlier submission. A Submission is valid only if:

1. Code was written and run in **Bob**, evaluation through **skore**.
2. You `put` this Ridge report; the console shows `https://skore.probabl.ai/…`.
3. CSV header is `Index,target`, one row per test visit.
4. The Kaggle **Submission Description** contains that report URL (otherwise the row is invalid even if Kaggle scored the file).

Later models use the same fit → CSV → URL path.

### Workbook questions

1. Across the `alpha` values you tried, how much does RMSE actually move? Why might `alpha` matter so little with only a few features and thousands of rows?
2. Did the coefficients change much between the smallest and largest `alpha`? Compare them on the hub.



## 3. Evaluate with patient-grouped CV

The default `splitter=0.2` shuffles **rows**. That is too kind here.

- **What it is**: GroupKFold is a dress rehearsal of the Kaggle split: a whole patient goes to train or to the held-out fold, never both.
- **How it works**: it cuts patients into 5 groups. Five times, it trains on 4 groups and scores on the 5th, then you read the average RMSE.
- **Why here**: the same person has many visits. A random row split is **leakage**: visits from one patient sit on both sides, so the model memorizes that person instead of generalizing, and metric looks better than the Kaggle competition (which holds out entire patients). Grouped CV blocks that leak.

skore’s sklearn path calls `splitter.split(X, y)` **without** `groups=`, so precompute the index pairs:

```python
from sklearn.model_selection import GroupKFold
from skore import evaluate

groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
report = evaluate(
    {"dummy": dummy, "ridge": ridge},
    X,
    y,
    splitter=cv_splits,
)
report.metrics.summarize()          # table of metrics
report.metrics.get("rmse")          # or report.metrics.rmse()
```

- `splitter=` a list of index pairs (or `5`, or a CV splitter) → `CrossValidationReport`
- several estimators (list or dict) → `ComparisonReport`

Use `cv_splits` from here on. `put` this report with a new key.

skrub DataOps can attach `groups` on the graph - see step 6.

### Workbook questions

Open the cross-validation report on Skore Hub.

1. Compare grouped-CV RMSE with the random-split RMSE from the Day 1 dummy and steps 1–2. Which model's score changed more?
2. Which of the two estimates (random split or grouped CV) is closer to your Kaggle public score? What does that tell you about which one to trust from now on?



## 4. HistGradientBoosting: missingness as signal

ON/OFF/LEDD/genetics are often missing; missingness is part of the generative process (see [CONTEXT.md](CONTEXT.md)). Ridge had to *fill* those holes with a median. Here we keep the holes.

- **What it is**: a **decision tree** is a flowchart of yes/no questions (`age > 62?`, `on` missing?). Each visit falls down the flowchart into a leaf, and that leaf predicts a number (a typical true OFF for visits that landed there). **Boosting** means we do not stop at one tree: we grow many small trees in a sequence, and each new tree is trained on the *mistakes* of the ones before it. The final prediction is the sum of all those small corrections. **Hist** (histogram) is an implementation trick: each numeric column is first cut into a few buckets (like a histogram), so the model looks at bucket ids instead of every distinct age. That keeps it fast on tens of thousands of visits.
- **How it works** -
  - Tree 1 fits a rough sketch of true OFF.
  - Tree 2 looks at the residuals (true OFF minus what tree 1 predicted) and tries to explain what is still wrong.
  - Trees 3, 4, … keep nipping at the remaining error. That is gradient boosting, in plain language: keep adding a small expert on the leftover mistakes.
  - At a split, the algorithm may send **missing** values left or right on purpose. It learns that route from the training data - so “OFF was not measured” can be a signal, not a defect.
  - `random_state=0` only makes the run repeatable.
- **Why here**: in this table, a missing OFF often means the visit was ON-only (uncomfortable OFF exams are skipped). That is information about the patient and the protocol, not noise to impute away. Do **not** median-fill NaNs unless you are testing that ablation (does the model get *worse* when you hide the holes?).

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from skore import evaluate

hgbr = HistGradientBoostingRegressor(random_state=0)
report = evaluate(hgbr, X, y, splitter=cv_splits)
```

Strings are still a problem: by default the model wants numbers or pandas `category` dtypes. `categorical_features="from_dtype"` (the default) treats a `category` column as “pick among a few labels” instead of as a fake number. Convert `cohort` / `gene` with `.astype("category")` if you add them to `X`.

### Workbook questions

Put the HGBR report next to Ridge (same `cv_splits`) and compare on Skore Hub.

1. Does HGBR beat Ridge on **every** fold, or only on average?
2. Compare fit and predict times. How much slower is HGBR? Is the gain worth it here?
3. Look at the feature importance (permutation importance). Which features does HGBR rely on most? Is that ranking different from the Ridge coefficients?



## 5. skrub `tabular_pipeline`: mixed types without hand-encoding

Ridge and the numeric HGBR above never saw `gene` or `cohort`: those are **strings**. A sklearn regressor cannot multiply `"GBA"` by a weight. Something has to turn text into numbers first. Doing that by hand (a custom encoder per column) is where pipelines usually rot.

- **What it is**: `tabular_pipeline("regressor")` is a ready-made two-step recipe from skrub: (1) `TableVectorizer` turns a messy table into a numeric matrix, (2) `HistGradientBoostingRegressor` predicts. You drop ids, pass the rest, and the vectorizer chooses an encoder **per column**.
- **How it works**: “cardinality” means how many distinct values a column has.
  - **Low cardinality** (a handful of labels, e.g. `sexM`, maybe `cohort`) → **one-hot**: one new column per label, `1` if that row has it, `0` otherwise. The model sees a switch, not a made-up ranking of labels.
  - **High cardinality** (many distinct strings, e.g. `gene` if it is messy) → `StringEncoder`: compress the text into a few numeric dimensions instead of hundreds of one-hot columns. You keep signal without exploding the width of `X`.
  - **Numbers** (`age`, `on`, `ledd`, …) pass through. HGBR can still use their NaNs, as in step 4.
  - `Index` and `patient_id` are identifiers, not clinical features. If you leave them in, the model can memorize ids - another form of leakage. Drop them (and `target`) before fitting.
- **Why here**: the interesting PD columns are mixed: numbers with holes *and* categoricals. This step is how you stop throwing `gene` / `cohort` away just because Ridge could not read them.

```python
from skrub import tabular_pipeline
from skore import evaluate

# include categoricals; keep Index / patient_id out of features
X_full = visits.drop(columns=["Index", "patient_id", "target"])
model = tabular_pipeline("regressor")
report = evaluate(model, X_full, y, splitter=cv_splits)
```

Same vectorizer, your own estimator (if you want to tweak HGBR):

```python
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor

model = make_pipeline(
    TableVectorizer(),
    HistGradientBoostingRegressor(random_state=0),
)
```



### Workbook questions

1. Compare this report with the numeric-only HGBR from step 4. Did adding `gene` and `cohort` lower the RMSE beyond the fold-to-fold spread?
2. Look at the fitted pipeline in the report. Which encoder did `TableVectorizer` pick for each string column? Does that choice match each column's cardinality from your EDA?



## 6. skrub DataOps: groups baked into the graph

Until now you have juggled separate objects: a DataFrame `visits`, a list `cv_splits`, a pipeline `model`. Easy to fit on the wrong columns or forget `groups` after a copy-paste. **DataOps** is skrub’s way to write the *recipe* once, as a graph, so features, target, and the grouped split live on the same object.

- **What it is**: a DataOp is not the prediction yet. It is a delayed plan: “when you give me a table named `visits`, drop `target`, vectorize, then apply HGBR.” `skore.evaluate` walks that plan and runs the CV that you attached to it.
- **How it works** -
  - `skrub.var("visits", visits)` names an **input**. Later, at predict time, you can pass a different table under the same name (`X_test`).
  - `.skb.mark_as_X(...)` tells skore “this node is the feature table.” `.skb.mark_as_y()` marks the target. Those two marks are how `evaluate` knows what to split.
  - Passing `cv=GroupKFold(...)` and `split_kwargs={"groups": groups}` **on** `mark_as_X` bakes patient-grouped CV into the graph. You no longer keep a side list `cv_splits` that can go stale.
  - `.skb.apply(transformer)` / `.skb.apply(estimator, y=...)` appends a step, like `make_pipeline`, but on the graph.
  - `evaluate(pred)` with **no** `splitter=` reads `cv` / `groups` from the DataOp. That is the point: the split cannot drift away from the data.
  - `skrub.X(value)` is shorthand for `skrub.var("X", value).skb.mark_as_X()`: no extra `cv` unless you call `mark_as_X` again.
  - `.skb.make_learner()` **freezes** the graph into a `SkrubLearner`. Then `fit` / `predict` take an **environment dict** (`{"visits": ...}`), because the input was a named `var`, not a naked array. You need that for the Kaggle file: test has no `target`.
- **Why here**: GroupKFold only works if `groups` is the `patient_id` of the *same* rows as `X`. Putting that on the graph is how you stop the leakage from step 3 from creeping back in through a forgotten argument.

```python
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from skore import evaluate

data = skrub.var("visits", visits)
groups = data["patient_id"]
X_op = data.drop("target", axis=1).skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups},
)
y_op = data["target"].skb.mark_as_y()
pred = X_op.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(random_state=0),
    y=y_op,
)

# evaluate reads cv / groups from the DataOp when splitter is omitted
report = evaluate(pred)
```



### Workbook questions

1. Compare this report with step 5. The model is the same; are the scores the same? If not, explain the difference.
2. Which columns does `X_op` actually contain? Compare with the columns dropped in step 5. Do `Index` or `patient_id` show up in the feature importance? What would that mean?



## 7. Fit the chosen model and submit again

Grouped CV (steps 3–6) tells you which idea is better. Submit it the same way as Ridge (step 2): `put` a **new** hub report, fit on **all** training visits, `predict` on `X_test`, write the CSV named after that report key, and upload it with the report URL in the Description.

```python
from pathlib import Path
from sklearn.base import clone

final = clone(model).fit(X_full, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test.drop(columns=["Index", "patient_id"], errors="ignore"))

submissions_dir = Path("submissions")
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "<key>.csv", index=False)  # <key> = the report key you put
```

Align test columns with whatever you trained on (same drops, same dtypes). For a DataOp / `SkrubLearner`:

```python
learner = pred.skb.make_learner()
learner.fit({"visits": visits})
pred_test = learner.predict({"visits": X_test})

submission = X_test[["Index"]].copy()
submission["target"] = pred_test
submission.to_csv(submissions_dir / "<key>.csv", index=False)
```



### Workbook questions: wrap-up

Use the project view on Skore Hub to see every report you have `put`.

1. Which single change (grouped CV, missing values kept, categoricals, a model change) gave the biggest improvement? Which gave none?
2. Pick the visits your best model gets most wrong (largest errors in the prediction-error plot). What do they have in common: missing `off`, long `time_since_intake_*`, a specific `cohort`, early or late disease?
3. Based on [CONTEXT.md](CONTEXT.md), which modelling problem (temporal progression, drug timing, missingness) have you **not** addressed yet? What feature would you build next to address it?

