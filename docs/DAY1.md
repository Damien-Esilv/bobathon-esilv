# Guided lab – Day 1

## 1. Data into `data/`

Competition tables are in git, folder `data/`.
You should see `X_train.csv`, `y_train.csv`, `X_test.csv`, and `sample_submission.csv`. Every teammate needs these CSVs locally.

### What the data is about and the goal

- **Context**: Parkinson’s disease (PD) causes motor symptoms (tremor, rigidity, bradykinesia) scored via MDS-UPDRS (0–132). Patients taking levodopa fluctuate between an **ON** state (symptoms relieved) and an **OFF** state (medication worn off). The OFF score reflects underlying neurodegeneration, but clinic measurements are biased by drug timing, human scoring subjectivity, and missingness (OFF tests are uncomfortable and often skipped).
- **The data**: Synthetic multi-cohort patient visits at irregular intervals. Columns include patient demographics, genetics, medication dose (`ledd`), measured `on` and `off` motor scores, and intake delays (`time_since_intake_`*). Test patients do not overlap train patients (`patient_id` holdout).
- **The goal**: Predict the debiased **true OFF** motor score (`target`) for every patient visit by capturing disease progression over time, leveraging drug intake delays, and treating missingness as signal. See [CONTEXT.md](CONTEXT.md) for full clinical background.



## 2. Explore the data (EDA)

**Exploratory data analysis (EDA)** is looking at the tables *before* you pick a model: shape, missingness, `patient_id` groups, what looks like a leak. In Bob this is the **G-EDA** step (`explore-ml-data`). It writes `data/eda.py`, a short narrative `data/eda.md`, and interactive `data/eda_<table>.html` pages.

### Example prompt:

```
Explore my data in the `data/` folder.
```



### Workbook questions

Questions to think about as you browse the EDA pages (`data/eda_<table>.html`, or the `eda` report on Skore Hub). No need to write anything down. They are there to point you at what to look for.

1. How many visits and how many distinct patients are in train? In test? What is the typical number of visits per patient, and what is the most?
2. Which three columns have the most missing values? What fraction of visits have **no** `off` score? **No** `on` score? Is there any visit where both are missing?
3. Look at the distribution of `target`. What range does it cover, compared with the theoretical 0–132 range of MDS-UPDRS? Is it skewed?
4. Which columns look most associated with `target`? Is `off` (the measured score) a near copy of `target`, or clearly different? What about `on`?
5. Is `ledd` missing at random, or is it missing more often for some `cohort` values? What could that mean clinically?
6. Which columns would you **not** feed to a model, and why? (Hint: think about identifiers and leakage.)



## 3. Dummy mean: a floor

Before building a real model, build the laziest possible one. A **dummy mean** model ignores everything it knows about a visit (age, medication, measured scores) and always gives the same answer: the **average** `target` over all training visits. If the average true OFF score in train is 30, it predicts 30 for every patient, every visit.
This sounds useless, and as a model it is. As a **reference point** (a *baseline*), it is essential:

- **It sets the floor.** Its error tells you how wrong you are when you know nothing. Any real model must do better; otherwise its columns are not helping at all.
- **It catches bugs early.** If a "smart" model scores the same as or worse than the dummy, something upstream is broken, not the model: the data was loaded wrong, rows got misaligned with their targets, the metric is miscomputed, or the submission file is malformed.
- **It gives your scores meaning.** An RMSE of 8 means nothing on its own. "8, compared with 12 for the dummy" means your model removes a third of the error.

Start by loading the visits.

Each **row** is a **visit**. The target is a **true / unbiased OFF** MDS-UPDRS motor score (`y_train.target`). Clinic ON/OFF scores are biased (subjectivity, missing values, levodopa timing).

- `patient_id`: several visits per patient. The Kaggle holdout is **by patient**.
- **Missingness**: `on`, `off`, `ledd`, `gene`, intake delays are often missing.
- **ON / OFF / LEDD / timing**: `on`, `off`, `ledd`, `time_since_intake_on`, `time_since_intake_off`. On train, years since diagnosis is `age - age_at_diagnosis` (test may already have `time_since_diagnosis`).

```python
import pandas as pd

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")

visits = X_train.merge(y_train, on="Index")
y = visits["target"]
```

Use `sample_submission.csv` as the shape template: columns `Index,target`. Each experiment writes its file to `submissions/<report-key>.csv`, so the columns and the filename both match the competition.

- **What it is**: Dummy always predicts the same number: the average true OFF in train.
- **How it works**: it ignores every column. You still pass an `X` so `skore.evaluate` can line up the same rows as `y`.
- **Why here**: if you cannot beat this floor, the data load, the metric, or the upload is broken - not the model.

```python
from sklearn.dummy import DummyRegressor
from skore import evaluate

feature_cols = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
X = visits[feature_cols]
y = visits["target"]

dummy = DummyRegressor(strategy="mean")
report = evaluate(dummy, X, y)
report.metrics.rmse()
```

`skore.evaluate` is the entry point. With no `splitter=`, it uses default `splitter=0.2` (a random row holdout). That is enough to check the floor; grouped splits come on Day 2.

**Project** stores reports. **Hub** is required for a valid Submission (the URL printed after `put`). `name=` is your project inside the hub workspace from `.skore`. Always `load_skore_credentials()` then `login(mode="hub")`:

```python
from parkinson.hub import load_skore_credentials
from skore import Project, login

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)
# The console prints: Consult your report at https://skore.probabl.ai/…
```

`Project.get` is by **id** from `project.summarize()`, not by the string key you passed to `put`.

### Example prompt:

```
Run a dummy regressor from sklearn to act as my baseline model. Push to the hub with the name '01_dummy' as the report name to the project 'bobathon-esilv'.
```



### Workbook questions

Open the `01_dummy` report URL on Skore Hub and answer from there.

1. What is the dummy's RMSE? Compare it with the standard deviation of `target` from your EDA. Why are those two numbers so close?
2. Look at the prediction-error plot (predicted vs. actual). Describe its shape. What would a perfect model look like on the same plot?
3. Write the dummy RMSE somewhere visible. Every model from now on must beat it. By how much would a model need to beat it before you call the improvement real, rather than noise?

