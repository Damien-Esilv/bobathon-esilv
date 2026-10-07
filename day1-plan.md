# Day 1 Plan

## Overview

Complete the Day 1 lab as described in `docs/DAY1.md`:
1. Run EDA on the data in `data/`
2. Build and evaluate a dummy mean baseline, produce `submissions/01_dummy.csv`
3. Push the report to Skore Hub as `01_dummy`
4. Answer the workbook questions from the EDA and dummy report

The data CSVs are already present in `data/`. The workspace has no `src/`, `experiments/`, or `journal/` yet — scaffolding is part of this plan.

---

## Sub-Task 1 — Exploratory Data Analysis (EDA)

**Intent:** Run the `explore-ml-data` skill to produce a structured understanding of the dataset before any modelling. This surfaces shape, missingness, target distribution, and patient grouping — all facts that justify later modelling decisions.

**Expected Outcomes:**
- `data/eda.py` created (jupytext `# %%` script)
- `data/eda.md` created (narrative report)
- `data/eda_X_train.html` and `data/eda_y_train.html` created (interactive skrub `TableReport` pages)
- `journal/JOURNAL.md` updated with the EDA section

**Todo List:**
- [ ] Invoke `explore-ml-data` skill with prompt: "Explore my data in the `data/` folder."
- [ ] Verify all four output files exist
- [ ] Read `data/eda.md` digest to extract answers to the EDA workbook questions (see Sub-Task 4)

**Relevant Context:**
- Data files: `data/X_train.csv`, `data/y_train.csv`, `data/X_test.csv`, `data/sample_submission.csv`
- Key columns to check: `patient_id` (grouping), `off`, `on`, `ledd`, `gene`, `time_since_intake_*`, `target`
- See `docs/DAY1.md §2` and `docs/CONTEXT.md` for clinical background

**Status:** `[ ] pending`

---

## Sub-Task 2 — Scaffold workspace + write dummy experiment

**Intent:** Set up the ML workspace layout and write `experiments/01_dummy.py` — a `DummyRegressor(strategy="mean")` evaluated with `skore.evaluate`, plus a `submissions/01_dummy.csv` output file shaped like `sample_submission.csv`.

**Expected Outcomes:**
- Workspace layout created: `src/parkinson/`, `experiments/`, `journal/`, `submissions/`
- `src/parkinson/hub.py` exists with `load_skore_credentials()` helper (reads `.skore`)
- `journal/JOURNAL.md` index exists with a `01_dummy` entry
- `journal/01_dummy.md` design note exists and is approved
- `experiments/01_dummy.py` exists and runs without error
- `submissions/01_dummy.csv` written (columns: `Index`, `target`)

**Todo List:**
- [ ] Invoke `organize-ml-workspace` skill to scaffold the layout
- [ ] Invoke `iterate-ml-experiment` to create and approve a design note for `01_dummy`
- [ ] Write `experiments/01_dummy.py` using the feature columns from `docs/DAY1.md §3`:
  - Load `X_train.csv`, `y_train.csv`; merge on `Index`; select `feature_cols`
  - `DummyRegressor(strategy="mean")` → `skore.evaluate(dummy, X, y)`
  - Write `submissions/01_dummy.csv` from `X_test` predictions
- [ ] Run the experiment and confirm RMSE is printed

**Relevant Context:**
- Feature columns (from `docs/DAY1.md`): `sexM`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`, `time_since_intake_off`, `on`, `off`
- Merge key: `Index` (both `X_train.csv` and `y_train.csv` have this column)
- `skore.evaluate` default splitter is `0.2` random holdout — correct for Day 1
- Submission shape: columns `Index, target` — use `sample_submission.csv` as template

**Status:** `[ ] pending`

---

## Sub-Task 3 — Push report to Skore Hub

**Intent:** Push the evaluated dummy report to Skore Hub so the URL can be used in the Kaggle submission description. This is the mandatory link between every Kaggle upload and a Skore report.

**Expected Outcomes:**
- `project.put("01_dummy", report)` executes without error
- Console prints a URL like `https://skore.probabl.ai/…`
- URL is recorded (in `journal/01_dummy.md` or `JOURNAL.md`)

**Todo List:**
- [ ] Add Hub push block to `experiments/01_dummy.py` (or run inline):
  ```python
  from parkinson.hub import load_skore_credentials
  from skore import Project, login

  cfg = load_skore_credentials()
  login(mode="hub")
  project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
  project.put("01_dummy", report)
  ```
- [ ] Run the script and capture the printed Hub URL
- [ ] Record the URL in `journal/JOURNAL.md`

**Relevant Context:**
- `.skore` contains `hub_url`, `workspace` (`damien-malo-esteban`), `workspace_id`, `api_key`
- `load_skore_credentials()` lives in `src/parkinson/hub.py` (to be created in Sub-Task 2)
- Project name is `bobathon-esilv` (fixed by competition rules)

**Status:** `[ ] pending`

---

## Sub-Task 4 — Answer Day 1 workbook questions

**Intent:** Use the EDA digest and the dummy report on Skore Hub to answer all reflection questions from `docs/DAY1.md`. Answers go into `data/eda.md` (EDA questions) and `journal/01_dummy.md` (dummy questions).

**Expected Outcomes:**
- All 6 EDA workbook questions answered with numbers from `data/eda.md`
- All 3 dummy workbook questions answered (RMSE, prediction-error plot description, floor comparison)

**Todo List:**
- [ ] Read `data/eda.md` and answer EDA questions 1–6
- [ ] Read the dummy RMSE from the experiment output and answer dummy questions 1–3
- [ ] Append answers to `journal/01_dummy.md` under a `## Workbook answers` section

**EDA Workbook Questions (docs/DAY1.md §2):**
1. How many visits and distinct patients in train vs test? Typical / max visits per patient?
2. Top-3 columns with most missing values; fraction with no `off`; no `on`; both missing?
3. Distribution of `target`: range, skew?
4. Columns most associated with `target`; is `off` a near-copy of `target`?
5. Is `ledd` missing more for certain `cohort` values?
6. Which columns should NOT go into a model, and why?

**Dummy Workbook Questions (docs/DAY1.md §3):**
1. What is the dummy RMSE? How does it compare with `std(target)`?
2. What does the prediction-error plot look like? What would a perfect model look like?
3. Record the dummy RMSE as the floor. How much improvement would be "real"?

**Status:** `[ ] pending`
