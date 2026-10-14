# Plan: Experiment 11 — Monotonic Bernstein Trajectory Projector

## Top-Level Overview

**Goal:** Add experiment 11, a post-processing layer that applies a per-patient
monotone degree-3 Bernstein curve to the best existing OOF blend (the
`RetrospectiveSmoothingRegressor` from experiment 10), then evaluates it with a
6-regime clinical breakdown and a paired patient-grouped bootstrap test.

**Step number:** `11` (highest existing experiment is `10_retrospective_smoothing`).

**Mode:** B — standalone. There is no `results/` directory and no frozen-split
file. The experiment runs its own 5-fold `GroupKFold` CV (matching the pattern
of every prior experiment).

**Scope:**
- New library module: `src/parkinson/trajectory.py` — `MonotonicBernsteinTrajectoryProjector`.
- New experiment script: `experiments/11_bernstein_conformal_super_moe.py`.
- New unit test: `tests/test_trajectory.py`.
- New journal note: `journal/11_bernstein_conformal_super_moe.md`.
- Update `journal/JOURNAL.md` history table.
- Conditionally update `experiments/00_best_model.py` and `journal/00_best_model.md`.

**Out of scope (not present in repo):**
- Frozen split file (`results/frozen_splits_5fold.csv` does not exist).
- Super-MoE ensemble loading (no prior OOF `.csv` files exist in `results/`).
- Conformal prediction wrapper (not part of the agreed scoped plan).
- Regime gating network with SLSQP (replaced by the simpler 2-regime blend from exp 10).

---

## Sub-Task 1 — Implement `MonotonicBernsteinTrajectoryProjector`

**Status:** [ ] pending

**Intent:**
Create `src/parkinson/trajectory.py` containing the
`MonotonicBernsteinTrajectoryProjector` class. This is the core scientific
contribution: it takes a (patient_id, age, raw_prediction) triple for a set of
visits and returns monotone predictions by fitting a constrained degree-3
Bernstein curve per patient using IRLS.

**Expected Outcomes:**
- `src/parkinson/trajectory.py` exists and is importable.
- `MonotonicBernsteinTrajectoryProjector` exposes a sklearn-style
  `fit_transform(X, raw_preds)` interface (or equivalent).
- No external dependencies beyond numpy and scipy.

**Todo List:**
1. Create `src/parkinson/trajectory.py`.
2. Implement helper: normalized age coordinate `s ∈ [0, 1]` per patient (clamp
   denominator to ≥ 0.25 to avoid division by zero for single-visit patients).
3. Implement helper: degree-3 Bernstein basis matrix `B` (shape `[n_visits, 4]`).
4. Implement reparameterization matrix `M` (lower-triangular ones, shape `[4, 4]`)
   so that the constrained variable `α` maps to non-decreasing `θ = M @ α`.
5. Implement observation weight assignment:
   - `w = 6.0` if `off` is not NaN and `time_since_intake_off` is not NaN (Slice A).
   - `w = 3.0` if `off` is not NaN and `time_since_intake_off` is NaN (Slice B).
   - `w = 1.0` if `off` is NaN (missing OFF visits).
6. Implement the 2-step IRLS solve:
   - Step 1: `scipy.optimize.lsq_linear` with bounds `α[0] ∈ (-∞, ∞)`, `α[1:] ≥ 0`,
     plus a curvature regularization term `λ_curv = 0.01` on `D₂ @ M @ α`.
   - Compute residuals; Huber-attenuate weights for visits with downward residual
     `> 3.0` points: `w_t ← w_t / (1 + (|r_t| / 3.0)²)`.
   - Step 2: re-solve with updated weights.
7. Implement `fit_transform(df, raw_preds)` that:
   - Groups by `patient_id`.
   - For single-visit patients (no trajectory possible): passes through the raw
     prediction unchanged.
   - Returns a numpy array of monotone predictions aligned to the input row order.
8. Add monotonicity assertion helper (returns fraction of consecutive pairs that
   are non-decreasing; used in the experiment script to verify the guarantee).

**Relevant Context:**
- `src/parkinson/features.py` — pattern for stateless patient-level transformations.
- `scipy.optimize.lsq_linear` — constrained least squares (already in the scipy
  version pinned by `requirements.txt`).
- Degree-3 Bernstein basis: `B₀(s) = (1-s)³`, `B₁(s) = 3s(1-s)²`,
  `B₂(s) = 3s²(1-s)`, `B₃(s) = s³`.
- Second-difference matrix `D₂` for 4 control points: `[[1, -2, 1, 0], [0, 1, -2, 1]]`.

---

## Sub-Task 2 — Add Unit Tests for the Projector

**Status:** [ ] pending

**Intent:**
Create `tests/test_trajectory.py` to verify the three core properties of
`MonotonicBernsteinTrajectoryProjector` on synthetic data before running the
expensive experiment script.

**Expected Outcomes:**
- `pytest tests/test_trajectory.py` passes with 0 failures.
- The full `pytest` suite still passes (no regressions).

**Todo List:**
1. Create `tests/test_trajectory.py`.
2. Test: `output has the same shape as input` — projector returns an array of the
   same length as the input visit list.
3. Test: `monotonicity guarantee` — for a multi-visit patient, consecutive
   predictions are non-decreasing (within floating-point tolerance `1e-6`).
4. Test: `single-visit passthrough` — a patient with one visit gets back the
   original raw prediction unchanged.
5. Test: `predictions clipped to [0, 132]` — even if the Bernstein curve
   extrapolates slightly, the experiment script should clip; verify the projector
   itself returns physically plausible values (or document that clipping is the
   experiment script's responsibility).
6. Run `ruff check tests/test_trajectory.py` — zero lint errors.

**Relevant Context:**
- `tests/test_pipeline.py` — shows the pattern for synthetic DataFrames with the
  right column schema (patient_id, age, off, time_since_intake_off).
- `tests/conftest.py` — look for shared fixtures before duplicating setup.

---

## Sub-Task 3 — Implement `experiments/11_bernstein_conformal_super_moe.py`

**Status:** [ ] pending

**Intent:**
Write the experiment script. It reproduces the exp-10 5-fold CV loop, applies
`MonotonicBernsteinTrajectoryProjector` on the OOF predictions fold-by-fold,
computes the 6-regime evaluation breakdown, runs the bootstrap test, trains
the full model, writes output files, and conditionally updates the best-model
pointer.

**Expected Outcomes:**
- Script runs end-to-end: `python experiments/11_bernstein_conformal_super_moe.py`.
- `submissions/11_bernstein_conformal_super_moe.csv` written (11 013 rows, zero
  NaN, values in [0.0, 132.0]).
- Global OOF RMSE, 6-regime breakdown, and bootstrap test printed to stdout.
- Strict monotonicity assertion: `0.00%` of consecutive within-patient pairs
  have downward diffs `< -1e-6`.
- `experiments/00_best_model.py` and `journal/00_best_model.md` updated if this
  experiment achieves a lower Global OOF RMSE than the current winner (3.303).

**Todo List:**
1. Scaffold script as a `# %%` jupytext percent file (matching repo convention).
2. Data loading: `data/X_train.csv`, `data/y_train.csv`, `data/X_test.csv`;
   sort deterministically by `["patient_id", "age", "Index"]`.
3. 6-regime labelling function:
   - Regime 0: `off` notna & `time_since_intake_off` notna.
   - Regime 1: `off` notna & `time_since_intake_off` isna & (`ledd` == 0 or isna).
   - Regime 2: `off` notna & `time_since_intake_off` isna & `ledd` > 0.
   - Regime 3: `off` isna & patient has ≥ 3 observed-OFF visits.
   - Regime 4: `off` isna & patient has 1–2 observed-OFF visits.
   - Regime 5: `off` isna & patient has 0 observed-OFF visits.
4. 5-fold CV loop using `GroupKFold(n_splits=5)` on `patient_id` groups:
   a. Train the `RetrospectiveSmoothingRegressor` on the fold's training data.
   b. Generate raw OOF blend predictions for the validation fold.
   c. Apply `MonotonicBernsteinTrajectoryProjector` to the OOF predictions
      within this fold (fit on the validation fold's visit structure only —
      the projector is deterministic and uses no targets, so no leakage).
   d. Collect fold OOF preds, targets, regime labels, and patient_ids.
5. Compute and print the 10-slice clinical evaluation table:
   - Global Pooled OOF RMSE.
   - Observed-OFF RMSE (regimes 0+1+2).
   - Missing-OFF RMSE (regimes 3+4+5).
   - Regime 0 (Slice A: Obs, τ_off not NaN).
   - Regime 1 (Slice B: Obs, τ_off NaN, untreated).
   - Regime 2 (Slice C: Obs, τ_off NaN, treated).
   - Regime 3 (Mis, Dense Intra-OFF ≥ 3 obs).
   - Regime 4 (Mis, Sparse Intra-OFF 1–2 obs).
   - Regime 5 (Mis, Zero Intra-OFF).
   - Terminal visits (patients with ≥ 8 visits — their last visit only).
6. Strict monotonicity assertion: compute fraction of consecutive within-patient
   prediction pairs with downward diff < -1e-6; print it; assert it == 0.0.
7. Paired patient-grouped bootstrap test (2 000 resamples) against exp-10 blend
   (use the raw blend as baseline, monotone-projected as challenger).
8. Train full `RetrospectiveSmoothingRegressor` on all training data; apply
   `MonotonicBernsteinTrajectoryProjector` to test predictions.
9. Write `submissions/11_bernstein_conformal_super_moe.csv` (Index, target).
10. Push metrics dict to Skore Hub (wrapped in try/except to not fail if offline).
11. Conditionally update `experiments/00_best_model.py` (WINNER line) and
    `journal/00_best_model.md` if new RMSE < current best (3.303).
12. Implement `run()` function (called by `00_best_model.py` if this becomes winner).
13. Run `ruff check experiments/11_bernstein_conformal_super_moe.py` — zero errors.

**Relevant Context:**
- `experiments/10_retrospective_smoothing.py` — direct template for CV loop,
  bootstrap test, Hub push, and best-model update pattern.
- `src/parkinson/pipeline.py` — `RetrospectiveSmoothingRegressor` API.
- `src/parkinson/trajectory.py` — `MonotonicBernsteinTrajectoryProjector` (from Sub-Task 1).
- The projector is applied **after** the blend, not instead of it; so the CV
  loop trains RSR normally and then post-processes OOF preds.
- The test-set monotone projection is computed patient-by-patient using
  `X_test`'s `patient_id` and `age` columns, with the full-model test preds
  as the raw input.

---

## Sub-Task 4 — Documentation & Tracking

**Status:** [ ] pending

**Intent:**
Write the journal design note and update the JOURNAL index so the new experiment
is discoverable, and conditionally update the best-model tracking files.

**Expected Outcomes:**
- `journal/11_bernstein_conformal_super_moe.md` exists and is well-structured.
- `journal/JOURNAL.md` history table has a new row for experiment 11.
- If RMSE < 3.303: `experiments/00_best_model.py` WINNER line updated and
  `journal/00_best_model.md` (if it exists) reflects the new winner.

**Todo List:**
1. Create `journal/11_bernstein_conformal_super_moe.md` following the template
   of `journal/10_retrospective_smoothing.md`:
   - Intent section.
   - Method section: Bernstein basis, reparameterization, IRLS weighting, 6-regime
     labelling.
   - Acceptance criteria checklist (filled in after the run).
   - Results table (Global OOF, 6-regime breakdown, bootstrap stats).
   - Clickable links to `experiments/11_...py`, `submissions/11_...csv`.
2. Add a new row to the `## History` table in `journal/JOURNAL.md`.
3. Check whether `journal/00_best_model.md` exists; if yes, update it when
   appropriate.

**Relevant Context:**
- `journal/10_retrospective_smoothing.md` — template to follow.
- `journal/JOURNAL.md` lines 29–42 — history table format.

---

## Notes for Implementation

- **No `results/` directory needed.** The OOF CSV files mentioned in the original
  request are an optional artifact; the experiment can produce them but they are
  not required by any existing test.
- **IRLS loop is per-patient.** Patients with only 1 visit skip the Bernstein fit
  (the polynomial is undefined for a single point) — pass through raw prediction.
- **Clipping is the experiment script's responsibility.** The projector returns
  the raw polynomial outputs; the script clips to [0, 132] before writing files.
- **scipy is already available** (`requirements.txt` includes scipy via the
  existing dependency chain).
- **SeqNet+GBDT CV training is the expensive part.** The projector adds
  negligible compute.
- **Monotonicity guarantee only applies within a patient's observed time range.**
  Extrapolation is not attempted.
