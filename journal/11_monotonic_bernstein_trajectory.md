# 11_monotonic_bernstein_trajectory — Design Note

## Intent

Apply a per-patient monotone degree-3 Bernstein polynomial as a post-processing
layer on top of the `RetrospectiveSmoothingRegressor` blend (experiment 10).

The projector exploits the clinical prior that true OFF MDS-UPDRS motor scores
are non-decreasing with disease age: enforcing this constraint at inference time
should remove noise-driven downward swings in the OOF predictions and tighten
the trajectory toward the true signal.

## Method

### Pipeline

```
data/X_train.csv + data/y_train.csv
  → sort by [patient_id, age, Index]
  → 5-fold GroupKFold CV (zero patient leakage)
     for each fold:
       → RetrospectiveSmoothingRegressor.fit(X_tr, y_tr)
       → RSR.predict(X_val)  →  raw blend OOF preds
       → MonotonicBernsteinTrajectoryProjector.transform(X_val, blend_preds)
       → clipped Bernstein OOF preds ∈ [0, 132]
  → 6-regime clinical breakdown
  → paired patient bootstrap vs exp-10 blend
  → full-dataset RSR train + test projection
  → submissions/11_monotonic_bernstein_trajectory.csv
```

### Monotonic Bernstein Trajectory Projector (`src/parkinson/trajectory.py`)

**Normalized age:**

```
s = (t − t₁) / max(t_K − t₁, 0.25)   ∈ [0, 1]
```

**Degree-3 Bernstein basis (one row per visit):**

```
B = [B₀,₃(s)  B₁,₃(s)  B₂,₃(s)  B₃,₃(s)]
B₀,₃(s) = (1−s)³   B₁,₃(s) = 3s(1−s)²
B₂,₃(s) = 3s²(1−s) B₃,₃(s) = s³
```

**Monotonicity reparameterization:**

θ = M @ α, where M is the lower-triangular 4×4 matrix of ones.
Bounds: α[0] ∈ ℝ (intercept), α[1:] ≥ 0 (non-negative increments).
This guarantees θ₀ ≤ θ₁ ≤ θ₂ ≤ θ₃, which in turn guarantees y'(s) ≥ 0.

**2-step asymmetric IRLS:**

| Condition | Weight |
|---|---|
| `off` notna & `time_since_intake_off` notna (Slice A) | 6.0 |
| `off` notna & `time_since_intake_off` isna (Slice B) | 3.0 |
| `off` isna (missing OFF) | 1.0 |

Step 1: `scipy.optimize.lsq_linear` (method=bvls) with curvature penalty
λ_curv = 0.01 on D₂ @ M @ α.

Step 2: Huber-attenuate weights for visits where `observation − curve < −3.0`
via `w_t ← w_t / (1 + (|r_t| / 3.0)²)`, then re-solve.

**Single-visit patients:** passed through unchanged (polynomial underdetermined).

### 6-Regime Clinical Stratification

| Regime | Name | Condition |
|---|---|---|
| 0 | Obs – τ_off recorded | `off` notna & `time_since_intake_off` notna |
| 1 | Obs – τ_off NaN, untreated | `off` notna & τ isna & (ledd == 0 or NaN) |
| 2 | Obs – τ_off NaN, treated | `off` notna & τ isna & ledd > 0 |
| 3 | Mis – dense intra-OFF (≥3) | `off` isna & patient has ≥ 3 observed-OFF visits |
| 4 | Mis – sparse intra-OFF (1–2) | `off` isna & patient has 1–2 observed-OFF visits |
| 5 | Mis – zero intra-OFF | `off` isna & patient has 0 observed-OFF visits |

## Acceptance Criteria

- [ ] `src/parkinson/trajectory.py` is importable; passes `ruff check`.
- [ ] `tests/test_trajectory.py` passes (6 tests: shape, monotonicity, passthrough,
      finite, violation rate, row alignment).
- [ ] `experiments/11_monotonic_bernstein_trajectory.py` runs end-to-end.
- [ ] Global OOF RMSE, 6-regime breakdown, and bootstrap test printed.
- [ ] Monotonicity violation rate asserted to be 0.00%.
- [ ] `submissions/11_monotonic_bernstein_trajectory.csv` written (11 013 rows,
      zero NaN, values in [0.0, 132.0]).
- [ ] Hub metrics pushed as `"11_monotonic_bernstein_trajectory"`.
- [ ] `00_best_model.py` updated if Bernstein RMSE < 3.303.
- [ ] Full `pytest` suite passes with 0 regressions.

## Results

*To be filled after the experiment script is run on the full dataset.*

| Metric | Blend (exp 10) | Bernstein (exp 11) |
|---|---|---|
| Global Pooled OOF RMSE | 3.3032 | TBD |
| Observed-OFF RMSE | 2.9429 | TBD |
| Missing-OFF RMSE | 3.7372 | TBD |
| Regime 0 (Obs, τ recorded) | TBD | TBD |
| Regime 1 (Obs, τ NaN, untreated) | TBD | TBD |
| Regime 2 (Obs, τ NaN, treated) | TBD | TBD |
| Regime 3 (Mis, dense ≥3) | TBD | TBD |
| Regime 4 (Mis, sparse 1–2) | TBD | TBD |
| Regime 5 (Mis, zero) | TBD | TBD |
| Terminal visit (≥8-visit patients) | TBD | TBD |
| Monotonicity violation rate | N/A | 0.00% |
| P(Bernstein < blend) bootstrap | — | TBD |

## Links

- Script: [`experiments/11_monotonic_bernstein_trajectory.py`](../experiments/11_monotonic_bernstein_trajectory.py)
- Projector: [`src/parkinson/trajectory.py`](../src/parkinson/trajectory.py)
- Tests: [`tests/test_trajectory.py`](../tests/test_trajectory.py)
- Submission: [`submissions/11_monotonic_bernstein_trajectory.csv`](../submissions/11_monotonic_bernstein_trajectory.csv)
