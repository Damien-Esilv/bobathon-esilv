# Sequence Transformer Plan — Experiment 14

## Top-Level Overview

**Goal:** Build a bi-directional self-attention sequence model (`SeqNet`) in
PyTorch that treats each patient's full visit timeline as a sequence, exploiting
the retrospective nature of this competition (all past and future visits are
simultaneously available). The model predicts a normalized residual on top of
an anchored medication baseline (`off` if observed, else `on`), and is trained
with stochastic masking of observed OFF scores to force reconstruction of
missing OFF states.

**Expected gain:** The current GBDT ceiling is **6.040 pooled RMSE** (08). The
Transformer's ability to propagate denoised signal across a patient's visits —
especially filling in the 42 % missing OFF regime — should push this below 6.0.

**Files created / changed:**
- `src/parkinson/dataset.py` — `ParkinsonDataset` + `collate_visits` for
  PyTorch DataLoader
- `src/parkinson/seqnet.py` — `SeqNet` model module
- `experiments/09_sequence_transformer.py` — training + CV evaluation script
- `journal/09_sequence_transformer.md` — design note
- `tests/test_seqnet.py` — unit tests for tensor shapes, masking, determinism
- `requirements.txt` — add `torch`

**Constraints carried forward:**
- GroupKFold(n_splits=5) by patient_id — no patient leakage.
- Predictions clipped to [0, 132].
- Hub workspace: parKINGson, project: bobathon-esilv.
- ruff lint + type annotations on all new code.
- Comparison baseline: 08_clinical_gbdt pooled RMSE = 6.040.

---

## Data Understanding for Sequence Modelling

Raw columns available per visit:
```
patient_id  cohort  sexM  gene  age_at_diagnosis  age
ledd  time_since_intake_on  time_since_intake_off
rater_id  on  off
```

For `SeqNet`, we use **raw columns only** (not the full `build_features` output)
because per-patient aggregate stats in `build_features` (e.g. `patient_on_mean`)
embed global context that should emerge from self-attention instead.

The `base` anchor per visit:
```
base = off  if off is not NaN
base = on   if off is NaN (and on is not NaN)
base = grand_mean_target  if both are NaN  (fallback)
```

Continuous features (z-score standardized per training fold):
  age, age_at_diagnosis, ledd, time_since_intake_on, time_since_intake_off,
  on, off, base

Categorical features (integer-encoded, passed to embedding layers):
  cohort (2 values), sexM (binary), gene (4 + null = 5 values), rater_id (50)

Missingness indicator flags (binary, never standardized):
  off_is_missing, on_is_missing, ledd_is_missing

Temporal axis: `age` is used as the continuous time coordinate within each
patient sequence (sorted ascending). `Δage` between consecutive visits is the
temporal distance fed to sinusoidal temporal embeddings.

Sequence length: 1–12 visits per patient (median 7). Sequences padded to the
maximum length in each mini-batch.

---

## Sub-Task 1 — Add torch dependency

**Status:** `[ ] pending`

**Intent:** Install PyTorch and record it in requirements.txt.

**Expected Outcomes:**
- `import torch` succeeds in the venv.
- `requirements.txt` updated with `torch`.

**Todo List:**
1. Run `.venv/bin/pip install torch` (CPU wheel; the device selection code in
   the experiment will automatically pick MPS on Apple Silicon).
2. Add `torch` to `requirements.txt`.
3. Verify: `.venv/bin/python -c "import torch; print(torch.__version__)"`.

---

## Sub-Task 2 — Implement `src/parkinson/dataset.py`

**Status:** `[ ] pending`

**Intent:** Provide a PyTorch `Dataset` that groups visits by patient, sorts
them chronologically by `age`, assembles per-visit feature tensors, and
implements stochastic masking of the OFF score during training.

**Expected Outcomes:**
- `ParkinsonDataset` is instantiable from an X DataFrame + y Series (or y=None
  for test).
- `__getitem__` returns a dict with keys `features`, `base`, `target` (NaN for
  test), `off_mask` (bool: True where off is observed), `age` (for temporal
  embeddings).
- `collate_visits` pads variable-length sequences to the batch maximum length
  and returns boolean `attention_mask` (True = valid token, False = padding).
- A `get_stats` class method (or separate function) returns the z-score
  normalization statistics fitted on the training split.

**Design Details:**

```
Per-visit feature vector (d_feat = ~11 continuous + categorical embeddings):
  Continuous (z-scored): age, age_at_diagnosis, ledd,
                         time_since_intake_on, time_since_intake_off,
                         on_filled, off_filled, base
  Binary flags:          off_is_missing, on_is_missing, ledd_is_missing
  Categorical IDs:       cohort_id, sexM, gene_id, rater_id_enc
    (integer indices for nn.Embedding; NOT one-hot in the tensor)
```

For continuous missingness: fill NaN with 0 BEFORE standardization, then use
the missingness flags as side channels. The model sees (filled_value,
is_missing) pairs, not raw NaN.

Stochastic masking (training only, controlled by `mask_prob=0.35`):
- For each visit where `off_is_missing == 0` (off observed), sample a Bernoulli
  with p=mask_prob.
- If drawn: set `off_filled = 0.0` (mask token) and `off_is_missing = 1`
  (indicate mask). The `base` anchor for masked visits becomes `on` (as if
  off were missing), and the target residual is still computed against the
  true target.
- Only applied during training (pass `augment=True` to `__getitem__` or
  control via a dataset-level flag).

**Δage temporal feature:** Compute `delta_age = age[t] - age[t-1]` for t > 0,
with `delta_age[0] = 0`. This 1-D series is passed separately to `SeqNet` for
the sinusoidal temporal embedding projection.

**Todo List:**
1. Create `src/parkinson/dataset.py`.
2. Implement `VisitNormalizer` dataclass: holds mean/std per continuous column
   computed from training data; `fit(df)` / `transform(df)` API.
3. Implement `CategoricalEncoder` dataclass: maps each categorical column to a
   0-based integer vocab; `fit(df)` / `transform(df)` (returns integer Series).
   Handle unknown categories with an `<unk>` index.
4. Implement `ParkinsonDataset(Dataset)`:
   - `__init__`: receives normalized X, optional y, `augment: bool`.
   - Internally groups by `patient_id`, sorts by `age`.
   - `__getitem__(idx)` returns a dict for patient `idx` with:
     `features` (T × d_cont), `cat_ids` (T × n_cat), `age` (T,),
     `delta_age` (T,), `base` (T,), `target` (T,) [NaN-filled 0 for test],
     `off_is_missing` (T bool), `valid_target_mask` (T bool — False for test).
5. Implement `collate_visits(batch)` function: pads all tensors to max T in
   batch, returns `attention_mask` (B × T_max bool, True=valid).
6. Run `ruff check` and `ruff format`.

**Relevant Context:**
- Raw CSV columns: see "Data Understanding" above.
- Categorical vocab sizes (approx): cohort=2, sexM=2, gene=5, rater_id=50.
  Add 1 for `<unk>` per category → embedding sizes: [3, 3, 6, 51].
- `on` has 29.6% missingness; `ledd` has 36.6%; `time_since_intake_off` 78.8%.
- No datetime column — `age` (float) is the temporal axis.

---

## Sub-Task 3 — Implement `src/parkinson/seqnet.py`

**Status:** `[ ] pending`

**Intent:** Define the `SeqNet` PyTorch module: a bi-directional Transformer
encoder that operates over a padded patient visit sequence and outputs a
scalar residual per visit.

**Expected Outcomes:**
- `SeqNet` is instantiable, `forward()` accepts (features, cat_ids, delta_age,
  attention_mask) and returns predictions of shape (B, T).
- The anchoring formula `ŷ = base + 20 × r̂` is applied inside `forward()`.
- Padding positions are zeroed out in the output (via the attention mask).
- Module passes a single forward pass without errors for a random batch.

**Architecture:**

```
Input per visit:
  x_cont  (B, T, d_cont)   ← continuous features after z-score
  x_cat   (B, T, n_cat)    ← integer category IDs
  delta_age (B, T)          ← Δage intervals

Step 1 — Categorical embeddings:
  Each categorical column → nn.Embedding → concat → project to d_cat
  d_cat = sum of per-column embedding dims (e.g. 4+4+8+16 = 32)

Step 2 — Continuous linear projection:
  x_cont (B, T, d_cont) → nn.Linear(d_cont, d_model - d_cat - d_time)

Step 3 — Sinusoidal temporal embedding:
  Compute sinusoidal encoding of delta_age via a learned frequency basis:
    ω_k = 1 / 10000^(2k/d_time)  for k in [0, d_time//2)
    sin(ω_k * Δage) and cos(ω_k * Δage) → concat → shape (B, T, d_time)
  d_time = 16 (8 sin + 8 cos pairs)

Step 4 — Token embedding:
  Concat [proj_cont, embed_cat, sin_temp] → (B, T, d_model) where
  d_model = d_cont_proj + d_cat + d_time = e.g. 128

Step 5 — TransformerEncoder:
  nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
      d_model=d_model, nhead=4, dim_feedforward=256,
      dropout=0.1, batch_first=True
    ),
    num_layers=2
  )
  Pass src_key_padding_mask = ~attention_mask  (True = pad, ignore)

Step 6 — Residual head:
  nn.Linear(d_model, 1) → squeeze → r̂ (B, T)

Step 7 — Anchored prediction:
  ŷ = base + 20 * r̂
  Zero out padding positions in ŷ.
```

**Parameters (constructor):**
```python
SeqNet(
    d_cont: int,        # number of continuous input features
    cat_vocab_sizes: list[int],  # vocab size per categorical column
    cat_embed_dims: list[int],   # embedding dim per categorical column
    d_time: int = 16,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
)
```

**Todo List:**
1. Create `src/parkinson/seqnet.py`.
2. Implement `SinusoidalTemporalEmbedding(nn.Module)` submodule: accepts
   `delta_age (B, T)`, returns `(B, T, d_time)`.
3. Implement `SeqNet(nn.Module)` as described above.
4. Ensure `forward(features, cat_ids, delta_age, attention_mask, base)` returns
   `predictions (B, T)` — padded positions output 0.
5. Run `ruff check` and `ruff format`.

**Relevant Context:**
- `nn.TransformerEncoder` with `batch_first=True` and `src_key_padding_mask`:
  True in the mask means "ignore this position" (PyTorch convention is inverted
  from the attention_mask in `collate_visits`). Pass `~attention_mask`.
- `d_cont`: 8 continuous columns = age, age_at_diagnosis, ledd,
  time_since_intake_on, time_since_intake_off, on_filled, off_filled, base
  + 3 binary flags (off_is_missing, on_is_missing, ledd_is_missing) = 11 total.
- Categorical columns: cohort (embed_dim=4), sexM (embed_dim=4),
  gene (embed_dim=8), rater_id (embed_dim=16) → total cat embedding = 32.
- d_model = 128: d_cont_proj = 128 - 32 - 16 = 80.

---

## Sub-Task 4 — Implement `experiments/09_sequence_transformer.py`

**Status:** `[ ] pending`

**Intent:** Training script that runs 5-fold patient-grouped CV, computes the
three RMSE metrics, pushes to Skore Hub, writes the submission CSV, and
conditionally updates `00_best_model.py`.

**Expected Outcomes:**
- Script runs end-to-end: CV loop trains 5 fold models, prints metrics.
- Report pushed to Hub as `"09_sequence_transformer"`.
- `submissions/09_sequence_transformer.csv` written.
- If pooled RMSE < 6.040 (baseline 08), `00_best_model.py` updated and
  `submissions/best_submission.csv` regenerated.

**Training Loop Design:**

```
For each fold:
  1. Fit VisitNormalizer + CategoricalEncoder on training fold.
  2. Build ParkinsonDataset(train, augment=True) and DataLoader(shuffle=True).
  3. Build ParkinsonDataset(val, augment=False) and DataLoader(shuffle=False).
  4. Instantiate SeqNet, move to device.
  5. Optimizer: AdamW(lr=1e-3, weight_decay=1e-4).
  6. Scheduler: OneCycleLR(max_lr=1e-3, epochs=EPOCHS, steps_per_epoch=...).
  7. Train EPOCHS epochs:
     - Forward → anchored predictions → MSE loss on non-padding, non-test tokens.
     - Backward → step optimizer + scheduler.
  8. Evaluate on val set (augment=False): collect predictions per visit,
     compute pooled / branch-A / branch-B RMSE.
```

**Hyperparameters (module-level constants):**
```python
EPOCHS = 30
BATCH_SIZE = 32   # patients per batch
LR = 1e-3
WEIGHT_DECAY = 1e-4
MASK_PROB = 0.35
RESIDUAL_SCALE = 20.0
BASELINE_RMSE_08 = 6.040
```

**Skore Hub reporting:**
- The `run()` function retrains SeqNet on the full training set (no CV),
  predicts on test, and calls `project.put("best_submission", ...)`.
- For the experiment key, use a `CrossValidationReport`-compatible wrapper or
  push a custom dict summary (since `skore.evaluate` is sklearn-pipeline-only).
  Use `project.put("09_sequence_transformer", {"metrics": metrics_dict})`
  as a fallback since SeqNet is a PyTorch module, not a sklearn estimator.

**Submission prediction (inference mode):**
- Fit normalizer + encoder on full training data.
- Build test DataLoader (augment=False).
- Run SeqNet in `eval()` mode with `torch.no_grad()`.
- Map predictions back from (patient, visit_t) to the original Index order.
- Clip to [0, 132].

**Todo List:**
1. Create `experiments/09_sequence_transformer.py` as a jupytext `# %%` script.
2. Define module-level constants (hyperparameters, paths).
3. Implement `_train_one_epoch(model, loader, optimizer, scheduler, device)`.
4. Implement `_eval_fold(model, loader, device)` → dict of arrays indexed by
   original row Index for per-regime RMSE.
5. Implement `_cv_evaluate()` → prints fold metrics + returns mean/std dict.
6. Implement `run()` for `00_best_model.py` compatibility.
7. Implement `if __name__ == "__main__"` block.
8. Run `ruff check` and `ruff format`.

**Relevant Context:**
- Device selection: `"mps" if torch.backends.mps.is_available() else "cuda" if
  torch.cuda.is_available() else "cpu"`.
- `OneCycleLR` requires `total_steps` or `epochs + steps_per_epoch`; compute
  `steps_per_epoch = len(train_loader)` before instantiating.
- Experiment 08's `_cv_evaluate` function is a good structural reference.
- The `run()` function pattern follows `07_tuning.py` and `08_clinical_gbdt.py`.

---

## Sub-Task 5 — Implement `tests/test_seqnet.py`

**Status:** `[ ] pending`

**Intent:** Lightweight unit tests verifying tensor shapes, attention masking
behavior, and forward/backward determinism.

**Expected Outcomes:**
- `pytest tests/test_seqnet.py` passes with zero errors.
- Tests cover: forward pass shape, padding masking, masked OFF reconstruction,
  gradient flow.

**Tests to implement:**
1. `test_seqnet_forward_shape`: create a random batch of 3 patients with
   variable sequence lengths [5, 3, 7], collate, forward → assert output shape
   is (3, 7) and padding positions are 0.
2. `test_attention_mask_padding`: verify that changing padding values in the
   input has no effect on the non-padding output tokens (the mask blocks them).
3. `test_off_mask_stochastic`: verify that `ParkinsonDataset` with `augment=True`
   produces at least some masked OFF visits, and that those visits have
   `off_is_missing=1` in the returned tensor.
4. `test_backward_determinism`: with fixed seeds, two forward+backward passes
   on identical batches produce identical gradients.

**Todo List:**
1. Create `tests/test_seqnet.py`.
2. Add a `_make_random_batch()` helper that constructs a valid collated batch
   without loading the real CSV data.
3. Implement the four tests above.
4. Run `pytest tests/test_seqnet.py` to confirm all pass.

---

## Sub-Task 6 — Write `journal/09_sequence_transformer.md` and update `JOURNAL.md`

**Status:** `[ ] pending`

**Intent:** Record the design rationale, method, and results for experiment 14.

**Todo List:**
1. Create `journal/09_sequence_transformer.md` with intent, method, acceptance
   criteria, and (after run) recorded metrics.
2. Add row for `09_sequence_transformer` in `journal/JOURNAL.md` History table.
3. After the experiment runs, fill in actual RMSE numbers and Hub URL.

---

## Sub-Task 7 — Run setup checks and validate end-to-end

**Status:** `[ ] pending`

**Intent:** Confirm all code is correct and the experiment runs cleanly.

**Todo List:**
1. Run `.venv/bin/python -m pytest tests/test_setup.py` — all pass.
2. Run `.venv/bin/python -m pytest tests/test_seqnet.py` — all pass.
3. Run `.venv/bin/python experiments/09_sequence_transformer.py`.
4. Run `.venv/bin/ruff check src/parkinson/ experiments/09_sequence_transformer.py tests/test_seqnet.py`.
5. Record final metrics in `journal/09_sequence_transformer.md`.

---

## Implementation Order

```
Sub-Task 1 → Sub-Task 2 → Sub-Task 3 → Sub-Task 5 → Sub-Task 4 → Sub-Task 6 → Sub-Task 7
```

Sub-Tasks 2 and 3 can be developed in parallel (Dataset and SeqNet are
independent). Sub-Task 5 (tests) is written after Sub-Tasks 2 and 3 are
complete but before Sub-Task 4 runs end-to-end. Sub-Task 6 (journal) is
drafted before Sub-Task 4 and finalized after.

---

## Open Design Notes for Implementation

### Vocab size calculation
At fit time, `CategoricalEncoder` scans training data:
- `cohort`: likely 2 values (A, B) → vocab_size = 3 (+ unk)
- `sexM`: binary 0/1 → can be left as raw int (no embedding needed; include
  in continuous features as binary)
- `gene`: 4 known values + NaN → vocab_size = 6
- `rater_id`: 50 known raters + unk → vocab_size = 51

`sexM` is best treated as a continuous binary feature (no embedding), reducing
the categorical embedding list to [cohort, gene, rater_id] with dims [4, 8, 16]
= 28. Then d_cont = 11 + 1 (sexM) = 12 continuous features.
d_model = 128: d_cont_proj = 128 - 28 - 16 = 84.

### Loss mask
The loss must be masked against:
1. Padding positions (attention_mask == False).
2. In inference mode (test set), positions where we have no true target.
During training, ALL non-padding positions contribute to the loss, including
stochastically masked OFF visits (the whole point of the masking).

### `run()` function for `00_best_model.py`
Since `00_best_model.py` does a dynamic import and calls `run()`, the function
must be importable without triggering training. All training code lives inside
`if __name__ == "__main__"` or inside `_cv_evaluate()`. The `run()` function
itself trains on full data and writes the submission.

### Experiment numbering gap (08 → 14)
The user explicitly specified experiment 14. The gap (09–13) is intentional —
treat as reserved for intermediate experiments. No action needed.
