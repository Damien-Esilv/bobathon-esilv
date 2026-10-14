# %% [markdown]
# # 11_monotonic_bernstein_trajectory — Monotone Bernstein post-processor
#
# Applies a per-patient degree-3 Bernstein trajectory projector on top of the
# `RetrospectiveSmoothingRegressor` blend (exp 10).  The projector enforces
# non-decreasing motor UPDRS predictions with age while preserving the blend's
# clinical accuracy.
#
# CV loop is identical to experiment 10; the Bernstein projection is applied
# after `preds_blend` is computed in the main block.
#
# Output artefacts:
#   submissions/11_monotonic_bernstein_trajectory.csv  (11 013 rows)

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from skore import Project, login
from skrub import TableVectorizer
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parkinson.dataset import (
    CAT_COLS,
    CONT_COLS,
    CategoricalEncoder,
    VisitNormalizer,
    make_dataloader,
    prepare_dataframe,
)
from parkinson.dual_branch import DualBranchGBDT
from parkinson.features import build_features
from parkinson.hub import load_skore_credentials
from parkinson.mixed_effects import PatientTrajectoryTransformer
from parkinson.pipeline import RetrospectiveSmoothingRegressor
from parkinson.rater import RaterCalibrator
from parkinson.seqnet import build_seqnet
from parkinson.trajectory import MonotonicBernsteinTrajectoryProjector

# %% [markdown]
# ## Constants

# %%
# GBDT hyperparameters (matching experiment 08).
LGBM_PARAMS: dict = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": 1,
}

# SeqNet hyperparameters (matching experiment 09).
EPOCHS: int = 30
BATCH_SIZE: int = 32
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4
MASK_PROB: float = 0.35
RESIDUAL_SCALE: float = 20.0
D_MODEL: int = 128
D_TIME: int = 16
NHEAD: int = 4
NUM_LAYERS: int = 2
DIM_FEEDFORWARD: int = 256
DROPOUT: float = 0.1

BASELINE_RMSE_10: float = 3.303  # exp-10 RSR blend pooled OOF RMSE

DEVICE: torch.device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

_PIPELINE_DROP = ["Index", "target"]
_MODEL_DROP = ["patient_id"]

# 6-regime labels for the clinical breakdown table.
_REGIME_NAMES = {
    0: "Obs - tau_off recorded",
    1: "Obs - tau_off NaN, untreated",
    2: "Obs - tau_off NaN, treated",
    3: "Mis - dense intra-OFF (>=3)",
    4: "Mis - sparse intra-OFF (1-2)",
    5: "Mis - zero intra-OFF",
}


# %% [markdown]
# ## Data loading

# %%


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
    y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]
    X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")
    return X_train, X_test, y_train


# %% [markdown]
# ## 6-regime labelling

# %%


def assign_regimes(X: pd.DataFrame) -> np.ndarray:
    """Assign a clinical regime label (0-5) to each visit in X.

    Parameters
    ----------
    X:
        Visit DataFrame containing ``off``, ``time_since_intake_off``,
        ``ledd``, and ``patient_id`` columns.

    Returns
    -------
    np.ndarray
        Integer array of shape (len(X),) with regime labels 0-5.
    """
    off_obs_count = X.groupby("patient_id")["off"].transform(
        lambda s: s.notna().sum()
    )

    regimes = np.full(len(X), -1, dtype=int)
    off_notna = X["off"].notna().values
    tau_notna = X["time_since_intake_off"].notna().values

    ledd = X["ledd"].values
    ledd_zero_or_nan = np.isnan(ledd) | (ledd == 0)
    ledd_positive = ~np.isnan(ledd) & (ledd > 0)

    obs_count = off_obs_count.values
    off_isna = ~off_notna

    regimes[off_notna & tau_notna] = 0
    regimes[off_notna & ~tau_notna & ledd_zero_or_nan] = 1
    regimes[off_notna & ~tau_notna & ledd_positive] = 2

    regimes[off_isna & (obs_count >= 3)] = 3
    regimes[off_isna & (obs_count >= 1) & (obs_count <= 2)] = 4
    regimes[off_isna & (obs_count == 0)] = 5

    return regimes


# %% [markdown]
# ## GBDT fold helpers (inlined from experiment 08, matching exp 10)

# %%


def _gbdt_preprocess_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit RC + PTT on training fold; return (X_tr_t, X_val_t) with patient_id."""
    rc = RaterCalibrator(shrinkage=10.0)
    rc.fit(X_tr, y_tr)
    X_tr_rc = rc.transform(X_tr)
    X_val_rc = rc.transform(X_val)

    ptt = PatientTrajectoryTransformer(use_mixedlm=False)
    ptt.fit(X_tr_rc)
    return ptt.transform(X_tr_rc), ptt.transform(X_val_rc)


# %% [markdown]
# ## SeqNet fold helpers (inlined from experiment 09, matching exp 10)

# %%


def _seqnet_train_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
) -> tuple[nn.Module, VisitNormalizer, CategoricalEncoder]:
    """Train SeqNet on one fold's training data; return (model_cpu, norm, enc)."""
    torch.manual_seed(42)
    np.random.seed(42)

    df_tr = prepare_dataframe(X_tr, y_tr)
    norm = VisitNormalizer()
    norm.fit(df_tr)
    df_tr = norm.transform(df_tr)
    enc = CategoricalEncoder()
    enc.fit(df_tr)
    df_tr = enc.transform(df_tr)

    loader = make_dataloader(
        df_tr, augment=True, batch_size=BATCH_SIZE, shuffle=True, mask_prob=MASK_PROB
    )
    cat_vocab = [enc.vocab_sizes_[c] for c in CAT_COLS]
    model = build_seqnet(
        len(CONT_COLS),
        cat_vocab,
        D_TIME,
        D_MODEL,
        NHEAD,
        NUM_LAYERS,
        DIM_FEEDFORWARD,
        DROPOUT,
        RESIDUAL_SCALE,
    ).to(DEVICE)

    opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(loader), pct_start=0.1)

    for ep in range(EPOCHS):
        model.train()
        for batch in loader:
            f = batch["features"].to(DEVICE)
            c = batch["cat_ids"].to(DEVICE)
            d = batch["delta_age"].to(DEVICE)
            m = batch["attention_mask"].to(DEVICE)
            b = batch["base"].to(DEVICE)
            t = batch["target"].to(DEVICE)
            v = batch["valid_target"].to(DEVICE)
            p = model(f, c, d, m, b)
            lmask = m & v
            if not lmask.any():
                continue
            loss = ((p - t) ** 2)[lmask].mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
        if (ep + 1) % 10 == 0:
            print(f"    SeqNet epoch {ep + 1}/{EPOCHS}", flush=True)

    model.eval().cpu()
    return model, norm, enc


@torch.no_grad()
def _seqnet_predict_val(
    model: nn.Module,
    X_val: pd.DataFrame,
    norm: VisitNormalizer,
    enc: CategoricalEncoder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict on validation fold; return (preds, targets, off_missing).

    Predictions are re-sorted by Index to align with the Index-sorted order
    used by both targets and the GBDT arm.
    """
    df_val = prepare_dataframe(X_val, y=None)
    df_val = norm.transform(df_val)
    df_val = enc.transform(df_val)
    df_sorted = df_val.sort_values(["patient_id", "age"]).copy()

    loader = make_dataloader(
        df_sorted, augment=False, batch_size=BATCH_SIZE, shuffle=False
    )
    all_preds: list[float] = []
    for batch in loader:
        f = batch["features"]
        c = batch["cat_ids"]
        d = batch["delta_age"]
        m = batch["attention_mask"]
        b = batch["base"]
        p = model(f, c, d, m, b)
        for i in range(p.shape[0]):
            seq_len = int(m[i].sum())
            all_preds.extend(p[i, :seq_len].tolist())

    df_sorted["_pred"] = all_preds
    aligned = df_sorted.sort_values("Index")
    return (
        np.clip(aligned["_pred"].values, 0, 132),
        X_val.sort_values("Index")["target"].values
        if "target" in X_val.columns
        else np.full(len(X_val), np.nan),
        X_val.sort_values("Index")["off"].isna().values,
    )


# %% [markdown]
# ## Weight optimisation (identical to experiment 10)

# %%


def _optimize_weights(
    preds_gbdt: np.ndarray,
    preds_seqnet: np.ndarray,
    targets: np.ndarray,
    off_missing: np.ndarray,
) -> tuple[float, float]:
    """Grid search alpha_obs, alpha_miss in [0, 1] with step 0.01.

    Returns (alpha_obs, alpha_miss) minimising pooled RMSE.
    """
    best_rmse = np.inf
    best_obs, best_miss = 0.5, 0.5
    grid = np.arange(0.0, 1.01, 0.01)
    for a_obs in grid:
        for a_miss in grid:
            alpha = np.where(off_missing, a_miss, a_obs)
            blend = np.clip(alpha * preds_seqnet + (1.0 - alpha) * preds_gbdt, 0, 132)
            rmse = float(np.sqrt(np.mean((blend - targets) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_obs = float(a_obs)
                best_miss = float(a_miss)
    return best_obs, best_miss


# %% [markdown]
# ## Clustered bootstrap test

# %%


def _bootstrap_superiority(
    preds_new: np.ndarray,
    preds_base: np.ndarray,
    targets: np.ndarray,
    patient_ids: np.ndarray,
    n_bootstrap: int = 2_000,
    rng_seed: int = 42,
) -> float:
    """P(RMSE_new < RMSE_base) via patient-grouped bootstrap.

    Resamples at the patient level to respect within-patient correlation.
    """
    rng = np.random.default_rng(rng_seed)
    unique_pids, inv = np.unique(patient_ids, return_inverse=True)
    n_p = len(unique_pids)

    se_new = (preds_new - targets) ** 2
    se_base = (preds_base - targets) ** 2
    pid_se_new = np.bincount(inv, weights=se_new)
    pid_se_base = np.bincount(inv, weights=se_base)
    pid_diff = pid_se_new - pid_se_base

    n_wins = 0
    for _ in range(n_bootstrap):
        sampled_unique = np.unique(rng.integers(0, n_p, size=n_p))
        n_wins += int(pid_diff[sampled_unique].sum() < 0)
    return n_wins / n_bootstrap


# %% [markdown]
# ## 5-fold OOF collection (identical to exp 10, plus val_indices)

# %%


def _cv_collect_oof(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """Run 5-fold CV collecting pooled OOF predictions from both arms.

    Returns dict with keys:
        preds_gbdt, preds_seqnet, targets, off_missing, patient_ids,
        val_indices  (concatenated positional indices into X_train)
    """
    groups = X_train["patient_id"].values
    cv = GroupKFold(n_splits=5)

    all_gbdt: list[np.ndarray] = []
    all_seqnet: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_off_miss: list[np.ndarray] = []
    all_pids: list[np.ndarray] = []
    all_val_idx: list[np.ndarray] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups)):
        print(f"\n=== Fold {fold_idx + 1}/5 ===", flush=True)

        X_tr_raw = X_train.iloc[train_idx].reset_index(drop=True)
        X_val_raw = X_train.iloc[val_idx].reset_index(drop=True)
        y_tr = y_train.iloc[train_idx].reset_index(drop=True)
        y_val = y_train.iloc[val_idx].reset_index(drop=True)

        # Align validation targets and missingness mask.
        val_off_missing = X_val_raw["off"].isna().values
        val_targets = y_val.values

        # --- SeqNet arm (trained first to avoid OpenMP/pthreads collision) ---
        print("  Training SeqNet arm...", flush=True)
        X_tr_with_y = X_tr_raw.copy()
        X_tr_with_y["target"] = y_tr.values
        X_val_with_y = X_val_raw.copy()
        X_val_with_y["target"] = y_val.values

        seqnet_model, norm, enc = _seqnet_train_fold(X_tr_with_y, y_tr)
        preds_seqnet_fold, _, _ = _seqnet_predict_val(
            seqnet_model, X_val_with_y, norm, enc
        )

        # --- GBDT arm ---
        print("  Training GBDT arm...", flush=True)
        X_tr_fe_raw = X_tr_raw.copy()
        X_tr_fe_raw["target"] = y_tr.values
        X_tr_fe = build_features(X_tr_fe_raw).drop(columns=_PIPELINE_DROP)

        X_val_fe_raw = X_val_raw.copy()
        X_val_fe = build_features(X_val_fe_raw).drop(columns=["Index"], errors="ignore")

        X_tr_t, X_val_t = _gbdt_preprocess_fold(X_tr_fe, y_tr, X_val_fe)
        X_tr_model = X_tr_t.drop(columns=_MODEL_DROP, errors="ignore")
        X_val_model = X_val_t.drop(columns=_MODEL_DROP, errors="ignore")

        gbdt_pipe = make_pipeline(TableVectorizer(), DualBranchGBDT(**LGBM_PARAMS))
        gbdt_pipe.fit(X_tr_model, y_tr)
        preds_gbdt_fold = np.clip(gbdt_pipe.predict(X_val_model), 0, 132)

        rmse_gbdt = float(np.sqrt(np.mean((preds_gbdt_fold - val_targets) ** 2)))
        rmse_seqnet = float(np.sqrt(np.mean((preds_seqnet_fold - val_targets) ** 2)))
        print(f"  GBDT: {rmse_gbdt:.4f}  SeqNet: {rmse_seqnet:.4f}", flush=True)

        all_gbdt.append(preds_gbdt_fold)
        all_seqnet.append(preds_seqnet_fold)
        all_targets.append(val_targets)
        all_off_miss.append(val_off_missing)
        all_pids.append(X_val_raw["patient_id"].values)
        all_val_idx.append(val_idx)

    return {
        "preds_gbdt": np.concatenate(all_gbdt),
        "preds_seqnet": np.concatenate(all_seqnet),
        "targets": np.concatenate(all_targets),
        "off_missing": np.concatenate(all_off_miss),
        "patient_ids": np.concatenate(all_pids),
        "val_indices": np.concatenate(all_val_idx),
    }


# %% [markdown]
# ## Utility: RMSE

# %%


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# %% [markdown]
# ## 6-regime evaluation breakdown

# %%


def print_breakdown(
    regimes: np.ndarray,
    targets: np.ndarray,
    blend: np.ndarray,
    bern: np.ndarray,
    X_oof: pd.DataFrame,
) -> None:
    """Print the clinical evaluation table comparing blend vs Bernstein."""
    print("\n" + "=" * 72, flush=True)
    print("Clinical 6-Regime Evaluation (OOF)", flush=True)
    print("=" * 72, flush=True)
    fmt = "{:<38} {:>10} {:>10}"
    print(fmt.format("Slice", "Blend", "Bernstein"), flush=True)
    print("-" * 72, flush=True)

    def _row(label: str, mask: np.ndarray) -> None:
        print(
            fmt.format(
                label,
                f"{_rmse(targets[mask], blend[mask]):.4f}",
                f"{_rmse(targets[mask], bern[mask]):.4f}",
            ),
            flush=True,
        )

    _row("Global Pooled OOF", np.ones(len(targets), dtype=bool))
    _row("Observed-OFF (regimes 0+1+2)", regimes < 3)
    _row("Missing-OFF  (regimes 3+4+5)", regimes >= 3)
    print("-" * 72, flush=True)
    for r in range(6):
        _row(f"  Regime {r}: {_REGIME_NAMES[r]}", regimes == r)
    print("-" * 72, flush=True)

    # Terminal visits: patients with >= 8 visits; only their last visit.
    visit_counts = X_oof.groupby("patient_id")["Index"].transform("count")
    last_visit = X_oof.groupby("patient_id")["age"].transform("max")
    terminal_mask = (
        (visit_counts.values >= 8) & (X_oof["age"].values == last_visit.values)
    )
    _row("Terminal visit (>=8-visit patients)", terminal_mask)

    print("=" * 72, flush=True)


# %% [markdown]
# ## run() — called by 00_best_model.py when this experiment is the winner

# %%


def run() -> None:
    """Train full Bernstein RSR ensemble, write best_submission.csv, push Hub."""
    X_train, X_test, y_train = _load_raw()
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    print("Training RetrospectiveSmoothingRegressor on full dataset...")
    rsr = RetrospectiveSmoothingRegressor(
        alpha_obs=1.00,
        alpha_miss=0.95,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    rsr.fit(X_train, y_train)
    preds_test_raw = rsr.predict(X_test)

    projector = MonotonicBernsteinTrajectoryProjector(
        w_obs_b=3.0, w_obs_a=6.0, w_mis=1.0, lambda_curv=0.01, downward_threshold=3.0
    )
    preds_test_bernstein = np.clip(
        projector.transform_predictions(X_test, preds_test_raw), 0.0, 132.0
    )

    submission = pd.DataFrame(
        {"Index": X_test["Index"], "target": preds_test_bernstein}
    )
    submission.to_csv(submissions_dir / "best_submission.csv", index=False)

    try:
        cfg = load_skore_credentials()
        login(mode="hub")
        project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
        project.put("best_submission", {"experiment": "11_monotonic_bernstein_trajectory"})
        print(f"best_submission.csv written ({len(submission)} rows) and pushed to Hub.")
    except Exception as exc:  # noqa: BLE001
        print(
            f"best_submission.csv written ({len(submission)} rows). Hub push skipped: {exc}"
        )


# %% [markdown]
# ## Main: OOF → blend → Bernstein → breakdown → bootstrap → submission → tracking

# %%
if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("11_monotonic_bernstein_trajectory", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("=" * 60, flush=True)

    X_train, X_test, y_train = _load_raw()

    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    # --- 5-fold OOF collection (identical to exp 10) ---
    print("\n--- 5-fold OOF collection ---", flush=True)
    oof = _cv_collect_oof(X_train, y_train)

    # --- Optimise blend weights (identical to exp 10) ---
    print("\n--- Optimising blend weights ---", flush=True)
    alpha_obs, alpha_miss = _optimize_weights(
        oof["preds_gbdt"],
        oof["preds_seqnet"],
        oof["targets"],
        oof["off_missing"],
    )
    print(f"  alpha_obs  = {alpha_obs:.2f}  (SeqNet weight when OFF observed)", flush=True)
    print(f"  alpha_miss = {alpha_miss:.2f}  (SeqNet weight when OFF missing)", flush=True)

    # --- Compute blended OOF predictions ---
    alpha_arr = np.where(oof["off_missing"], alpha_miss, alpha_obs)
    preds_blend = np.clip(
        alpha_arr * oof["preds_seqnet"] + (1.0 - alpha_arr) * oof["preds_gbdt"],
        0,
        132,
    )
    rmse_blend = _rmse(oof["targets"], preds_blend)
    print(f"\n  Blend OOF RMSE     : {rmse_blend:.4f}", flush=True)

    # --- Apply Bernstein projector on pooled OOF rows ---
    # Rebuild the validation DataFrame in the exact same row order as preds_blend.
    # val_indices are the positional indices into X_train (concatenated across folds
    # in the same order as the OOF arrays), so X_train.iloc[val_indices] gives the
    # rows aligned with preds_blend without any re-sorting.
    df_oof = X_train.iloc[oof["val_indices"]].copy().reset_index(drop=True)

    projector = MonotonicBernsteinTrajectoryProjector(
        w_obs_b=3.0, w_obs_a=6.0, w_mis=1.0, lambda_curv=0.01, downward_threshold=3.0
    )
    preds_bernstein = projector.transform_predictions(df_oof, preds_blend)
    rmse_bernstein = _rmse(oof["targets"], preds_bernstein)
    print(f"  Bernstein OOF RMSE : {rmse_bernstein:.4f}", flush=True)

    delta = BASELINE_RMSE_10 - rmse_bernstein
    pct = 100.0 * delta / BASELINE_RMSE_10
    print(
        f"  vs 10_retrospective_smoothing ({BASELINE_RMSE_10:.4f}): "
        f"D = {delta:+.4f} ({pct:+.1f}%)",
        flush=True,
    )

    # --- 6-regime evaluation breakdown ---
    regimes = assign_regimes(df_oof)
    print_breakdown(regimes, oof["targets"], preds_blend, preds_bernstein, df_oof)

    # --- Strict monotonicity assertion ---
    viol_rate = MonotonicBernsteinTrajectoryProjector.monotonicity_violation_rate(
        df_oof, preds_bernstein
    )
    print(f"\n  Monotonicity violation rate: {viol_rate * 100:.4f}%", flush=True)
    assert viol_rate == 0.0, (
        f"Monotonicity violated: {viol_rate * 100:.4f}% of consecutive pairs are downward"
    )
    print("  Strict monotonicity confirmed (0.00%)", flush=True)

    # --- Paired bootstrap ---
    print("\n--- Paired patient bootstrap (2 000 resamples) ---", flush=True)
    p_sup = _bootstrap_superiority(
        preds_bernstein, preds_blend, oof["targets"], oof["patient_ids"]
    )
    print(f"  P(RMSE_bernstein < RMSE_blend) = {p_sup:.4f}", flush=True)

    # --- Train full model and project test predictions ---
    print("\n--- Training full RSR on all training data ---", flush=True)
    rsr = RetrospectiveSmoothingRegressor(
        alpha_obs=alpha_obs,
        alpha_miss=alpha_miss,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    rsr.fit(X_train, y_train)
    preds_test_raw = rsr.predict(X_test)

    preds_test_bernstein = np.clip(
        projector.transform_predictions(X_test, preds_test_raw), 0.0, 132.0
    )

    # Validate test output.
    assert len(preds_test_bernstein) == len(X_test), (
        f"Expected {len(X_test)} test rows, got {len(preds_test_bernstein)}"
    )
    assert not np.any(np.isnan(preds_test_bernstein)), "NaN in test predictions"
    assert np.all(preds_test_bernstein >= 0.0) and np.all(
        preds_test_bernstein <= 132.0
    ), "Test predictions out of [0, 132]"
    print(
        f"  Test rows: {len(preds_test_bernstein)}  NaN: 0  "
        f"Range: [{preds_test_bernstein.min():.2f}, {preds_test_bernstein.max():.2f}]",
        flush=True,
    )

    pd.DataFrame(
        {"Index": X_test["Index"], "target": preds_test_bernstein}
    ).to_csv(submissions_dir / "11_monotonic_bernstein_trajectory.csv", index=False)
    print(
        f"  Wrote submissions/11_monotonic_bernstein_trajectory.csv "
        f"({len(X_test)} rows)",
        flush=True,
    )

    # --- Push metrics to Hub ---
    metrics_payload: dict = {
        "experiment": "11_monotonic_bernstein_trajectory",
        "bernstein_oof_rmse": rmse_bernstein,
        "blend_oof_rmse": rmse_blend,
        "rmse_obs_off": _rmse(
            oof["targets"][regimes < 3], preds_bernstein[regimes < 3]
        ),
        "rmse_mis_off": _rmse(
            oof["targets"][regimes >= 3], preds_bernstein[regimes >= 3]
        ),
        "p_superior_vs_blend": p_sup,
        "monotonicity_violation_rate": viol_rate,
    }
    try:
        cfg = load_skore_credentials()
        login(mode="hub")
        project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
        project.put("11_monotonic_bernstein_trajectory", metrics_payload)
        print("  Pushed metrics to Hub as '11_monotonic_bernstein_trajectory'.")
    except Exception as exc:  # noqa: BLE001
        print(f"  Hub push skipped / failed: {exc}")

    # --- Update 00_best_model.py if Bernstein beats exp-10 ---
    if rmse_bernstein < BASELINE_RMSE_10:
        best_model_path = PROJECT_ROOT / "experiments" / "00_best_model.py"
        content = best_model_path.read_text()
        old_winner_line = next(
            (ln for ln in content.splitlines() if ln.strip().startswith("WINNER =")),
            None,
        )
        if old_winner_line is not None:
            best_model_path.write_text(
                content.replace(
                    old_winner_line,
                    'WINNER = "11_monotonic_bernstein_trajectory.py"',
                )
            )
            pd.DataFrame(
                {"Index": X_test["Index"], "target": preds_test_bernstein}
            ).to_csv(submissions_dir / "best_submission.csv", index=False)
            print(
                f"\n  00_best_model.py -> WINNER = '11_monotonic_bernstein_trajectory.py' "
                f"(Bernstein {rmse_bernstein:.4f} < blend {BASELINE_RMSE_10:.4f})"
            )
            print(
                f"  Wrote submissions/best_submission.csv ({len(X_test)} rows)"
            )
        else:
            print(
                "  Could not locate WINNER line - update 00_best_model.py manually."
            )
    else:
        print(
            f"\n  10_retrospective_smoothing remains winner "
            f"(Bernstein {rmse_bernstein:.4f} >= blend baseline {BASELINE_RMSE_10:.4f})."
        )
