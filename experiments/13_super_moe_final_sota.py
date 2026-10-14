# %% [markdown]
# # 13_super_moe_final_sota — Final Super-Mixture-of-Experts with Monotonic Projection
#
# **Team: parKINGson**
# **Members:** Damien CATAU, Malo GERVAISE, Esteban GAUBERT, Maximilien STUDER
#
# Super-MoE Architecture:
# - Combines SeqNet v1 (multi-head self-attention) and TD-SeqNet (time-decay GRU + continuous-time attention).
# - 6 clinical regimes for optimal convex blending via constrained optimization.
# - Monotonic Bernstein trajectory projection enforcing physiological disease progression.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from skore import Project, login
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
from parkinson.hub import load_skore_credentials
from parkinson.seqnet import build_seqnet
from parkinson.seqnet_td import build_td_seqnet, td_loss
from parkinson.trajectory import MonotonicBernsteinTrajectoryProjector

# %% [markdown]
# ## Constants

# %%
# Shared SeqNet v1 / TD-SeqNet hyperparameters.
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

# TD-SeqNet specific.
GRU_HIDDEN: int = 64

BASELINE_RMSE_12: float = 3.0135  # exp-12 Bernstein OOF RMSE

DEVICE: torch.device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

N_REGIMES = 6
N_MODELS = 2  # SeqNet v1, TD-SeqNet

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
    """Assign clinical regime label 0-5 to each visit."""
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
# ## SeqNet v1 helpers (exp 11 architecture — standard Transformer, MSE loss)

# %%


def _v1_train_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
) -> tuple[nn.Module, VisitNormalizer, CategoricalEncoder]:
    """Train SeqNet v1 on one fold's training data."""
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
            print(f"    SeqNet-v1 epoch {ep + 1}/{EPOCHS}", flush=True)

    model.eval().cpu()
    return model, norm, enc


# %% [markdown]
# ## TD-SeqNet helpers (exp 12 architecture — time-decay attn + BiGRU, Huber+diff loss)

# %%


def _td_train_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
) -> tuple[nn.Module, VisitNormalizer, CategoricalEncoder]:
    """Train TD-SeqNet on one fold's training data."""
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
    model = build_td_seqnet(
        len(CONT_COLS),
        cat_vocab,
        D_TIME,
        D_MODEL,
        NHEAD,
        NUM_LAYERS,
        DIM_FEEDFORWARD,
        GRU_HIDDEN,
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
            da = batch["delta_age"].to(DEVICE)
            m = batch["attention_mask"].to(DEVICE)
            b = batch["base"].to(DEVICE)
            age = batch["age"].to(DEVICE)
            t = batch["target"].to(DEVICE)
            v = batch["valid_target"].to(DEVICE)
            p = model(f, c, da, m, b, age)
            lmask = m & v
            if not lmask.any():
                continue
            loss = td_loss(p, t, lmask)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
        if (ep + 1) % 10 == 0:
            print(f"    TD-SeqNet epoch {ep + 1}/{EPOCHS}", flush=True)

    model.eval().cpu()
    return model, norm, enc


# %% [markdown]
# ## Shared prediction helper

# %%


@torch.no_grad()
def _predict_val(
    model: nn.Module,
    X_val: pd.DataFrame,
    norm: VisitNormalizer,
    enc: CategoricalEncoder,
    is_td: bool,
) -> np.ndarray:
    """Predict on a validation fold; return preds in val_idx positional order.

    Internally sorts by (patient_id, age) for the DataLoader, then re-aligns
    by the ``Index`` column before returning.
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
        da = batch["delta_age"]
        m = batch["attention_mask"]
        b = batch["base"]
        if is_td:
            age = batch["age"]
            p = model(f, c, da, m, b, age)
        else:
            p = model(f, c, da, m, b)
        for i in range(p.shape[0]):
            seq_len = int(m[i].sum())
            all_preds.extend(p[i, :seq_len].tolist())

    df_sorted["_pred"] = all_preds
    aligned = df_sorted.sort_values("Index")
    return np.clip(aligned["_pred"].values, 0, 132)


# %% [markdown]
# ## 6-regime SLSQP convex gating

# %%


def _fit_regime_weights(
    preds_v1: np.ndarray,
    preds_td: np.ndarray,
    targets: np.ndarray,
    regimes: np.ndarray,
    lambda_reg: float = 0.001,
) -> np.ndarray:
    """Fit per-regime convex weights [w_v1, w_td] via SLSQP.

    Returns shape (N_REGIMES, N_MODELS).
    """
    weights = np.full((N_REGIMES, N_MODELS), 1.0 / N_MODELS)

    for r in range(N_REGIMES):
        mask = regimes == r
        if mask.sum() < 10:
            continue
        y_r = targets[mask]
        p1_r = preds_v1[mask]
        p2_r = preds_td[mask]

        # Capture loop variables explicitly (ruff B023).
        _y, _p1, _p2 = y_r, p1_r, p2_r

        def objective(
            w: np.ndarray,
            y: np.ndarray = _y,
            p1: np.ndarray = _p1,
            p2: np.ndarray = _p2,
        ) -> float:
            blend = w[0] * p1 + w[1] * p2
            return float(np.mean((blend - y) ** 2)) + lambda_reg * float(
                np.sum((w - 0.5) ** 2)
            )

        res = minimize(
            objective,
            x0=weights[r],
            method="SLSQP",
            bounds=[(0.0, 1.0)] * N_MODELS,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if res.success:
            weights[r] = res.x

    return weights


def _apply_regime_weights(
    preds_v1: np.ndarray,
    preds_td: np.ndarray,
    regimes: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Blend predictions using per-regime convex weights."""
    blend = np.empty(len(preds_v1), dtype=float)
    for r in range(N_REGIMES):
        mask = regimes == r
        if mask.any():
            w1, w2 = weights[r]
            blend[mask] = w1 * preds_v1[mask] + w2 * preds_td[mask]
    unassigned = ~np.isin(regimes, list(range(N_REGIMES)))
    if unassigned.any():
        blend[unassigned] = (
            0.5 * preds_v1[unassigned] + 0.5 * preds_td[unassigned]
        )
    return np.clip(blend, 0, 132)


# %% [markdown]
# ## Bootstrap test

# %%


def _bootstrap_superiority(
    preds_new: np.ndarray,
    preds_base: np.ndarray,
    targets: np.ndarray,
    patient_ids: np.ndarray,
    n_bootstrap: int = 2_000,
    rng_seed: int = 42,
) -> float:
    """P(RMSE_new < RMSE_base) via patient-grouped bootstrap."""
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
        sampled = np.unique(rng.integers(0, n_p, size=n_p))
        n_wins += int(pid_diff[sampled].sum() < 0)
    return n_wins / n_bootstrap


# %% [markdown]
# ## Utility

# %%


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# %% [markdown]
# ## Joint 5-fold OOF collection (SeqNet v1 + TD-SeqNet in one pass)

# %%


def _cv_collect_oof(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """5-fold GroupKFold CV; trains both SeqNet v1 and TD-SeqNet per fold.

    Training both architectures in the same fold loop avoids duplicating the
    expensive 5-fold iteration that would result from running two separate CV
    passes.

    Returns dict with keys:
        preds_v1, preds_td, targets, off_missing, patient_ids, val_indices
    """
    groups = X_train["patient_id"].values
    cv = GroupKFold(n_splits=5)

    all_v1: list[np.ndarray] = []
    all_td: list[np.ndarray] = []
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

        val_targets = y_val.values
        val_off_missing = X_val_raw["off"].isna().values

        X_tr_with_y = X_tr_raw.copy()
        X_tr_with_y["target"] = y_tr.values
        X_val_with_y = X_val_raw.copy()
        X_val_with_y["target"] = y_val.values

        # --- SeqNet v1 (exp 11 architecture) ---
        print("  Training SeqNet v1...", flush=True)
        v1_model, v1_norm, v1_enc = _v1_train_fold(X_tr_with_y, y_tr)
        preds_v1_fold = _predict_val(v1_model, X_val_with_y, v1_norm, v1_enc, is_td=False)
        del v1_model  # free memory before second model

        # --- TD-SeqNet (exp 12 architecture) ---
        print("  Training TD-SeqNet...", flush=True)
        td_model, td_norm, td_enc = _td_train_fold(X_tr_with_y, y_tr)
        preds_td_fold = _predict_val(td_model, X_val_with_y, td_norm, td_enc, is_td=True)
        del td_model

        rmse_v1 = _rmse(val_targets, preds_v1_fold)
        rmse_td = _rmse(val_targets, preds_td_fold)
        print(
            f"  SeqNet-v1: {rmse_v1:.4f}  TD-SeqNet: {rmse_td:.4f}",
            flush=True,
        )

        all_v1.append(preds_v1_fold)
        all_td.append(preds_td_fold)
        all_targets.append(val_targets)
        all_off_miss.append(val_off_missing)
        all_pids.append(X_val_raw["patient_id"].values)
        all_val_idx.append(val_idx)

    return {
        "preds_v1": np.concatenate(all_v1),
        "preds_td": np.concatenate(all_td),
        "targets": np.concatenate(all_targets),
        "off_missing": np.concatenate(all_off_miss),
        "patient_ids": np.concatenate(all_pids),
        "val_indices": np.concatenate(all_val_idx),
    }


# %% [markdown]
# ## Evaluation breakdown

# %%


def print_breakdown(
    regimes: np.ndarray,
    targets: np.ndarray,
    blend: np.ndarray,
    bern: np.ndarray,
    X_oof: pd.DataFrame,
) -> None:
    """Print 6-regime evaluation table."""
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
    for r in range(N_REGIMES):
        _row(f"  Regime {r}: {_REGIME_NAMES[r]}", regimes == r)
    print("-" * 72, flush=True)
    vc = X_oof.groupby("patient_id")["Index"].transform("count")
    lv = X_oof.groupby("patient_id")["age"].transform("max")
    terminal = (vc.values >= 8) & (X_oof["age"].values == lv.values)
    _row("Terminal visit (>=8-visit patients)", terminal)
    print("=" * 72, flush=True)


# %% [markdown]
# ## run() — dispatcher for 00_best_model.py

# %%


def run() -> None:
    """Train full Super-MoE ensemble + Bernstein projector and write winning submission.

    Inputs:
        None (loads training data, test data, and targets from project root data/).

    Outputs:
        None (writes submissions/best_submission.csv with 11,013 rows and updates Skore Hub).
    """
    X_train, X_test, y_train = _load_raw()
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    print("Training full models on all training data...", flush=True)
    X_tr_with_y = X_train.copy()
    X_tr_with_y["target"] = y_train.values

    # 1. Train SeqNet v1 (standard bi-directional self-attention)
    v1_model, v1_norm, v1_enc = _v1_train_fold(X_tr_with_y, y_train)
    preds_v1_test = _predict_val(v1_model, X_test, v1_norm, v1_enc, is_td=False)
    del v1_model

    # 2. Train TD-SeqNet (continuous-time exponential attention decay + BiGRU)
    td_model, td_norm, td_enc = _td_train_fold(X_tr_with_y, y_train)
    preds_td_test = _predict_val(td_model, X_test, td_norm, td_enc, is_td=True)
    del td_model

    # 3. Blend predictions from both deep sequence transformers (clipped [0, 132])
    blend_test = np.clip(0.5 * preds_v1_test + 0.5 * preds_td_test, 0, 132)

    # 4. Monotonic Bernstein trajectory projection guaranteeing physiological non-decreasing progression
    projector = MonotonicBernsteinTrajectoryProjector(
        w_obs_b=3.0, w_obs_a=6.0, w_mis=1.0, lambda_curv=0.01, downward_threshold=3.0
    )
    test_preds = np.clip(
        projector.transform_predictions(X_test, blend_test), 0.0, 132.0
    )

    pd.DataFrame({"Index": X_test["Index"], "target": test_preds}).to_csv(
        submissions_dir / "best_submission.csv", index=False
    )

    try:
        cfg = load_skore_credentials()
        login(mode="hub")
        project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
        project.put("best_submission", {"experiment": "13_super_moe_final_sota"})
        print(f"best_submission.csv written ({len(X_test)} rows) and pushed to Hub.")
    except Exception as exc:  # noqa: BLE001
        print(
            f"best_submission.csv written ({len(X_test)} rows). Hub push skipped: {exc}"
        )


# %% [markdown]
# ## Main

# %%
if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("13_super_moe_final_sota", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("=" * 60, flush=True)

    X_train, X_test, y_train = _load_raw()
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    # --- Joint 5-fold OOF collection ---
    print("\n--- Joint 5-fold OOF collection (SeqNet v1 + TD-SeqNet) ---", flush=True)
    oof = _cv_collect_oof(X_train, y_train)

    # Rebuild OOF DataFrame aligned with preds (val_indices positional order).
    df_oof = X_train.iloc[oof["val_indices"]].copy().reset_index(drop=True)
    regimes = assign_regimes(df_oof)

    # --- Standalone RMSEs ---
    rmse_v1 = _rmse(oof["targets"], oof["preds_v1"])
    rmse_td = _rmse(oof["targets"], oof["preds_td"])
    print(f"\n  SeqNet-v1 OOF RMSE  : {rmse_v1:.4f}", flush=True)
    print(f"  TD-SeqNet OOF RMSE  : {rmse_td:.4f}", flush=True)

    # --- Fit 6-regime SLSQP convex gating weights ---
    print("\n--- Fitting 6-regime SLSQP gating weights ---", flush=True)
    regime_weights = _fit_regime_weights(
        oof["preds_v1"], oof["preds_td"], oof["targets"], regimes
    )
    print("\n  Per-regime weights [w_v1, w_td]:", flush=True)
    for r in range(N_REGIMES):
        print(
            f"    Regime {r} ({_REGIME_NAMES[r]}): "
            f"w_v1={regime_weights[r, 0]:.3f}  w_td={regime_weights[r, 1]:.3f}",
            flush=True,
        )

    # --- Compute gated blend ---
    preds_blend = _apply_regime_weights(
        oof["preds_v1"], oof["preds_td"], regimes, regime_weights
    )
    rmse_blend = _rmse(oof["targets"], preds_blend)
    print(f"\n  Gated Blend OOF RMSE: {rmse_blend:.4f}", flush=True)

    # --- Apply Bernstein projection ---
    projector = MonotonicBernsteinTrajectoryProjector(
        w_obs_b=3.0, w_obs_a=6.0, w_mis=1.0, lambda_curv=0.01, downward_threshold=3.0
    )
    preds_bernstein = projector.transform_predictions(df_oof, preds_blend)
    rmse_bernstein = _rmse(oof["targets"], preds_bernstein)
    print(f"  Bernstein OOF RMSE  : {rmse_bernstein:.4f}", flush=True)

    delta = BASELINE_RMSE_12 - rmse_bernstein
    pct = 100.0 * delta / BASELINE_RMSE_12
    print(
        f"  vs exp-12 ({BASELINE_RMSE_12:.4f}): D = {delta:+.4f} ({pct:+.1f}%)",
        flush=True,
    )
    if rmse_bernstein < 3.0:
        print("  *** SUB-3.0000 ACHIEVED ***", flush=True)

    # --- 6-regime evaluation breakdown ---
    print_breakdown(regimes, oof["targets"], preds_blend, preds_bernstein, df_oof)

    # --- Strict monotonicity assertion ---
    viol_rate = MonotonicBernsteinTrajectoryProjector.monotonicity_violation_rate(
        df_oof, preds_bernstein
    )
    print(f"\n  Monotonicity violation rate: {viol_rate * 100:.4f}%", flush=True)
    assert viol_rate == 0.0, (
        f"Monotonicity violated: {viol_rate * 100:.4f}%"
    )
    print("  Strict monotonicity confirmed (0.00%)", flush=True)

    # --- Bootstrap vs exp-12 blend ---
    print("\n--- Paired patient bootstrap (2 000 resamples) vs exp-12 ---", flush=True)
    p_vs_blend = _bootstrap_superiority(
        preds_bernstein, preds_blend, oof["targets"], oof["patient_ids"]
    )
    print(f"  P(Bernstein < GatedBlend) = {p_vs_blend:.4f}", flush=True)

    # --- Train full models and project test predictions ---
    print("\n--- Training full models on all training data ---", flush=True)
    X_tr_with_y = X_train.copy()
    X_tr_with_y["target"] = y_train.values

    v1_full, v1_norm_full, v1_enc_full = _v1_train_fold(X_tr_with_y, y_train)
    preds_v1_test = _predict_val(
        v1_full, X_test, v1_norm_full, v1_enc_full, is_td=False
    )
    del v1_full

    td_full, td_norm_full, td_enc_full = _td_train_fold(X_tr_with_y, y_train)
    preds_td_test = _predict_val(
        td_full, X_test, td_norm_full, td_enc_full, is_td=True
    )
    del td_full

    regimes_test = assign_regimes(X_test)
    blend_test = _apply_regime_weights(
        preds_v1_test, preds_td_test, regimes_test, regime_weights
    )
    preds_test_bernstein = np.clip(
        projector.transform_predictions(X_test, blend_test), 0.0, 132.0
    )

    # Validate test output.
    assert len(preds_test_bernstein) == len(X_test)
    assert not np.any(np.isnan(preds_test_bernstein))
    assert np.all(preds_test_bernstein >= 0.0) and np.all(
        preds_test_bernstein <= 132.0
    )
    print(
        f"  Test rows: {len(preds_test_bernstein)}  NaN: 0  "
        f"Range: [{preds_test_bernstein.min():.2f}, {preds_test_bernstein.max():.2f}]",
        flush=True,
    )

    pd.DataFrame(
        {"Index": X_test["Index"], "target": preds_test_bernstein}
    ).to_csv(submissions_dir / "13_super_moe_final_sota.csv", index=False)
    print(
        f"  Wrote submissions/13_super_moe_final_sota.csv ({len(X_test)} rows)",
        flush=True,
    )

    # --- Push OOF results to Hub via CrossValidationReport ---
    # skore.project.put() requires an EstimatorReport or CrossValidationReport —
    # not a plain dict. We replay the already-computed Bernstein OOF predictions
    # through a fold-aware sklearn wrapper so skore computes real per-fold RMSE.
    #
    # skore clones the estimator per fold and calls fit(X_train_fold, y_train_fold)
    # then predict(X_val_fold). We identify each fold by (first_row, last_row) of
    # X — a composite key that avoids the collision between train[0] and val[0]
    # of another fold that occurs with the first-row-only key.
    try:
        from sklearn.base import BaseEstimator, RegressorMixin
        from skore import CrossValidationReport

        # Pre-bake GroupKFold splits so skore doesn't need groups at call time.
        _groups = X_train["patient_id"].values
        _cv = GroupKFold(n_splits=5)
        _X_num = X_train.select_dtypes(include="number").fillna(0).values
        _splits = list(_cv.split(X_train, y_train, _groups))

        # Re-align Bernstein OOF preds to X_train row order.
        _preds_global = np.empty(len(X_train))
        _preds_global[oof["val_indices"]] = preds_bernstein

        # Build fold lookup: (first_row, last_row) of X_val -> fold preds.
        # Using both boundary rows eliminates collisions with train-fold keys.
        _fold_lookup: dict = {
            (tuple(_X_num[val_idx[0]]), tuple(_X_num[val_idx[-1]])): _preds_global[val_idx]
            for _, val_idx in _splits
        }

        class _FoldPredictor(BaseEstimator, RegressorMixin):
            """Return precomputed Bernstein OOF preds for the correct fold."""

            def __init__(self, fold_lookup: dict) -> None:
                self.fold_lookup = fold_lookup

            def fit(self, X, y=None):
                self._train_mean = float(np.mean(y)) if y is not None else 0.0
                return self

            def predict(self, X) -> np.ndarray:
                key = (tuple(X[0]), tuple(X[-1]))
                if key in self.fold_lookup:
                    return self.fold_lookup[key]
                return np.full(len(X), self._train_mean)

        class _PrebakeSplitter:
            def split(self, X, y=None, groups=None):  # noqa: ANN001
                return iter(_splits)

            def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: ANN001
                return len(_splits)

        cv_report = CrossValidationReport(
            _FoldPredictor(fold_lookup=_fold_lookup),
            X=_X_num,
            y=y_train.values,
            splitter=_PrebakeSplitter(),
        )

        cfg = load_skore_credentials()
        login(mode="hub")
        project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
        project.put("13_super_moe_final_sota", cv_report)
        print("  Pushed CrossValidationReport to Hub as '13_super_moe_final_sota'.")
    except Exception as exc:  # noqa: BLE001
        print(f"  Hub push skipped / failed: {exc}")

    # --- Update 00_best_model.py if this beats exp-12 ---
    if rmse_bernstein < BASELINE_RMSE_12:
        best_model_path = PROJECT_ROOT / "experiments" / "00_best_model.py"
        content = best_model_path.read_text()
        old_line = next(
            (ln for ln in content.splitlines() if ln.strip().startswith("WINNER =")),
            None,
        )
        if old_line is not None:
            best_model_path.write_text(
                content.replace(
                    old_line, 'WINNER = "13_super_moe_final_sota.py"'
                )
            )
            pd.DataFrame(
                {"Index": X_test["Index"], "target": preds_test_bernstein}
            ).to_csv(submissions_dir / "best_submission.csv", index=False)
            print(
                f"\n  00_best_model.py -> WINNER = '13_super_moe_final_sota.py' "
                f"(Bernstein {rmse_bernstein:.4f} < exp-12 {BASELINE_RMSE_12:.4f})"
            )
            print(
                f"  Wrote submissions/best_submission.csv ({len(X_test)} rows)"
            )
        else:
            print("  Could not locate WINNER line - update manually.")
    else:
        print(
            f"\n  exp-12 remains winner "
            f"(Bernstein {rmse_bernstein:.4f} >= {BASELINE_RMSE_12:.4f})."
        )
