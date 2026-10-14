# %% [markdown]
# # 10_retrospective_smoothing — Calibrated Dual-Regime Ensemble
#
# **Team: parKINGson**
# **Members:** Damien CATAU, Malo GERVAISE, Esteban GAUBERT, Maximilien STUDER
#
# Blends `08_clinical_gbdt` (DualBranchGBDT) and `09_sequence_transformer`
# (SeqNet) under a per-regime convex combination:
#
#   ŷ = α * ŷ_SeqNet + (1 − α) * ŷ_GBDT
#
# where α = α_obs when OFF is observed, and α_miss when OFF is missing.
# Ensemble weights are optimized on pooled out-of-fold predictions.

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

BASELINE_RMSE_09: float = 3.308  # SeqNet 5-fold pooled RMSE

DEVICE: torch.device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Columns dropped before GBDT pipeline.
_PIPELINE_DROP = ["Index", "target"]
_MODEL_DROP = ["patient_id"]


# %% [markdown]
# ## Data loading

# %%


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
    y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]
    X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")
    return X_train, X_test, y_train


# %% [markdown]
# ## GBDT fold helpers (inlined from experiment 08)

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


def _gbdt_preprocess_full(
    X_full: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, RaterCalibrator, PatientTrajectoryTransformer]:
    rc = RaterCalibrator(shrinkage=10.0)
    rc.fit(X_full, y)
    X_rc = rc.transform(X_full)
    ptt = PatientTrajectoryTransformer(use_mixedlm=False)
    ptt.fit(X_rc)
    return ptt.transform(X_rc), rc, ptt


# %% [markdown]
# ## SeqNet fold helpers (inlined from experiment 09)

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
    """Predict on validation fold; return (preds, targets, off_missing)."""
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
# ## Weight optimization

# %%


def _optimize_weights(
    preds_gbdt: np.ndarray,
    preds_seqnet: np.ndarray,
    targets: np.ndarray,
    off_missing: np.ndarray,
) -> tuple[float, float]:
    """Perform fine grid search to optimize convex blend weights across clinical regimes.

    Inputs:
        preds_gbdt (np.ndarray): Out-of-fold predictions from DualBranchGBDT.
        preds_seqnet (np.ndarray): Out-of-fold predictions from SeqNet Transformer.
        targets (np.ndarray): True target values (MDS-UPDRS true OFF).
        off_missing (np.ndarray): Boolean mask indicating visits where OFF was missing.

    Outputs:
        tuple[float, float]: (alpha_obs, alpha_miss) that minimizes pooled cross-validation RMSE.
    """
    best_rmse = np.inf
    best_obs, best_miss = 0.5, 0.5
    grid = np.arange(0.0, 1.01, 0.01)
    
    # Evaluate 2D grid of weights: alpha_obs for OFF-present, alpha_miss for OFF-missing
    for a_obs in grid:
        for a_miss in grid:
            alpha = np.where(off_missing, a_miss, a_obs)
            # Clip blended predictions to valid clinical bounds [0, 132]
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
    preds_blend: np.ndarray,
    preds_seqnet: np.ndarray,
    targets: np.ndarray,
    patient_ids: np.ndarray,
    n_bootstrap: int = 10_000,
    rng_seed: int = 42,
) -> float:
    """Clustered bootstrap statistical test: P(RMSE_blend < RMSE_SeqNet).

    Resamples at the patient level (not visit level) to respect longitudinal correlation.
    Pre-aggregates squared errors by patient to compute the clustered test
    in O(n_bootstrap) scalar operations.

    Inputs:
        preds_blend (np.ndarray): Ensemble predictions.
        preds_seqnet (np.ndarray): Benchmark SeqNet predictions.
        targets (np.ndarray): Ground-truth target scores.
        patient_ids (np.ndarray): Patient identifier for each visit.
        n_bootstrap (int): Number of bootstrap iterations. Default 10,000.
        rng_seed (int): Random seed for reproducibility. Default 42.

    Outputs:
        float: Empirical probability that the ensemble outperforms SeqNet.
    """
    rng = np.random.default_rng(rng_seed)
    unique_patients, inverse_indices = np.unique(patient_ids, return_inverse=True)
    n_patients = len(unique_patients)

    # Pre-aggregate squared errors per patient to make bootstrap vectorized and fast
    se_blend = (preds_blend - targets) ** 2
    se_seqnet = (preds_seqnet - targets) ** 2
    patient_se_blend = np.bincount(inverse_indices, weights=se_blend)
    patient_se_seqnet = np.bincount(inverse_indices, weights=se_seqnet)
    patient_diff = patient_se_blend - patient_se_seqnet

    # Count fraction of patient-resampled datasets where blend achieves lower MSE
    n_wins = 0
    for _ in range(n_bootstrap):
        sampled_unique = np.unique(rng.integers(0, n_patients, size=n_patients))
        n_wins += int(patient_diff[sampled_unique].sum() < 0)
    return n_wins / n_bootstrap


# %% [markdown]
# ## 5-fold OOF collection

# %%


def _cv_collect_oof(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, np.ndarray]:
    """Run 5-fold CV collecting pooled OOF predictions from both arms.

    Returns dict with keys:
        preds_gbdt, preds_seqnet, targets, off_missing, patient_ids
    """
    groups = X_train["patient_id"].values
    cv = GroupKFold(n_splits=5)

    all_gbdt: list[np.ndarray] = []
    all_seqnet: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_off_miss: list[np.ndarray] = []
    all_pids: list[np.ndarray] = []

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
        # Merge target into X for prepare_dataframe compatibility.
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
        # build_features needs target to compute features safely
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

        # Fold RMSEs.
        rmse_gbdt = float(np.sqrt(np.mean((preds_gbdt_fold - val_targets) ** 2)))
        rmse_seqnet = float(np.sqrt(np.mean((preds_seqnet_fold - val_targets) ** 2)))
        print(
            f"  GBDT: {rmse_gbdt:.4f}  SeqNet: {rmse_seqnet:.4f}",
            flush=True,
        )

        all_gbdt.append(preds_gbdt_fold)
        all_seqnet.append(preds_seqnet_fold)
        all_targets.append(val_targets)
        all_off_miss.append(val_off_missing)
        all_pids.append(X_val_raw["patient_id"].values)

    return {
        "preds_gbdt": np.concatenate(all_gbdt),
        "preds_seqnet": np.concatenate(all_seqnet),
        "targets": np.concatenate(all_targets),
        "off_missing": np.concatenate(all_off_miss),
        "patient_ids": np.concatenate(all_pids),
    }


# %% [markdown]
# ## run() — called by 00_best_model.py when this experiment is the winner

# %%


def run() -> None:
    """Train full RSR ensemble, write best_submission.csv, push Hub."""
    X_train, X_test, y_train = _load_raw()
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    print("Training RetrospectiveSmoothingRegressor on full dataset...")
    rsr = RetrospectiveSmoothingRegressor(
        alpha_obs=0.78,
        alpha_miss=0.76,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    rsr.fit(X_train, y_train)
    preds = rsr.predict(X_test)

    submission = pd.DataFrame({"Index": X_test["Index"], "target": preds})
    submission.to_csv(submissions_dir / "best_submission.csv", index=False)

    try:
        cfg = load_skore_credentials()
        login(mode="hub")
        project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
        project.put("best_submission", {"experiment": "10_retrospective_smoothing"})
        print(f"best_submission.csv written ({len(submission)} rows) and pushed to Hub.")
    except Exception as exc:  # noqa: BLE001
        print(f"best_submission.csv written ({len(submission)} rows). Hub push skipped: {exc}")


# %% [markdown]
# ## Main: collect OOF → optimize weights → bootstrap → Hub → submission

# %%
if __name__ == "__main__":
    print("=" * 60)
    print("10_retrospective_smoothing — Calibrated dual-regime ensemble")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    X_train, X_test, y_train = _load_raw()
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    # --- Collect 5-fold OOF predictions ---
    print("\n--- 5-fold OOF collection ---")
    oof = _cv_collect_oof(X_train, y_train)

    # --- Optimize blend weights ---
    print("\n--- Optimizing blend weights ---", flush=True)
    alpha_obs, alpha_miss = _optimize_weights(
        oof["preds_gbdt"], oof["preds_seqnet"], oof["targets"], oof["off_missing"]
    )
    print(f"  α_obs  = {alpha_obs:.2f}  (SeqNet weight when OFF observed)", flush=True)
    print(f"  α_miss = {alpha_miss:.2f}  (SeqNet weight when OFF missing)", flush=True)

    # --- Compute blended OOF RMSE ---
    alpha_arr = np.where(oof["off_missing"], alpha_miss, alpha_obs)
    preds_blend = np.clip(
        alpha_arr * oof["preds_seqnet"] + (1.0 - alpha_arr) * oof["preds_gbdt"], 0, 132
    )
    rmse_blend = float(np.sqrt(np.mean((preds_blend - oof["targets"]) ** 2)))

    mask_obs = ~oof["off_missing"]
    mask_miss = oof["off_missing"]
    rmse_a = float(
        np.sqrt(np.mean((preds_blend[mask_obs] - oof["targets"][mask_obs]) ** 2))
    )
    rmse_b = float(
        np.sqrt(np.mean((preds_blend[mask_miss] - oof["targets"][mask_miss]) ** 2))
    )
    rmse_seqnet_oof = float(
        np.sqrt(np.mean((oof["preds_seqnet"] - oof["targets"]) ** 2))
    )
    rmse_gbdt_oof = float(np.sqrt(np.mean((oof["preds_gbdt"] - oof["targets"]) ** 2)))

    delta = BASELINE_RMSE_09 - rmse_blend
    pct = 100.0 * delta / BASELINE_RMSE_09

    print(f"\n  GBDT OOF RMSE   : {rmse_gbdt_oof:.4f}", flush=True)
    print(f"  SeqNet OOF RMSE : {rmse_seqnet_oof:.4f}", flush=True)
    print(f"  Blend OOF RMSE  : {rmse_blend:.4f}", flush=True)
    print(f"  Branch-A (off observed) : {rmse_a:.4f}", flush=True)
    print(f"  Branch-B (off missing)  : {rmse_b:.4f}", flush=True)
    print(
        f"  vs 09_sequence_transformer ({BASELINE_RMSE_09:.4f}): Δ = {delta:+.4f} ({pct:+.1f}%)",
        flush=True,
    )

    # --- Clustered bootstrap test ---
    print("\n--- Clustered bootstrap test (10,000 resamples) ---", flush=True)
    p_superior = _bootstrap_superiority(
        preds_blend,
        oof["preds_seqnet"],
        oof["targets"],
        oof["patient_ids"],
    )
    print(f"  P(RMSE_blend < RMSE_SeqNet) = {p_superior:.4f}", flush=True)

    # --- Train full ensemble with optimized weights ---
    print("\n--- Training full RSR on all training data ---", flush=True)
    rsr = RetrospectiveSmoothingRegressor(
        alpha_obs=alpha_obs,
        alpha_miss=alpha_miss,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    rsr.fit(X_train, y_train)
    preds_test = rsr.predict(X_test)

    submission = pd.DataFrame({"Index": X_test["Index"], "target": preds_test})
    submission.to_csv(submissions_dir / "10_retrospective_smoothing.csv", index=False)
    print(
        f"  Wrote submissions/10_retrospective_smoothing.csv  ({len(submission)} rows)",
        flush=True,
    )

    # --- Push to Hub ---
    metrics_payload: dict = {
        "experiment": "10_retrospective_smoothing",
        "blend_rmse": rmse_blend,
        "branch_a_rmse": rmse_a,
        "branch_b_rmse": rmse_b,
        "seqnet_oof_rmse": rmse_seqnet_oof,
        "gbdt_oof_rmse": rmse_gbdt_oof,
        "alpha_obs": alpha_obs,
        "alpha_miss": alpha_miss,
        "p_superior_vs_seqnet": p_superior,
    }
    try:
        cfg = load_skore_credentials()
        login(mode="hub")
        project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
        project.put("10_retrospective_smoothing", metrics_payload)
        print("  Pushed metrics to Hub as '10_retrospective_smoothing'.")
    except Exception as exc:  # noqa: BLE001
        print(f"  Hub push skipped / failed: {exc}")

    # --- Update 00_best_model.py if blend beats SeqNet ---
    if rmse_blend < BASELINE_RMSE_09:
        best_model_path = PROJECT_ROOT / "experiments" / "00_best_model.py"
        content = best_model_path.read_text()
        old_winner_line = next(
            (
                line
                for line in content.splitlines()
                if line.strip().startswith("WINNER =")
            ),
            None,
        )
        if old_winner_line is not None:
            best_model_path.write_text(
                content.replace(
                    old_winner_line, 'WINNER = "10_retrospective_smoothing.py"'
                )
            )
            submission.to_csv(submissions_dir / "best_submission.csv", index=False)
            print(
                f"\n  ✓ 00_best_model.py → WINNER = '10_retrospective_smoothing.py' "
                f"(blend {rmse_blend:.4f} < SeqNet {BASELINE_RMSE_09:.4f})"
            )
            print(
                f"  ✓ Wrote submissions/best_submission.csv  ({len(submission)} rows)"
            )
        else:
            print(
                "  ⚠ Could not locate WINNER line — update 00_best_model.py manually."
            )
    else:
        print(
            f"\n  09_sequence_transformer remains winner "
            f"(blend {rmse_blend:.4f} ≥ SeqNet baseline {BASELINE_RMSE_09:.4f})."
        )
