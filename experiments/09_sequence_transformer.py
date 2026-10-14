# %% [markdown]
# # 09_sequence_transformer — Bi-Directional SeqNet for Patient Visit Timelines
#
# **Team: parKINGson**
# **Members:** Damien CATAU, Malo GERVAISE, Esteban GAUBERT, Maximilien STUDER
#
# Formulates Parkinson's prediction as a sequence modelling problem:
# - Each patient's entire visit history is processed via bi-directional Transformer self-attention.
# - Anchored residual prediction: ŷ = base + 20 × r̂ (where base = off if observed, else on).
# - Stochastic OFF masking (p=0.35) during training teaches the model to reconstruct missing OFF scores.
# - Uses AdamW optimizer with OneCycleLR scheduler.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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

# %% [markdown]
# ## Hyperparameters and constants

# %%
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

BASELINE_RMSE_08: float = 6.040  # pooled RMSE from 08_clinical_gbdt

DEVICE: torch.device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# %% [markdown]
# ## Data loading

# %%


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load raw CSV files.

    Returns
    -------
    X_train, X_test, y_train
    """
    X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
    y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]
    X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")
    return X_train, X_test, y_train


# %% [markdown]
# ## Per-fold preprocessing

# %%


def _build_encoders(
    X_tr: pd.DataFrame, y_tr: pd.Series
) -> tuple[VisitNormalizer, CategoricalEncoder]:
    """Fit VisitNormalizer and CategoricalEncoder on training fold data."""
    df_tr = prepare_dataframe(X_tr, y_tr)
    norm = VisitNormalizer()
    norm.fit(df_tr)
    enc = CategoricalEncoder()
    enc.fit(df_tr)
    return norm, enc


def _apply_encoders(
    X: pd.DataFrame,
    y: pd.Series | None,
    norm: VisitNormalizer,
    enc: CategoricalEncoder,
) -> pd.DataFrame:
    """Prepare, normalize, and integer-encode a DataFrame."""
    df = prepare_dataframe(X, y)
    df = norm.transform(df)
    df = enc.transform(df)
    return df


# %% [markdown]
# ## Training loop

# %%


def _train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: AdamW,
    scheduler: OneCycleLR,
    device: torch.device,
) -> float:
    """Train one epoch; return mean loss."""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        features = batch["features"].to(device)
        cat_ids = batch["cat_ids"].to(device)
        delta_age = batch["delta_age"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        base = batch["base"].to(device)
        target = batch["target"].to(device)
        valid = batch["valid_target"].to(device)  # non-padding + has label

        # Forward.
        preds = model(features, cat_ids, delta_age, attention_mask, base)

        # Loss: MSE on valid (non-padding, has-target) positions.
        loss_mask = attention_mask & valid
        if not loss_mask.any():
            continue

        loss = ((preds - target) ** 2)[loss_mask].mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * int(loss_mask.sum())
        total_tokens += int(loss_mask.sum())

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def _eval_fold(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Evaluate model; return dict of arrays indexed by DataFrame row order.

    Returns
    -------
    dict with keys: "preds", "targets", "off_missing"
        Each is a flat numpy array over all valid (non-padding) visits.
    """
    model.eval()
    all_preds: list[float] = []
    all_targets: list[float] = []
    all_off_missing: list[bool] = []

    for batch in loader:
        features = batch["features"].to(device)
        cat_ids = batch["cat_ids"].to(device)
        delta_age = batch["delta_age"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        base = batch["base"].to(device)
        target = batch["target"]
        valid = batch["valid_target"]
        off_missing = batch["off_is_missing"]

        preds = model(features, cat_ids, delta_age, attention_mask, base)
        preds = preds.cpu()

        # Collect only valid (non-padding, has-target) positions.
        mask = attention_mask.cpu() & valid
        all_preds.extend(preds[mask].tolist())
        all_targets.extend(target[mask].tolist())
        all_off_missing.extend(off_missing[mask].tolist())

    return {
        "preds": np.array(all_preds),
        "targets": np.array(all_targets),
        "off_missing": np.array(all_off_missing, dtype=bool),
    }


# %% [markdown]
# ## Cross-validation evaluation

# %%


def _cv_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, float]:
    """5-fold patient-grouped CV; return mean/std of three RMSE metrics."""
    groups = X_train["patient_id"].values
    cv = GroupKFold(n_splits=5)

    pooled_rmses: list[float] = []
    branch_a_rmses: list[float] = []
    branch_b_rmses: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups)):
        X_tr = X_train.iloc[train_idx].reset_index(drop=True)
        X_val = X_train.iloc[val_idx].reset_index(drop=True)
        y_tr = y_train.iloc[train_idx].reset_index(drop=True)
        y_val = y_train.iloc[val_idx].reset_index(drop=True)

        # Fit per-fold encoders.
        norm, enc = _build_encoders(X_tr, y_tr)
        df_tr = _apply_encoders(X_tr, y_tr, norm, enc)
        df_val = _apply_encoders(X_val, y_val, norm, enc)

        # DataLoaders.
        train_loader = make_dataloader(
            df_tr,
            augment=True,
            batch_size=BATCH_SIZE,
            shuffle=True,
            mask_prob=MASK_PROB,
        )
        val_loader = make_dataloader(
            df_val, augment=False, batch_size=BATCH_SIZE, shuffle=False
        )

        # Model + optimizer + scheduler.
        cat_vocab_sizes = [enc.vocab_sizes_[c] for c in CAT_COLS]
        model = build_seqnet(
            d_cont=len(CONT_COLS),
            cat_vocab_sizes=cat_vocab_sizes,
            d_time=D_TIME,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            residual_scale=RESIDUAL_SCALE,
        ).to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        total_steps = EPOCHS * len(train_loader)
        scheduler = OneCycleLR(
            optimizer, max_lr=LR, total_steps=total_steps, pct_start=0.1
        )

        # Training loop.
        for epoch in range(EPOCHS):
            loss = _train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
            if (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch + 1}/{EPOCHS}  train_loss={loss:.4f}")

        # Evaluation.
        results = _eval_fold(model, val_loader, DEVICE)
        preds = np.clip(results["preds"], 0, 132)
        targets = results["targets"]
        off_missing = results["off_missing"]

        pooled_rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
        pooled_rmses.append(pooled_rmse)

        mask_a = ~off_missing
        mask_b = off_missing
        if mask_a.any():
            branch_a_rmses.append(
                float(np.sqrt(np.mean((preds[mask_a] - targets[mask_a]) ** 2)))
            )
        if mask_b.any():
            branch_b_rmses.append(
                float(np.sqrt(np.mean((preds[mask_b] - targets[mask_b]) ** 2)))
            )

        print(
            f"  Fold {fold_idx + 1}/5 — pooled: {pooled_rmse:.4f}  "
            f"branch-A: {branch_a_rmses[-1]:.4f}  branch-B: {branch_b_rmses[-1]:.4f}"
        )

    return {
        "pooled_mean": float(np.mean(pooled_rmses)),
        "pooled_std": float(np.std(pooled_rmses)),
        "branch_a_mean": float(np.mean(branch_a_rmses)),
        "branch_a_std": float(np.std(branch_a_rmses)),
        "branch_b_mean": float(np.mean(branch_b_rmses)),
        "branch_b_std": float(np.std(branch_b_rmses)),
    }


# %% [markdown]
# ## Inference helpers

# %%


def _train_full(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[nn.Module, VisitNormalizer, CategoricalEncoder]:
    """Fit encoders and train SeqNet on the full training set."""
    norm, enc = _build_encoders(X_train, y_train)
    df_train = _apply_encoders(X_train, y_train, norm, enc)

    train_loader = make_dataloader(
        df_train, augment=True, batch_size=BATCH_SIZE, shuffle=True, mask_prob=MASK_PROB
    )

    cat_vocab_sizes = [enc.vocab_sizes_[c] for c in CAT_COLS]
    model = build_seqnet(
        d_cont=len(CONT_COLS),
        cat_vocab_sizes=cat_vocab_sizes,
        d_time=D_TIME,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        residual_scale=RESIDUAL_SCALE,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps, pct_start=0.1)

    for epoch in range(EPOCHS):
        loss = _train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        if (epoch + 1) % 10 == 0:
            print(f"  full-train epoch {epoch + 1}/{EPOCHS}  loss={loss:.4f}")

    return model, norm, enc


@torch.no_grad()
def _predict_test(
    model: nn.Module,
    X_test: pd.DataFrame,
    norm: VisitNormalizer,
    enc: CategoricalEncoder,
) -> pd.DataFrame:
    """Run inference on X_test; return DataFrame with Index and target columns."""
    model.eval()
    df_test = _apply_encoders(X_test, None, norm, enc)
    test_loader = make_dataloader(
        df_test, augment=False, batch_size=BATCH_SIZE, shuffle=False
    )

    # Collect predictions indexed by patient + visit position.
    # We need to reconstruct the original Index order.
    # Build a lookup: patient_id × visit_rank → (original Index, prediction).
    # The dataset sorts by age within each patient, so we need to track the
    # original row Index through the DataLoader.
    # Simplest approach: sort df_test by patient_id + age, predict, then re-index.
    df_sorted = df_test.sort_values(["patient_id", "age"]).copy()

    all_preds: list[float] = []
    for batch in test_loader:
        features = batch["features"].to(DEVICE)
        cat_ids = batch["cat_ids"].to(DEVICE)
        delta_age = batch["delta_age"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        base = batch["base"].to(DEVICE)

        preds = model(features, cat_ids, delta_age, attention_mask, base)
        preds = preds.cpu()

        # Collect valid (non-padding) predictions in sequence order.
        mask = attention_mask.cpu()
        for i in range(preds.shape[0]):
            seq_len = int(mask[i].sum())
            all_preds.extend(preds[i, :seq_len].tolist())

    preds_arr = np.clip(all_preds, 0, 132)
    df_sorted["target"] = preds_arr
    return df_sorted[["Index", "target"]].sort_values("Index").reset_index(drop=True)


# %% [markdown]
# ## run() — called by 00_best_model.py when this experiment is the winner

# %%


def run() -> None:
    """Train SeqNet on full data, write best_submission.csv, push to Hub."""
    X_train, X_test, y_train = _load_raw()
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    print("Training SeqNet on full training set...")
    model, norm, enc = _train_full(X_train, y_train)

    submission = _predict_test(model, X_test, norm, enc)
    submission.to_csv(submissions_dir / "best_submission.csv", index=False)

    metrics: dict = {
        "experiment": "09_sequence_transformer",
        "note": "full-train prediction",
    }
    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    project.put("best_submission", metrics)
    print(f"best_submission.csv written ({len(submission)} rows) and pushed to Hub.")


# %% [markdown]
# ## Main: CV evaluation → Hub push → submission → conditional winner update

# %%
if __name__ == "__main__":
    print("=" * 60)
    print("09_sequence_transformer — Bi-directional SeqNet")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    X_train, X_test, y_train = _load_raw()
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    # --- 5-fold OOF cross-validation ---
    print("\n--- 5-fold GroupKFold CV ---")
    metrics = _cv_evaluate(X_train, y_train)

    print(
        f"\n  Pooled RMSE  : {metrics['pooled_mean']:.4f} ± {metrics['pooled_std']:.4f}"
    )
    print(
        f"  Branch-A RMSE (off observed): {metrics['branch_a_mean']:.4f} ± {metrics['branch_a_std']:.4f}"
    )
    print(
        f"  Branch-B RMSE (off missing) : {metrics['branch_b_mean']:.4f} ± {metrics['branch_b_std']:.4f}"
    )

    delta = BASELINE_RMSE_08 - metrics["pooled_mean"]
    pct = 100.0 * delta / BASELINE_RMSE_08
    print(
        f"\n  vs 08_clinical_gbdt ({BASELINE_RMSE_08:.4f}): Δ = {delta:+.4f}  ({pct:+.1f}%)"
    )

    # --- Train on full data for submission ---
    print("\n--- Training on full dataset for submission ---")
    model, norm, enc = _train_full(X_train, y_train)

    submission = _predict_test(model, X_test, norm, enc)
    submission.to_csv(submissions_dir / "09_sequence_transformer.csv", index=False)
    print(f"  Wrote submissions/09_sequence_transformer.csv  ({len(submission)} rows)")

    # --- Push metrics to Hub ---
    metrics_payload: dict = {
        "experiment": "09_sequence_transformer",
        "pooled_rmse_mean": metrics["pooled_mean"],
        "pooled_rmse_std": metrics["pooled_std"],
        "branch_a_rmse_mean": metrics["branch_a_mean"],
        "branch_b_rmse_mean": metrics["branch_b_mean"],
        "device": str(DEVICE),
        "epochs": EPOCHS,
    }
    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    project.put("09_sequence_transformer", metrics_payload)
    print("  Pushed metrics to Hub as '09_sequence_transformer'.")

    # --- Conditionally update 00_best_model.py ---
    if metrics["pooled_mean"] < BASELINE_RMSE_08:
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
            new_content = content.replace(
                old_winner_line, 'WINNER = "09_sequence_transformer.py"'
            )
            best_model_path.write_text(new_content)
            submission.to_csv(submissions_dir / "best_submission.csv", index=False)
            print(
                f"\n  ✓ 00_best_model.py updated: WINNER = '09_sequence_transformer.py' "
                f"(pooled RMSE {metrics['pooled_mean']:.4f} < baseline {BASELINE_RMSE_08:.4f})"
            )
            print(
                f"  ✓ Wrote submissions/best_submission.csv  ({len(submission)} rows)"
            )
        else:
            print(
                "  ⚠ Could not locate WINNER line in 00_best_model.py — update manually."
            )
    else:
        print(
            f"\n  08_clinical_gbdt remains the winner "
            f"(pooled RMSE {metrics['pooled_mean']:.4f} ≥ baseline {BASELINE_RMSE_08:.4f})."
        )
