"""
src/training/train.py
─────────────────────────────────────────────────────────────────────────────
Training loop for Cats vs Dogs binary classification.
Logs all params, metrics, and artefacts (confusion matrix, loss curve) to MLflow.
Saves best model checkpoint to models/baseline_cnn.pt
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend (no display needed)
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.data.dataset import build_dataloaders
from src.models.cnn import build_model
from src.models.utils import save_model

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
PARAMS_FILE   = ROOT / "params.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR     = ROOT / "models"
METRICS_FILE  = ROOT / "metrics.json"
ARTIFACTS_DIR = ROOT / "artifacts"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_params() -> dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def plot_loss_curves(
    train_losses: list,
    val_losses: list,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train Loss", marker="o")
    ax.plot(val_losses,   label="Val Loss",   marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (BCE)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Loss curve saved → %s", out_path)


def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    out_path: Path,
) -> None:
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Cat", "Dog"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (Test Set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Confusion matrix saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: str,
    is_train: bool,
) -> tuple[float, float]:
    """Run one epoch. Returns (avg_loss, accuracy)."""
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in tqdm(loader, leave=False):
            imgs   = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)   # (B,1)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds      = (torch.sigmoid(logits) >= 0.5).long()
            correct    += (preds == labels.long()).sum().item()
            total      += imgs.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train() -> None:
    params      = load_params()
    data_cfg    = params["data"]
    model_cfg   = params["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = build_dataloaders(
        processed_dir=PROCESSED_DIR,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        img_size=data_cfg["img_size"],
    )

    # ── Model / Loss / Optimiser ───────────────────────────────────────────────
    model     = build_model(model_cfg["architecture"], model_cfg["dropout"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=model_cfg["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_experiment("cats-vs-dogs")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=model_cfg["architecture"]) as run:
        # Log all hyperparameters
        mlflow.log_params({
            "architecture":  model_cfg["architecture"],
            "learning_rate": model_cfg["learning_rate"],
            "epochs":        model_cfg["epochs"],
            "dropout":       model_cfg["dropout"],
            "batch_size":    data_cfg["batch_size"],
            "img_size":      data_cfg["img_size"],
            "train_split":   data_cfg["train_split"],
            "val_split":     data_cfg["val_split"],
        })

        train_losses, val_losses = [], []
        best_val_loss = float("inf")

        # ── Training loop ─────────────────────────────────────────────────────
        for epoch in range(1, model_cfg["epochs"] + 1):
            log.info("Epoch %d/%d", epoch, model_cfg["epochs"])

            tr_loss, tr_acc = run_epoch(
                model, loaders["train"], criterion, optimizer, device, is_train=True
            )
            val_loss, val_acc = run_epoch(
                model, loaders["val"], criterion, optimizer, device, is_train=False
            )
            scheduler.step(val_loss)

            train_losses.append(tr_loss)
            val_losses.append(val_loss)

            log.info(
                "  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f",
                tr_loss, tr_acc, val_loss, val_acc,
            )

            mlflow.log_metrics({
                "train_loss": tr_loss,
                "train_acc":  tr_acc,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
            }, step=epoch)

            # Save best model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = MODEL_DIR / "baseline_cnn.pt"
                save_model(model, model_path)
                log.info("  ✅ Best model saved (val_loss=%.4f)", best_val_loss)

        # ── Test evaluation ───────────────────────────────────────────────────
        log.info("Evaluating on test set …")
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in loaders["test"]:
                imgs   = imgs.to(device)
                logits = model(imgs)
                preds  = (torch.sigmoid(logits) >= 0.5).long().cpu().squeeze(1)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        log.info("Test accuracy: %.4f", test_acc)
        mlflow.log_metric("test_acc", test_acc)

        # ── Artefacts ─────────────────────────────────────────────────────────
        # 1. Loss curves
        loss_plot = ARTIFACTS_DIR / "loss_curves.png"
        plot_loss_curves(train_losses, val_losses, loss_plot)
        mlflow.log_artifact(str(loss_plot))

        # 2. Confusion matrix
        cm_plot = ARTIFACTS_DIR / "confusion_matrix.png"
        plot_confusion_matrix(all_labels, all_preds, cm_plot)
        mlflow.log_artifact(str(cm_plot))

        # 3. Log best model file
        mlflow.log_artifact(str(MODEL_DIR / "baseline_cnn.pt"), artifact_path="model")

        # ── DVC metrics file ──────────────────────────────────────────────────
        metrics = {
            "test_acc":     round(test_acc, 4),
            "best_val_loss": round(best_val_loss, 4),
        }
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)

        log.info("Run ID: %s", run.info.run_id)
        log.info("✅ Training complete.")


if __name__ == "__main__":
    train()