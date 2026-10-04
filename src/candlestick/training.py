"""Training loop — TEAM_STANDARDS §5–§6.

Supports AdamW, AMP mixed precision, early stopping, gradient clipping.
All hyper-parameters from config; metrics logged per epoch.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification (Lin et al. 2017).

    gamma=0 reduces to standard cross-entropy.
    weight applies per-class weighting on top of the focal modulation.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def _resolve_device(cfg: dict) -> torch.device:
    """Pick device following ``project.device`` setting.

    Auto-detection order: CUDA → MPS (Apple Silicon) → CPU.
    """
    choice = cfg["project"].get("device", "auto")
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def _get_git_hash() -> str:
    """Return short git hash for experiment tracking (TEAM_STANDARDS §6)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    class_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Train the model and return metrics history.

    Parameters
    ----------
    model : nn.Module
        Model returned by :func:`model.build_model`.
    train_loader, val_loader : DataLoader
        Training and validation data loaders.
    cfg : dict
        Full resolved config.

    Returns
    -------
    dict
        Keys: ``history``, ``best_state_dict``, ``best_epoch``, ``run_meta``.
    """
    device = _resolve_device(cfg)
    model = model.to(device)

    t_cfg = cfg["training"]
    o_cfg = cfg["optimizer"]
    epochs = t_cfg["epochs"]
    patience = t_cfg.get("early_stopping_patience", 7)
    use_amp = t_cfg.get("mixed_precision", True) and device.type == "cuda"

    # Optimizer
    opt_name = o_cfg.get("name", "adamw")
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=o_cfg["lr"],
            weight_decay=o_cfg.get("weight_decay", 1e-4),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=o_cfg["lr"],
            weight_decay=o_cfg.get("weight_decay", 1e-4),
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience // 2),
    )

    # Loss function — selected by training.loss config key
    weights_tensor: torch.Tensor | None = None
    if class_weights is not None:
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    loss_type = t_cfg.get("loss", "cross_entropy")
    if loss_type == "focal":
        focal_gamma = t_cfg.get("focal_loss", {}).get("gamma", 2.0)
        criterion: nn.Module = FocalLoss(gamma=focal_gamma, weight=weights_tensor)
    elif loss_type == "weighted_cross_entropy" and weights_tensor is not None:
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "val_f1": [], "val_precision": [], "val_recall": [], "val_accuracy": [],
    }
    best_val_loss = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    wait = 0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        n_batches = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(X_b)
                loss = criterion(logits, y_b)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        history["train_loss"].append(train_loss)

        # --- Validate ---
        val_loss, val_metrics = _evaluate_epoch(model, val_loader, criterion, device, use_amp)
        history["val_loss"].append(val_loss)
        for k in val_metrics:
            history.setdefault(k, []).append(val_metrics[k])

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train loss {train_loss:.4f} | "
                f"Val loss {val_loss:.4f} | "
                f"Val F1(macro) {val_metrics['val_f1']:.4f} | "
                f"Val F1(pos) {val_metrics['val_f1_pos']:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} (best={best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    return {
        "history": history,
        "best_state_dict": best_state,
        "best_epoch": best_epoch,
        "run_meta": {
            "git_hash": _get_git_hash(),
            "device": str(device),
            "epochs_run": len(history["train_loss"]),
            "best_val_loss": best_val_loss,
        },
    }


def _evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, dict[str, float]]:
    """Run one validation pass and return loss + classification metrics."""
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    n_batches = 0

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(X_b)
                loss = criterion(logits, y_b)
            running_loss += loss.item()
            n_batches += 1
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_b.cpu().numpy())

    val_loss = running_loss / max(n_batches, 1)
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    metrics = {
        "val_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "val_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "val_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "val_accuracy": accuracy_score(y_true, y_pred),
        # Positive-class metrics: the honest view under heavy class imbalance.
        # Macro F1 averages in the easy negative class and inflates the score.
        "val_f1_pos": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "val_precision_pos": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "val_recall_pos": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    return val_loss, metrics


def save_run_artifacts(
    result: dict,
    cfg: dict,
    output_dir: Path | None = None,
) -> Path:
    """Save model checkpoint, config snapshot, and metrics (TEAM_STANDARDS §6)."""
    if output_dir is None:
        output_dir = Path(cfg["paths"]["outputs_root"]) / cfg["project"]["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint
    if result["best_state_dict"] is not None:
        torch.save(result["best_state_dict"], output_dir / "best_model.pt")

    # Config snapshot
    import yaml
    with open(output_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Metrics
    with open(output_dir / "metrics.json", "w") as f:
        serialisable = {
            k: [float(v) for v in vals] if isinstance(vals, list) else vals
            for k, vals in result["history"].items()
        }
        serialisable["run_meta"] = result["run_meta"]
        json.dump(serialisable, f, indent=2)

    print(f"Artifacts saved to {output_dir}")
    return output_dir
