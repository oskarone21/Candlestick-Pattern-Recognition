from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


def _mps_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def _select_device(device_pref: str) -> torch.device:
    pref = (device_pref or "auto").lower()

    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA was explicitly requested but is not available.")

    if pref == "mps":
        if _mps_available():
            return torch.device("mps")
        raise RuntimeError("MPS was explicitly requested but is not available.")

    if pref == "cpu":
        return torch.device("cpu")

    if pref != "auto":
        raise ValueError(f"Unsupported device preference: {device_pref}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def runtime_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    requested_device = str(cfg.get("project", {}).get("device", "auto"))
    device = _select_device(requested_device)
    train_cfg = cfg.get("training", {})
    num_workers = int(train_cfg.get("num_workers", 0))
    use_amp = bool(train_cfg.get("mixed_precision", False)) and device.type == "cuda"

    gpu_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    elif device.type == "mps":
        gpu_name = "Apple Metal"

    return {
        "requested_device": requested_device,
        "selected_device": device.type,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(_mps_available()),
        "gpu_name": gpu_name,
        "mixed_precision_enabled": use_amp,
        "pin_memory": device.type == "cuda",
        "num_workers": num_workers,
        "persistent_workers": bool(train_cfg.get("persistent_workers", False) and num_workers > 0),
        "torch_version": torch.__version__,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_normalization_stats(X_train: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.mean(X_train, axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.std(X_train, axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    return {"mean": mean, "std": std}


def _apply_normalization(X: np.ndarray, stats: dict[str, np.ndarray] | None) -> np.ndarray:
    if not stats:
        return X.astype(np.float32, copy=False)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((X.astype(np.float32, copy=False) - mean) / std).astype(np.float32, copy=False)


@dataclass
class TorchBinaryModel:
    model: nn.Module
    model_name: str
    init_kwargs: dict[str, Any]
    device: torch.device
    normalization_stats: dict[str, np.ndarray] | None = None

    def predict_proba(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        self.model.eval()
        X_norm = _apply_normalization(X, self.normalization_stats)
        ds = SequenceDataset(X_norm)
        pin_memory = self.device.type == "cuda"
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for xb in dl:
                xb = xb.to(self.device, non_blocking=pin_memory)
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.device.type == "cuda"
                    else nullcontext()
                )
                with autocast_context:
                    logits = self.model(xb)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                outputs.append(probs)

        if not outputs:
            return np.array([], dtype=np.float32)
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def save(self, path: str | Path) -> None:
        ckpt = {
            "model_name": self.model_name,
            "init_kwargs": self.init_kwargs,
            "state_dict": self.model.state_dict(),
            "normalization_stats": {
                "mean": self.normalization_stats["mean"],
                "std": self.normalization_stats["std"],
            }
            if self.normalization_stats
            else None,
        }
        torch.save(ckpt, path)


def train_torch_binary(
    model_ctor: Callable[..., nn.Module],
    model_name: str,
    init_kwargs: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict[str, Any],
    seed: int = 42,
    override_params: dict[str, Any] | None = None,
) -> TorchBinaryModel:
    override_params = override_params or {}
    _set_seed(seed)

    train_cfg = cfg.get("training", {})
    opt_cfg = cfg.get("optimizer", {})

    batch_size = int(override_params.get("batch_size", train_cfg.get("batch_size", 64)))
    epochs = int(override_params.get("epochs", train_cfg.get("epochs", 30)))
    patience = int(override_params.get("early_stopping_patience", train_cfg.get("early_stopping_patience", 7)))
    lr = float(override_params.get("lr", opt_cfg.get("lr", 3.0e-4)))
    weight_decay = float(override_params.get("weight_decay", opt_cfg.get("weight_decay", 1.0e-4)))
    num_workers = int(train_cfg.get("num_workers", 0))
    persistent_workers = bool(train_cfg.get("persistent_workers", False) and num_workers > 0)

    device = _select_device(cfg.get("project", {}).get("device", "auto"))
    use_amp = bool(train_cfg.get("mixed_precision", False)) and device.type == "cuda"
    pin_memory = device.type == "cuda"

    normalization_stats = _fit_normalization_stats(X_train)
    X_train_norm = _apply_normalization(X_train, normalization_stats)
    X_val_norm = _apply_normalization(X_val, normalization_stats)

    model = model_ctor(**init_kwargs).to(device)

    train_ds = SequenceDataset(X_train_norm, y_train)
    val_ds = SequenceDataset(X_val_norm, y_val)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    pos = max(float((y_train == 1).sum()), 1.0)
    neg = max(float((y_train == 0).sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_state = None
    best_val_loss = float("inf")
    no_improve = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
                if use_amp
                else nullcontext()
            )
            with autocast_context:
                logits = model(xb)
                loss = criterion(logits, yb)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device, non_blocking=pin_memory)
                yb = yb.to(device, non_blocking=pin_memory)
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
                    if use_amp
                    else nullcontext()
                )
                with autocast_context:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                val_losses.append(float(loss.item()))

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TorchBinaryModel(
        model=model,
        model_name=model_name,
        init_kwargs=init_kwargs,
        device=device,
        normalization_stats=normalization_stats,
    )


def load_torch_binary(path: str | Path, model_ctor: Callable[..., nn.Module], device: str = "cpu") -> TorchBinaryModel:
    ckpt = torch.load(path, map_location=device)
    model = model_ctor(**ckpt["init_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    normalization_stats = ckpt.get("normalization_stats")
    if normalization_stats is not None:
        normalization_stats = {
            "mean": np.asarray(normalization_stats["mean"], dtype=np.float32),
            "std": np.asarray(normalization_stats["std"], dtype=np.float32),
        }
    return TorchBinaryModel(
        model=model,
        model_name=ckpt["model_name"],
        init_kwargs=ckpt["init_kwargs"],
        device=torch.device(device),
        normalization_stats=normalization_stats,
    )
