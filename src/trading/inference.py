"""Inference and threshold-tuning helpers for the Stage 1 detector."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset


def predict_sequence_probabilities(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Return positive-class probabilities for sequence windows."""
    if len(X) == 0:
        return np.empty(0, dtype=float)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    probs = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(batch_probs)

    return np.concatenate(probs).astype(float)


def choose_detection_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    cfg: dict,
) -> tuple[float, list[dict], dict]:
    """Choose a Stage 1 threshold that favors recall with a precision guard."""
    search_cfg = cfg["product"]["stage1"]["threshold_search"]
    thresholds = np.linspace(
        search_cfg["min"],
        search_cfg["max"],
        int(search_cfg["num"]),
    )
    precision_floor = float(search_cfg.get("precision_floor", 0.0))

    rows = []
    for threshold in thresholds:
        metrics = compute_binary_metrics(y_true, probs, float(threshold))
        rows.append(metrics)

    valid = [row for row in rows if row["precision"] >= precision_floor]
    if not valid:
        valid = rows

    best = max(
        valid,
        key=lambda row: (
            row["recall"],
            row["precision"],
            row["f1"],
            -row["threshold"],
        ),
    )
    return float(best["threshold"]), rows, best


def compute_binary_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict:
    """Compute thresholded binary-classification metrics."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    y_pred = (probs >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "predicted_positive": int(y_pred.sum()),
    }


def build_thresholded_report(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    positive_name: str,
) -> dict:
    """Return a classification report for a tuned probability threshold."""
    y_pred = (np.asarray(probs) >= threshold).astype(int)
    return classification_report(
        np.asarray(y_true, dtype=int),
        y_pred,
        target_names=["Negative", positive_name],
        output_dict=True,
        zero_division=0,
    )

