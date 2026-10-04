"""Evaluation utilities — precision, recall, F1, confusion matrix.

Metrics list driven by ``evaluation.metrics`` in config.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    cfg: dict,
    device: torch.device | None = None,
) -> dict:
    """Full evaluation on the test split.

    Returns
    -------
    dict
        Keys: ``y_true``, ``y_pred``, ``metrics`` (dict of named floats),
        ``classification_report`` (str), ``confusion_matrix`` (array).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            logits = model(X_b)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_b.numpy())
            all_probs.append(probs)

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    class_names = [
        cfg["labeling"]["classes"]["negative_name"],
        cfg["labeling"]["classes"]["positive_name"],
    ]

    metrics: dict[str, float] = {}
    requested = cfg["evaluation"].get("metrics", ["f1_macro"])
    metric_fns = {
        "accuracy": lambda yt, yp: accuracy_score(yt, yp),
        "precision_macro": lambda yt, yp: precision_score(yt, yp, average="macro", zero_division=0),
        "recall_macro": lambda yt, yp: recall_score(yt, yp, average="macro", zero_division=0),
        "f1_macro": lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        "precision_pos": lambda yt, yp: precision_score(yt, yp, pos_label=1, zero_division=0),
        "recall_pos": lambda yt, yp: recall_score(yt, yp, pos_label=1, zero_division=0),
        "f1_pos": lambda yt, yp: f1_score(yt, yp, pos_label=1, zero_division=0),
    }
    # Always report positive-class metrics alongside whatever was requested.
    requested = list(dict.fromkeys(list(requested) + ["precision_pos", "recall_pos", "f1_pos"]))
    for m in requested:
        if m in metric_fns:
            metrics[m] = float(metric_fns[m](y_true, y_pred))

    # Explicitly pass all expected labels so the report doesn't crash
    # when the model predicts only one class (common with imbalanced data).
    all_labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred,
        labels=all_labels,
        target_names=class_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def print_evaluation(result: dict) -> None:
    """Pretty-print evaluation results."""
    print("=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    for k, v in result["metrics"].items():
        print(f"  {k:20s}: {v:.4f}")
    print()
    print(result["classification_report"])
    print("Confusion Matrix:")
    print(result["confusion_matrix"])
    print("=" * 50)


def plot_learning_curves(
    history: dict[str, list[float]],
    save_path: Path | None = None,
) -> None:
    """Plot training/validation loss and F1 curves."""
    if not HAS_PLT:
        print("matplotlib not available — skipping plot")
        return

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "val_f1" in history:
        axes[1].plot(epochs, history["val_f1"], label="Val F1 (macro)", color="green")
    if "val_precision" in history:
        axes[1].plot(epochs, history["val_precision"], label="Val Precision", linestyle="--")
    if "val_recall" in history:
        axes[1].plot(epochs, history["val_recall"], label="Val Recall", linestyle="--")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path | None = None,
) -> None:
    """Plot confusion matrix heatmap."""
    if not HAS_PLT:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["open", "high", "low", "close", "volume"]


def compute_shap_values(
    model: nn.Module,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    n_background: int = 100,
    device: torch.device | None = None,
) -> np.ndarray:
    """Compute SHAP values for the positive class using DeepExplainer.

    Parameters
    ----------
    model : nn.Module
        Trained model (TCN / LSTM / Transformer).
    X_background : array, shape (n, lookback, 5)
        Background reference samples (typically training set).
    X_explain : array, shape (m, lookback, 5)
        Samples to explain (typically test positives).
    n_background : int
        Number of background samples for DeepExplainer (more = slower but stable).
    device : torch.device or None

    Returns
    -------
    shap_vals : array, shape (m, lookback, 5)
        SHAP values for the positive class (class index 1).
    """
    if not HAS_SHAP:
        raise ImportError("shap is not installed. Run: pip install shap")

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    idx = np.random.default_rng(42).choice(len(X_background), size=min(n_background, len(X_background)), replace=False)
    bg = torch.tensor(X_background[idx], dtype=torch.float32).to(device)
    explain = torch.tensor(X_explain, dtype=torch.float32).to(device)

    # GradientExplainer is more compatible with TCN causal convolutions
    # than DeepExplainer (which fails the additivity check on dilated CNNs).
    # GradientExplainer returns shape (n, lookback, features, n_classes)
    explainer = shap.GradientExplainer(model, bg)
    sv = explainer.shap_values(explain)
    sv = np.array(sv)
    if sv.ndim == 4:
        # (n, lookback, features, n_classes) → take positive class
        return sv[:, :, :, 1]
    # Fallback: list of arrays, one per class
    if isinstance(sv, list):
        return np.array(sv[1])
    return sv


def plot_shap_feature_importance(
    shap_vals: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Bar chart of mean |SHAP| aggregated over time steps, per OHLCV feature.

    Parameters
    ----------
    shap_vals : array, shape (n, lookback, 5)
    """
    if not HAS_PLT:
        return
    # Mean absolute SHAP per feature (average over samples and time steps)
    mean_abs = np.abs(shap_vals).mean(axis=(0, 1))   # shape (5,)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(FEATURE_NAMES, mean_abs, color="steelblue")
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP) — Positive Class")
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_shap_timeseries(
    shap_vals: np.ndarray,
    feature_idx: int = 3,
    save_path: Path | None = None,
) -> None:
    """Line plot of mean SHAP over the lookback window for one feature.

    Shows which time steps (relative to breakout bar) matter most.

    Parameters
    ----------
    shap_vals : array, shape (n, lookback, 5)
    feature_idx : int
        0=open, 1=high, 2=low, 3=close, 4=volume
    """
    if not HAS_PLT:
        return
    fname = FEATURE_NAMES[feature_idx]
    vals = shap_vals[:, :, feature_idx]           # (n, lookback)
    mean_shap = vals.mean(axis=0)
    std_shap  = vals.std(axis=0)
    T = len(mean_shap)
    x = np.arange(-T + 1, 1)                      # relative to breakout bar (t=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, mean_shap, color="steelblue", label=f"Mean SHAP ({fname})")
    ax.fill_between(x, mean_shap - std_shap, mean_shap + std_shap, alpha=0.25, color="steelblue")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="red", linewidth=1.0, linestyle="--", label="Breakout bar")
    ax.set_xlabel("Bars relative to breakout")
    ax.set_ylabel("SHAP value")
    ax.set_title(f"SHAP over time — {fname}")
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def threshold_metrics(y_true: np.ndarray, y_prob_pos: np.ndarray, threshold: float) -> dict:
    """Positive-class metrics at a given decision threshold.

    ``y_prob_pos`` is the probability of the positive class. PR-AUC is
    threshold-free and is the most informative single number when positives
    are rare (~1:10 here).
    """
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob_pos) >= threshold).astype(int)
    has_pos = int((y_true == 1).sum()) > 0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob_pos)) if has_pos else 0.0,
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
