"""Stage 2 trade-selection model, thresholding, and profitability metrics."""

from __future__ import annotations

import math

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_stage2_model(cfg: dict) -> Pipeline:
    """Build the tabular trade-selection model."""
    model_cfg = cfg["product"]["stage2"]["model"]
    class_weight = model_cfg.get("class_weight", "balanced")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=int(model_cfg.get("max_iter", 2000)),
                    C=float(model_cfg.get("c", 1.0)),
                    class_weight=class_weight,
                    solver=model_cfg.get("solver", "lbfgs"),
                ),
            ),
        ]
    )


def predict_trade_probabilities(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities from the Stage 2 model."""
    if len(X) == 0:
        return np.empty(0, dtype=float)
    return model.predict_proba(X)[:, 1].astype(float)


def choose_trade_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    net_returns: np.ndarray,
    gate_mask: np.ndarray,
    cfg: dict,
) -> tuple[float, list[dict], dict]:
    """Tune the Stage 2 execution threshold for net return with precision guard."""
    search_cfg = cfg["product"]["stage2"]["threshold_search"]
    thresholds = np.linspace(
        search_cfg["min"],
        search_cfg["max"],
        int(search_cfg["num"]),
    )
    precision_floor = float(search_cfg.get("precision_floor", 0.0))
    min_trades = int(search_cfg.get("min_trades", 1))

    rows = []
    for threshold in thresholds:
        metrics = compute_trade_signal_metrics(
            y_true=y_true,
            probs=probs,
            net_returns=net_returns,
            threshold=float(threshold),
            gate_mask=gate_mask,
        )
        rows.append(metrics)

    valid = [
        row for row in rows
        if row["executed_trades"] >= min_trades and row["precision"] >= precision_floor
    ]
    if not valid:
        valid = [row for row in rows if row["executed_trades"] >= 1] or rows

    best = max(
        valid,
        key=lambda row: (
            row["total_net_return_pct"],
            row["precision"],
            row["f1"],
            -row["threshold"],
        ),
    )
    return float(best["threshold"]), rows, best


def compute_trade_signal_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    net_returns: np.ndarray,
    threshold: float,
    gate_mask: np.ndarray | None = None,
) -> dict:
    """Evaluate classification quality and PnL for a thresholded trade signal."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    net_returns = np.asarray(net_returns, dtype=float)
    gate_mask = (
        np.asarray(gate_mask, dtype=bool)
        if gate_mask is not None
        else np.ones(len(y_true), dtype=bool)
    )

    y_pred = gate_mask & (probs >= threshold)
    executed_returns = net_returns[y_pred]

    gross_wins = executed_returns[executed_returns > 0].sum()
    gross_losses = -executed_returns[executed_returns < 0].sum()
    profit_factor = (
        float(gross_wins / gross_losses)
        if gross_losses > 0
        else (math.inf if gross_wins > 0 else 0.0)
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred.astype(int))),
        "precision": float(precision_score(y_true, y_pred.astype(int), zero_division=0)),
        "recall": float(recall_score(y_true, y_pred.astype(int), zero_division=0)),
        "f1": float(f1_score(y_true, y_pred.astype(int), zero_division=0)),
        "gated_candidates": int(gate_mask.sum()),
        "executed_trades": int(y_pred.sum()),
        "win_rate_pct": float(100.0 * (executed_returns > 0).mean()) if len(executed_returns) else 0.0,
        "avg_net_return_pct": float(100.0 * executed_returns.mean()) if len(executed_returns) else 0.0,
        "median_net_return_pct": float(100.0 * np.median(executed_returns)) if len(executed_returns) else 0.0,
        "total_net_return_pct": float(100.0 * executed_returns.sum()) if len(executed_returns) else 0.0,
        "profit_factor": profit_factor,
    }


def build_trade_classification_report(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    gate_mask: np.ndarray,
) -> dict:
    """Return a full Stage 2 classification report for reporting/saving."""
    y_pred = gate_mask & (np.asarray(probs, dtype=float) >= threshold)
    return classification_report(
        np.asarray(y_true, dtype=int),
        y_pred.astype(int),
        target_names=["Skip Trade", "Profitable Trade"],
        output_dict=True,
        zero_division=0,
    )


def extract_stage2_coefficients(model: Pipeline, feature_names: list[str]) -> list[dict]:
    """Return sorted logistic-regression coefficients for explainability."""
    clf = model.named_steps["clf"]
    coefs = clf.coef_.ravel()
    rows = [
        {"feature": name, "coefficient": float(weight), "abs_coefficient": float(abs(weight))}
        for name, weight in zip(feature_names, coefs)
    ]
    return sorted(rows, key=lambda row: row["abs_coefficient"], reverse=True)

