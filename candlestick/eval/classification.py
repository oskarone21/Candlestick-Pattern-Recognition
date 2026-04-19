from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from candlestick.domain import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_SEED,
    DEFAULT_THRESHOLD_GRID_SIZE,
    METRIC_F1,
    METRIC_F2,
    METRIC_PRECISION,
    METRIC_RECALL,
    SUPPORTED_SELECTION_METRICS,
)


def evaluate_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    positive_support = int((y_true == 1).sum())
    negative_support = int((y_true == 0).sum())

    # Avoid sklearn warning spam when a fold has no positive class.
    pr_auc = 0.0 if positive_support == 0 else float(average_precision_score(y_true, y_prob))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "support": {
            "positive": positive_support,
            "negative": negative_support,
        },
    }


def metric_value(metrics: dict, metric: str) -> float:
    key = (metric or METRIC_F1).lower()
    if key not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(f"Unsupported metric: {metric}")
    return float(metrics[key])


def threshold_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid_size: int = DEFAULT_THRESHOLD_GRID_SIZE,
) -> list[dict]:
    thresholds = np.linspace(0.05, 0.95, grid_size)
    return [evaluate_threshold_metrics(y_true, y_prob, threshold=float(t)) for t in thresholds]


def _ranked_thresholds(
    grid: list[dict],
    primary_metric: str,
    precision_floor: float,
    recall_floor: float,
) -> list[dict]:
    eligible = [
        row
        for row in grid
        if row["precision"] >= precision_floor and row["recall"] >= recall_floor
    ]

    if primary_metric == METRIC_F2:
        if eligible:
            return sorted(
                eligible,
                key=lambda row: (
                    row["recall"],
                    row["f2"],
                    row["precision"],
                    -abs(row["threshold"] - 0.5),
                ),
                reverse=True,
            )
        return sorted(
            grid,
            key=lambda row: (
                row["f2"],
                row["recall"],
                row["precision"],
                -abs(row["threshold"] - 0.5),
            ),
            reverse=True,
        )

    candidates = eligible if eligible else grid
    return sorted(
        candidates,
        key=lambda row: (
            metric_value(row, primary_metric),
            row["recall"],
            row["precision"],
            -abs(row["threshold"] - 0.5),
        ),
        reverse=True,
    )


def choose_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    precision_floor: float = 0.0,
    recall_floor: float = 0.0,
    primary_metric: str = "f1",
    grid_size: int = 181,
) -> tuple[float, dict]:
    """
    Choose a validation threshold and return diagnostics.

    For `primary_metric="f2"`, this becomes a recall-first policy:
    maximize recall under the floors, then tie-break on F2. If the floors
    are unattainable, fall back to the best unconstrained F2 threshold.
    """

    metric_name = (primary_metric or METRIC_F1).lower()
    grid = threshold_grid(y_true=y_true, y_prob=y_prob, grid_size=grid_size)
    ranked = _ranked_thresholds(
        grid=grid,
        primary_metric=metric_name,
        precision_floor=precision_floor,
        recall_floor=recall_floor,
    )
    selected = ranked[0] if ranked else evaluate_threshold_metrics(y_true, y_prob, threshold=0.5)
    diagnostics = {
        "primary_metric": metric_name,
        "precision_floor": float(precision_floor),
        "recall_floor": float(recall_floor),
        "selected_threshold": float(selected["threshold"]),
        "selected_metrics": selected,
        "grid": grid,
    }
    return float(selected["threshold"]), diagnostics


def _metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    if metric == METRIC_PRECISION:
        return lambda y, p: precision_score(y, p, zero_division=0)
    if metric == METRIC_RECALL:
        return lambda y, p: recall_score(y, p, zero_division=0)
    if metric == METRIC_F1:
        return lambda y, p: f1_score(y, p, zero_division=0)
    if metric == METRIC_F2:
        return lambda y, p: fbeta_score(y, p, beta=2.0, zero_division=0)
    raise ValueError(f"Unsupported metric: {metric}")


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    metric: str,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate confidence interval for thresholded metrics with bootstrap."""
    rng = np.random.default_rng(seed)
    fn = _metric_function(metric)

    y_pred = (y_prob >= threshold).astype(int)

    scores: list[float] = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_pred[idx]
        if len(np.unique(y_b)) < 2:
            continue
        scores.append(float(fn(y_b, p_b)))

    if not scores:
        point = float(fn(y_true, y_pred))
        return {"point": point, "ci_low": point, "ci_high": point}

    scores_np = np.asarray(scores)
    return {
        "point": float(fn(y_true, y_pred)),
        "ci_low": float(np.percentile(scores_np, 2.5)),
        "ci_high": float(np.percentile(scores_np, 97.5)),
    }


def attach_confidence_intervals(
    metrics: dict,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict:
    threshold = float(metrics["threshold"])
    out = dict(metrics)
    out["confidence_intervals"] = {
        METRIC_PRECISION: bootstrap_confidence_interval(
            y_true=y_true,
            y_prob=y_prob,
            threshold=threshold,
            metric=METRIC_PRECISION,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        METRIC_RECALL: bootstrap_confidence_interval(
            y_true=y_true,
            y_prob=y_prob,
            threshold=threshold,
            metric=METRIC_RECALL,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        METRIC_F1: bootstrap_confidence_interval(
            y_true=y_true,
            y_prob=y_prob,
            threshold=threshold,
            metric=METRIC_F1,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "f2": bootstrap_confidence_interval(
            y_true=y_true,
            y_prob=y_prob,
            threshold=threshold,
            metric="f2",
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
    }
    return out


def save_metrics_report(report: dict, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
