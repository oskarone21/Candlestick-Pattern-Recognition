from __future__ import annotations

import json
import math
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

from chart_patterns.domain import (
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_THRESHOLD_GRID_SIZE,
    METRIC_ACCURACY,
    METRIC_F1,
    METRIC_F2,
    METRIC_PRECISION,
    METRIC_PR_AUC,
    METRIC_RECALL,
    SUPPORTED_SELECTION_METRICS,
)


def _wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    if trials <= 0:
        return 0.0

    proportion = successes / trials
    z_sq = z * z
    denominator = 1.0 + (z_sq / trials)
    center = proportion + (z_sq / (2.0 * trials))
    margin = z * math.sqrt((proportion * (1.0 - proportion) + z_sq / (4.0 * trials)) / trials)
    return max(0.0, min(1.0, (center - margin) / denominator))


def _fbeta_from_precision_recall(precision: float, recall: float, beta: float) -> float:
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator <= 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denominator


def evaluate_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    positive_support = int((y_true == 1).sum())
    negative_support = int((y_true == 0).sum())

    # Avoid sklearn warning spam when a fold has no positive class.
    pr_auc = 0.0 if positive_support == 0 else float(average_precision_score(y_true, y_prob))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    predicted_positive_support = int(tp + fp)
    predicted_negative_support = int(tn + fn)
    precision_lower_bound = _wilson_lower_bound(tp, predicted_positive_support)
    recall_lower_bound = _wilson_lower_bound(tp, positive_support)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "precision_lower_bound": float(precision_lower_bound),
        "recall_lower_bound": float(recall_lower_bound),
        "f1_lower_bound": float(_fbeta_from_precision_recall(precision_lower_bound, recall_lower_bound, beta=1.0)),
        "f2_lower_bound": float(_fbeta_from_precision_recall(precision_lower_bound, recall_lower_bound, beta=2.0)),
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
            "predicted_positive": predicted_positive_support,
            "predicted_negative": predicted_negative_support,
        },
    }


def metric_value(metrics: dict, metric: str) -> float:
    key = (metric or METRIC_F1).lower()
    if key not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(f"Unsupported metric: {metric}")
    return float(metrics[key])


def selection_metric_value(metrics: dict, metric: str, conservative: bool = False) -> float:
    key = (metric or METRIC_F1).lower()
    if not conservative:
        return metric_value(metrics, key)

    conservative_aliases = {
        METRIC_PRECISION: "precision_lower_bound",
        METRIC_RECALL: "recall_lower_bound",
        METRIC_F1: "f1_lower_bound",
        METRIC_F2: "f2_lower_bound",
        METRIC_ACCURACY: METRIC_ACCURACY,
        METRIC_PR_AUC: METRIC_PR_AUC,
    }
    metric_key = conservative_aliases.get(key)
    if metric_key is None:
        raise ValueError(f"Unsupported metric: {metric}")
    return float(metrics[metric_key])


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
    minimum_predicted_positive_support: int,
    minimum_true_positive_support: int,
    conservative_selection: bool,
) -> tuple[list[dict], str]:
    support_eligible = [
        row
        for row in grid
        if row["support"]["predicted_positive"] >= minimum_predicted_positive_support
        and row["confusion_matrix"]["tp"] >= minimum_true_positive_support
    ]
    eligible = [
        row
        for row in support_eligible
        if row["precision"] >= precision_floor and row["recall"] >= recall_floor
    ]

    precision_key = "precision_lower_bound" if conservative_selection else "precision"
    recall_key = "recall_lower_bound" if conservative_selection else "recall"
    f2_key = "f2_lower_bound" if conservative_selection else "f2"

    if primary_metric == METRIC_F2:
        if eligible:
            return sorted(
                eligible,
                key=lambda row: (
                    row[recall_key],
                    row[f2_key],
                    row[precision_key],
                    -row["threshold"],
                ),
                reverse=True,
            ), "floors"
        if support_eligible:
            return sorted(
                support_eligible,
                key=lambda row: (
                    row[f2_key],
                    row[recall_key],
                    row[precision_key],
                    -row["threshold"],
                ),
                reverse=True,
            ), "support_only"
        return sorted(
            grid,
            key=lambda row: (
                row[f2_key],
                row[recall_key],
                row[precision_key],
                -row["threshold"],
            ),
            reverse=True,
        ), "full_grid"

    if eligible:
        candidates = eligible
        stage = "floors"
    elif support_eligible:
        candidates = support_eligible
        stage = "support_only"
    else:
        candidates = grid
        stage = "full_grid"
    return sorted(
        candidates,
        key=lambda row: (
            selection_metric_value(row, primary_metric, conservative=conservative_selection),
            row[recall_key],
            row[precision_key],
            -row["threshold"],
        ),
        reverse=True,
    ), stage


def choose_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    precision_floor: float = 0.0,
    recall_floor: float = 0.0,
    minimum_predicted_positive_support: int = 0,
    minimum_true_positive_support: int = 0,
    primary_metric: str = "f1",
    grid_size: int = 181,
    conservative_selection: bool = False,
) -> tuple[float, dict]:
    """
    Choose a validation threshold and return diagnostics.

    For `primary_metric="f2"`, this becomes a recall-first policy:
    maximize recall under the floors, then tie-break on F2. If the floors
    are unattainable, fall back to the best unconstrained F2 threshold.
    """

    metric_name = (primary_metric or METRIC_F1).lower()
    grid = threshold_grid(y_true=y_true, y_prob=y_prob, grid_size=grid_size)
    min_predicted_positive_support = max(int(minimum_predicted_positive_support), 0)
    min_true_positive_support = max(int(minimum_true_positive_support), 0)
    ranked, selection_stage = _ranked_thresholds(
        grid=grid,
        primary_metric=metric_name,
        precision_floor=precision_floor,
        recall_floor=recall_floor,
        minimum_predicted_positive_support=min_predicted_positive_support,
        minimum_true_positive_support=min_true_positive_support,
        conservative_selection=conservative_selection,
    )
    selected = (
        ranked[0]
        if ranked
        else evaluate_threshold_metrics(y_true, y_prob, threshold=DEFAULT_DECISION_THRESHOLD)
    )
    support_eligible_count = sum(
        1
        for row in grid
        if row["support"]["predicted_positive"] >= min_predicted_positive_support
        and row["confusion_matrix"]["tp"] >= min_true_positive_support
    )
    floor_eligible_count = sum(
        1
        for row in grid
        if row["support"]["predicted_positive"] >= min_predicted_positive_support
        and row["confusion_matrix"]["tp"] >= min_true_positive_support
        and row["precision"] >= precision_floor
        and row["recall"] >= recall_floor
    )
    diagnostics = {
        "primary_metric": metric_name,
        "conservative_selection": bool(conservative_selection),
        "precision_floor": float(precision_floor),
        "recall_floor": float(recall_floor),
        "minimum_predicted_positive_support": min_predicted_positive_support,
        "minimum_true_positive_support": min_true_positive_support,
        "selection_stage": selection_stage,
        "eligible_threshold_count": floor_eligible_count,
        "support_eligible_threshold_count": support_eligible_count,
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
