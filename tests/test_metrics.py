from __future__ import annotations

import numpy as np

from chart_patterns.eval.classification import (
    attach_confidence_intervals,
    choose_threshold,
    evaluate_threshold_metrics,
    selection_metric_value,
)


def test_metrics_bootstrap_reproducible():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=300)
    y_prob = rng.uniform(0.0, 1.0, size=300)

    threshold, _ = choose_threshold(
        y_true,
        y_prob,
        precision_floor=0.2,
        recall_floor=0.2,
        primary_metric="f2",
    )
    m = evaluate_threshold_metrics(y_true, y_prob, threshold)

    r1 = attach_confidence_intervals(m, y_true, y_prob, n_bootstrap=120, seed=7)
    r2 = attach_confidence_intervals(m, y_true, y_prob, n_bootstrap=120, seed=7)

    assert r1["confidence_intervals"]["f1"] == r2["confidence_intervals"]["f1"]
    assert r1["confidence_intervals"]["f2"] == r2["confidence_intervals"]["f2"]
    assert r1["confidence_intervals"]["precision"] == r2["confidence_intervals"]["precision"]


def test_choose_threshold_recall_first_under_precision_floor():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    y_prob = np.array([0.95, 0.70, 0.62, 0.55, 0.60, 0.30, 0.20, 0.10], dtype=float)

    threshold, diag = choose_threshold(
        y_true=y_true,
        y_prob=y_prob,
        precision_floor=0.5,
        recall_floor=0.0,
        primary_metric="f2",
        grid_size=19,
    )
    metrics = evaluate_threshold_metrics(y_true, y_prob, threshold)

    assert metrics["precision"] >= 0.5
    assert metrics["recall"] == 1.0
    assert diag["selected_metrics"]["f2"] == metrics["f2"]


def test_evaluate_threshold_metrics_exposes_support_and_lower_bounds():
    y_true = np.array([1, 1, 0, 0], dtype=int)
    y_prob = np.array([0.95, 0.80, 0.70, 0.10], dtype=float)

    metrics = evaluate_threshold_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["support"]["positive"] == 2
    assert metrics["support"]["negative"] == 2
    assert metrics["support"]["predicted_positive"] == 3
    assert metrics["support"]["predicted_negative"] == 1
    assert 0.0 <= metrics["precision_lower_bound"] <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall_lower_bound"] <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_lower_bound"] <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["f2_lower_bound"] <= metrics["f2"] <= 1.0


def test_selection_metric_value_uses_lower_bounds_when_conservative():
    metrics = {
        "precision": 0.9,
        "precision_lower_bound": 0.4,
        "recall": 0.8,
        "recall_lower_bound": 0.3,
        "f1": 0.85,
        "f1_lower_bound": 0.35,
        "f2": 0.82,
        "f2_lower_bound": 0.33,
        "accuracy": 0.7,
        "pr_auc": 0.6,
    }

    assert selection_metric_value(metrics, "f1", conservative=False) == 0.85
    assert selection_metric_value(metrics, "f1", conservative=True) == 0.35
    assert selection_metric_value(metrics, "precision", conservative=True) == 0.4
    assert selection_metric_value(metrics, "accuracy", conservative=True) == 0.7


def test_choose_threshold_respects_support_gates():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    y_prob = np.array([0.95, 0.85, 0.75, 0.15, 0.65, 0.55, 0.20, 0.10], dtype=float)

    threshold, diag = choose_threshold(
        y_true=y_true,
        y_prob=y_prob,
        primary_metric="precision",
        minimum_predicted_positive_support=4,
        minimum_true_positive_support=3,
        conservative_selection=True,
        grid_size=19,
    )

    assert threshold == 0.6
    assert diag["selected_metrics"]["support"]["predicted_positive"] >= 4
    assert diag["selected_metrics"]["confusion_matrix"]["tp"] >= 3
    assert diag["selection_stage"] == "floors"


def test_choose_threshold_reports_support_only_stage_when_floors_unmet():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    y_prob = np.array([0.95, 0.85, 0.75, 0.15, 0.65, 0.55, 0.20, 0.10], dtype=float)

    _, diag = choose_threshold(
        y_true=y_true,
        y_prob=y_prob,
        primary_metric="f1",
        precision_floor=0.95,
        recall_floor=0.95,
        minimum_predicted_positive_support=4,
        minimum_true_positive_support=3,
        conservative_selection=True,
        grid_size=19,
    )

    assert diag["selection_stage"] == "support_only"
    assert diag["support_eligible_threshold_count"] > 0
    assert diag["eligible_threshold_count"] == 0
