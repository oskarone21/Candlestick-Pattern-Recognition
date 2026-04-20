from __future__ import annotations

import numpy as np

from candlestick.eval.classification import (
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


def test_conservative_selection_metric_penalizes_tiny_perfect_fold():
    y_true = np.array([1, 1, 0, 0], dtype=int)
    y_prob = np.array([0.95, 0.85, 0.20, 0.10], dtype=float)

    metrics = evaluate_threshold_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["precision_lower_bound"] < 1.0
    assert metrics["recall_lower_bound"] < 1.0
    assert selection_metric_value(metrics, "f1", conservative=True) < 1.0


def test_choose_threshold_requires_minimum_prediction_support_for_precision_mode():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    y_prob = np.array([0.95, 0.70, 0.62, 0.55, 0.60, 0.30, 0.20, 0.10], dtype=float)

    threshold, diag = choose_threshold(
        y_true=y_true,
        y_prob=y_prob,
        precision_floor=0.75,
        recall_floor=0.25,
        minimum_predicted_positive_support=4,
        minimum_true_positive_support=3,
        primary_metric="precision",
        grid_size=19,
        conservative_selection=True,
    )
    metrics = evaluate_threshold_metrics(y_true, y_prob, threshold)

    assert np.isclose(threshold, 0.35)
    assert metrics["precision"] == 0.8
    assert metrics["recall"] == 1.0
    assert metrics["support"]["predicted_positive"] >= 4
    assert metrics["confusion_matrix"]["tp"] >= 3
    assert diag["selection_stage"] == "floors"


def test_choose_threshold_prefers_lower_threshold_when_scores_tie():
    y_true = np.array([1, 1, 0, 0], dtype=int)
    y_prob = np.array([0.90, 0.80, 0.04, 0.03], dtype=float)

    threshold, _ = choose_threshold(
        y_true=y_true,
        y_prob=y_prob,
        primary_metric="precision",
        grid_size=19,
    )

    assert threshold == 0.05
