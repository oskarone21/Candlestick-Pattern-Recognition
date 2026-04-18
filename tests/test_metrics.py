from __future__ import annotations

import numpy as np

from candlestick.eval.classification import (
    attach_confidence_intervals,
    choose_threshold,
    evaluate_threshold_metrics,
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
