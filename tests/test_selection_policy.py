from __future__ import annotations

import numpy as np

from candlestick.optuna.search import _validation_objective_score
from candlestick.project_utils import model_selection_from_cfg
from scripts.run_experiment_suite import _select_pattern_champion


def test_champion_selection_respects_primary_metric():
    model_results = [
        {
            "model": "model_a",
            "val_metrics": {"f1": 0.70, "f2": 0.60, "support": {"positive": 10}},
        },
        {
            "model": "model_b",
            "val_metrics": {"f1": 0.68, "f2": 0.82, "support": {"positive": 10}},
        },
    ]

    champion = _select_pattern_champion(
        model_results,
        primary_metric="f2",
        min_val_support=1,
        conservative_selection=False,
        allow_low_support_fallback=False,
    )

    assert champion is not None
    assert champion["model"] == "model_b"


def test_champion_selection_can_penalize_tiny_perfect_validation_scores():
    model_results = [
        {
            "model": "model_a",
            "val_metrics": {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "f2": 1.0,
                "precision_lower_bound": 0.34,
                "recall_lower_bound": 0.34,
                "f1_lower_bound": 0.34,
                "f2_lower_bound": 0.34,
                "support": {"positive": 2},
            },
        },
        {
            "model": "model_b",
            "val_metrics": {
                "precision": 0.9,
                "recall": 0.85,
                "f1": 0.875,
                "f2": 0.86,
                "precision_lower_bound": 0.72,
                "recall_lower_bound": 0.68,
                "f1_lower_bound": 0.70,
                "f2_lower_bound": 0.68,
                "support": {"positive": 18},
            },
        },
    ]

    champion = _select_pattern_champion(
        model_results,
        primary_metric="f1",
        min_val_support=1,
        conservative_selection=True,
        allow_low_support_fallback=False,
    )

    assert champion is not None
    assert champion["model"] == "model_b"


def test_champion_selection_can_skip_patterns_without_validation_support():
    model_results = [
        {
            "model": "model_a",
            "val_metrics": {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "f2": 1.0,
                "precision_lower_bound": 0.34,
                "recall_lower_bound": 0.34,
                "f1_lower_bound": 0.34,
                "f2_lower_bound": 0.34,
                "support": {"positive": 2},
            },
        }
    ]

    champion = _select_pattern_champion(
        model_results,
        primary_metric="precision",
        min_val_support=8,
        conservative_selection=True,
        allow_low_support_fallback=False,
    )

    assert champion is None


def test_optuna_validation_objective_uses_primary_metric(base_cfg):
    base_cfg["model_selection"]["primary_selection_metric"] = "f2"
    base_cfg["model_selection"]["precision_floor"] = 0.5
    y_val = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    val_prob = np.array([0.95, 0.70, 0.62, 0.55, 0.60, 0.30, 0.20, 0.10], dtype=float)

    score = _validation_objective_score("logreg", y_val, val_prob, base_cfg)

    assert score > 0.0


def test_model_selection_can_merge_pattern_specific_overrides(base_cfg):
    base_cfg["model_selection"]["primary_selection_metric"] = "precision"
    base_cfg["model_selection"]["precision_floor"] = 0.7
    base_cfg["model_selection"]["per_pattern"] = {
        "double_bottom": {
            "primary_selection_metric": "f2",
            "recall_floor": 0.35,
            "minimum_predicted_positive_support": 14,
        }
    }

    resolved = model_selection_from_cfg(base_cfg, pattern="double_bottom")

    assert resolved["primary_selection_metric"] == "f2"
    assert resolved["precision_floor"] == 0.7
    assert resolved["recall_floor"] == 0.35
    assert resolved["minimum_predicted_positive_support"] == 14
