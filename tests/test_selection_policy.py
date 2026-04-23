from __future__ import annotations

from pathlib import Path

import numpy as np

from chart_patterns.optuna.search import _resolve_storage_path, _validation_objective_score
from chart_patterns.project_utils import model_selection_from_cfg
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
        allow_low_support_fallback=True,
    )

    assert champion is not None
    assert champion["model"] == "model_b"


def test_champion_selection_requires_supported_validation_fold():
    model_results = [
        {
            "model": "model_a",
            "val_metrics": {"f1": 0.95, "f2": 0.95, "support": {"positive": 2}},
        },
        {
            "model": "model_b",
            "val_metrics": {"f1": 0.70, "f2": 0.82, "support": {"positive": 3}},
        },
    ]

    champion = _select_pattern_champion(
        model_results,
        primary_metric="f1",
        min_val_support=5,
        conservative_selection=False,
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


def test_optuna_validation_objective_uses_per_pattern_selection_config(base_cfg):
    base_cfg["model_selection"]["primary_selection_metric"] = "f1"
    base_cfg["model_selection"]["per_pattern"] = {
        "double_bottom": {
            "primary_selection_metric": "f2",
            "use_conservative_selection_scores": True,
            "minimum_predicted_positive_support": 4,
            "minimum_true_positive_support": 3,
        }
    }
    y_val = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    val_prob = np.array([0.95, 0.85, 0.75, 0.15, 0.65, 0.55, 0.20, 0.10], dtype=float)

    base_score = _validation_objective_score("logreg", y_val, val_prob, base_cfg, pattern_name="head_shoulders")
    pattern_score = _validation_objective_score("logreg", y_val, val_prob, base_cfg, pattern_name="double_bottom")

    assert pattern_score != base_score


def test_optuna_storage_path_supports_run_name_placeholder():
    path = _resolve_storage_path("outputs/optuna/{run_name}/studies.db", run_name="first_run_check_high_coverage")

    assert path == Path("outputs/optuna/first_run_check_high_coverage/studies.db")


def test_model_selection_from_cfg_merges_pattern_overrides(base_cfg):
    base_cfg["model_selection"]["precision_floor"] = 0.5
    base_cfg["model_selection"]["per_pattern"] = {
        "double_top": {
            "precision_floor": 0.7,
            "minimum_true_positive_support": 9,
        }
    }

    merged = model_selection_from_cfg(base_cfg, pattern="double_top")

    assert merged["precision_floor"] == 0.7
    assert merged["minimum_true_positive_support"] == 9
    assert merged["primary_selection_metric"] == base_cfg["model_selection"]["primary_selection_metric"]
