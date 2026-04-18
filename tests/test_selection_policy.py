from __future__ import annotations

import numpy as np

from candlestick.optuna.search import _validation_objective_score
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

    champion = _select_pattern_champion(model_results, primary_metric="f2", min_val_support=1)

    assert champion is not None
    assert champion["model"] == "model_b"


def test_optuna_validation_objective_uses_primary_metric(base_cfg):
    base_cfg["model_selection"]["primary_selection_metric"] = "f2"
    base_cfg["model_selection"]["precision_floor"] = 0.5
    y_val = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    val_prob = np.array([0.95, 0.70, 0.62, 0.55, 0.60, 0.30, 0.20, 0.10], dtype=float)

    score = _validation_objective_score("logreg", y_val, val_prob, base_cfg)

    assert score > 0.0
