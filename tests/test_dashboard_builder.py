from __future__ import annotations

import pandas as pd

from scripts.build_results_dashboard import (
    _build_presentation_payload,
    _payload_hero,
    _pattern_support_from_split,
    _presentation_config,
)


def test_pattern_support_uses_configured_thresholds():
    split_df = pd.DataFrame(
        {
            "split": ["train"] * 30 + ["val"] * 8 + ["test"] * 9,
            "label": [1] * 25 + [0] * 5 + [1] * 6 + [0] * 2 + [1] * 7 + [0] * 2,
        }
    )
    presentation_cfg = _presentation_config(
        {
            "dashboard": {
                "presentation": {
                    "minimum_validation_positive_support": 5,
                    "minimum_test_positive_support": 5,
                }
            }
        }
    )

    support = _pattern_support_from_split(split_df, presentation_cfg)

    assert support["train_positive_support"] == 25
    assert support["val_positive_support"] == 6
    assert support["test_positive_support"] == 7
    assert support["presentation_eligible"] is True


def test_presentation_payload_hides_low_support_patterns():
    presentation_cfg = _presentation_config(
        {
            "dashboard": {
                "presentation": {
                    "minimum_validation_positive_support": 5,
                    "minimum_test_positive_support": 5,
                    "minimum_visible_patterns": 3,
                    "minimum_champion_f1": 0.25,
                }
            }
        }
    )
    summary_df = pd.DataFrame(
        [
            {
                "pattern": "head_shoulders",
                "train_positive_support": 120,
                "val_positive_support": 7,
                "test_positive_support": 6,
                "presentation_eligible": True,
            },
            {
                "pattern": "double_bottom",
                "train_positive_support": 40,
                "val_positive_support": 3,
                "test_positive_support": 4,
                "presentation_eligible": False,
            },
        ]
    )
    champion_rows = [
        {
            "pattern": "head_shoulders",
            "test_f1": 0.62,
            "test_pr_auc": 0.71,
            "presentation_eligible": True,
        },
        {
            "pattern": "double_bottom",
            "test_f1": 0.0,
            "test_pr_auc": 0.19,
            "presentation_eligible": False,
        },
    ]

    presentation = _build_presentation_payload(summary_df, champion_rows, presentation_cfg)

    assert presentation["visible_patterns"] == ["head_shoulders"]
    assert presentation["hidden_patterns"] == ["double_bottom"]
    assert presentation["presentation_eligible"] is False


def test_presentation_payload_accepts_supported_subset_for_snapshot():
    presentation_cfg = _presentation_config(
        {
            "dashboard": {
                "presentation": {
                    "minimum_validation_positive_support": 5,
                    "minimum_test_positive_support": 5,
                    "minimum_visible_patterns": 3,
                    "minimum_champion_f1": 0.25,
                }
            }
        }
    )
    summary_df = pd.DataFrame(
        [
            {
                "pattern": "double_bottom",
                "train_positive_support": 77,
                "val_positive_support": 6,
                "test_positive_support": 19,
                "presentation_eligible": True,
            },
            {
                "pattern": "double_top",
                "train_positive_support": 80,
                "val_positive_support": 8,
                "test_positive_support": 22,
                "presentation_eligible": True,
            },
            {
                "pattern": "inverse_head_shoulders",
                "train_positive_support": 41,
                "val_positive_support": 20,
                "test_positive_support": 19,
                "presentation_eligible": True,
            },
            {
                "pattern": "head_shoulders",
                "train_positive_support": 8,
                "val_positive_support": 2,
                "test_positive_support": 2,
                "presentation_eligible": False,
            },
        ]
    )
    champion_rows = [
        {"pattern": "double_bottom", "test_f1": 1.0, "test_pr_auc": 1.0, "presentation_eligible": True},
        {"pattern": "double_top", "test_f1": 0.95, "test_pr_auc": 0.97, "presentation_eligible": True},
        {
            "pattern": "inverse_head_shoulders",
            "test_f1": 0.94,
            "test_pr_auc": 0.98,
            "presentation_eligible": True,
        },
        {"pattern": "head_shoulders", "test_f1": 0.67, "test_pr_auc": 0.70, "presentation_eligible": False},
    ]

    presentation = _build_presentation_payload(summary_df, champion_rows, presentation_cfg)

    assert presentation["presentation_eligible"] is True
    assert presentation["visible_patterns"] == [
        "double_bottom",
        "double_top",
        "inverse_head_shoulders",
    ]
    assert presentation["hidden_patterns"] == ["head_shoulders"]


def test_payload_hero_ignores_all_negative_backtests():
    summary_df = pd.DataFrame(
        [
            {
                "pattern": "head_shoulders",
                "model": "logreg",
                "f1": 0.7,
                "pr_auc": 0.67,
                "presentation_eligible": True,
            },
            {
                "pattern": "double_bottom",
                "model": "logreg",
                "f1": 0.0,
                "pr_auc": 0.19,
                "presentation_eligible": False,
            },
        ]
    )
    pair_backtest_df = pd.DataFrame(
        [
            {
                "pattern": "head_shoulders",
                "model": "logreg",
                "trades": 10,
                "total_pnl": -10.0,
                "profit_factor": 0.8,
                "presentation_eligible": True,
            },
            {
                "pattern": "double_bottom",
                "model": "logreg",
                "trades": 0,
                "total_pnl": 0.0,
                "profit_factor": 0.0,
                "presentation_eligible": False,
            },
        ]
    )
    champion_rows = [
        {"pattern": "head_shoulders", "total_pnl": -10.0, "presentation_eligible": True},
    ]

    hero = _payload_hero(summary_df, pair_backtest_df, champion_rows, macro={})

    assert hero["best_test_f1_pair"]["pattern"] == "head_shoulders"
    assert hero["best_backtest_pair"] == {}
