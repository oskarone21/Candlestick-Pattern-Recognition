from __future__ import annotations

from chart_patterns.config import load_config


def test_base_config_restores_broad_signal_research_defaults():
    cfg = load_config("configs/config.yaml")

    assert cfg["model_selection"]["candidate_models"] == ["hgb", "logreg", "lstm", "tcn", "transformer"]
    assert cfg["model_selection"]["primary_selection_metric"] == "f2"
    assert cfg["training"]["augmentation"]["enabled"] is False
    assert cfg["dashboard"]["primary_run_name"] == "dl_rerun_aug_windowminmax_stageAdata_20260419"
    assert cfg["dashboard"]["artifact_run_name"] == "rough_screen_all_models_2026_04_18_relaxed"
    assert cfg["dashboard"]["comparison_run_names"] == [
        "stageA_compare_all_no_optuna",
        "stageB_optuna_coarse",
        "rough_screen_all_models_2026_04_18_relaxed",
    ]
    assert cfg["dashboard"]["presentation"]["minimum_validation_positive_support"] == 5
    assert cfg["dashboard"]["presentation"]["minimum_test_positive_support"] == 5


def test_production_repro_override_matches_verified_profile():
    cfg = load_config(
        "configs/config.yaml",
        ["configs/overrides/production_repro.yaml"],
    )

    assert cfg["project"]["device"] == "cpu"
    assert cfg["project"]["seed"] == 42
    assert cfg["model"]["sequence_normalization_mode"] == "window_minmax"
    assert cfg["model_selection"]["candidate_models"] == ["tcn", "lstm"]
    assert cfg["model_selection"]["primary_selection_metric"] == "f1"
    assert cfg["training"]["augmentation"]["enabled"] is True
    assert cfg["training"]["epochs"] == 120
    assert cfg["training"]["early_stopping_patience"] == 15
    assert cfg["optuna"]["enabled"] is True
    assert cfg["optuna"]["n_trials"] == 6
    assert cfg["optuna"]["epochs_per_trial"] == 6
    assert cfg["dashboard"]["presentation"]["minimum_validation_positive_support"] == 5
    assert cfg["dashboard"]["presentation"]["minimum_test_positive_support"] == 5
    assert cfg["dashboard"]["presentation"]["minimum_visible_patterns"] == 3
    assert cfg["dashboard"]["presentation"]["minimum_champion_f1"] == 0.25
    assert cfg["labeling"]["double_top"]["geometry"]["min_pullback_pct"] == 0.0005
    assert cfg["labeling"]["double_bottom"]["geometry"]["min_bounce_pct"] == 0.0005


def test_first_run_repro_override_locks_first_run_runtime():
    cfg = load_config(
        "configs/config.yaml",
        ["configs/overrides/first_run_repro.yaml"],
    )

    assert cfg["project"]["run_name"] == "first_run_check"
    assert cfg["project"]["seed"] == 42
    assert cfg["project"]["device"] == "cpu"
    assert cfg["model_selection"]["candidate_models"] == ["hgb", "logreg", "lstm", "tcn", "transformer"]
    assert cfg["model_selection"]["primary_selection_metric"] == "f2"
    assert cfg["training"]["mixed_precision"] is False
    assert cfg["optuna"]["storage_path"] == "outputs/optuna/{run_name}/studies.db"


def test_legacy_first_run_repro_override_remains_supported():
    cfg = load_config(
        "configs/config.yaml",
        ["configs/overrides/broad_signal_repro.yaml"],
    )

    assert cfg["project"]["run_name"] == "first_run_check"
    assert cfg["project"]["device"] == "cpu"
