from __future__ import annotations

from candlestick.config import load_config


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
    assert cfg["labeling"]["double_top"]["geometry"]["min_pullback_pct"] == 0.0005
    assert cfg["labeling"]["double_bottom"]["geometry"]["min_bounce_pct"] == 0.0005
