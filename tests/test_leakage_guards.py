from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from chart_patterns.labeling.extrema import gaussian_smooth_causal
from scripts.run_experiment_suite import (
    _apply_train_augmentation,
    _augment_train_positives,
    _downsample_train_negatives,
    _prepare_split,
    run_experiment_suite,
)


def test_downsample_train_negatives_affects_train_only_distribution():
    X = np.random.default_rng(42).normal(size=(20, 8, 5)).astype(np.float32)
    y = np.array([1, 1, 1, 1] + [0] * 16, dtype=np.int64)

    X_out, y_out = _downsample_train_negatives(X, y, target_ratio="1:2", seed=42)

    assert X_out.shape[0] == len(y_out)
    assert int((y_out == 1).sum()) == 4
    assert int((y_out == 0).sum()) == 8


def test_prepare_split_enforces_min_embargo_when_config_is_zero(base_cfg):
    cfg = base_cfg
    cfg["evaluation"]["train_ratio"] = 0.6
    cfg["evaluation"]["val_ratio"] = 0.2
    cfg["evaluation"]["test_ratio"] = 0.2
    cfg["evaluation"]["embargo_bars"] = 0
    cfg["evaluation"]["enforce_min_embargo"] = True
    cfg["evaluation"]["allow_zero_embargo_fallback"] = False

    n = 700
    lookback = int(cfg["windowing"]["lookback_bars"])
    horizon = int(cfg["labeling"]["confirmation"]["confirm_break_within_bars"])
    expected_gap = lookback + horizon

    X = np.random.default_rng(7).normal(size=(n, lookback, 5)).astype(np.float32)
    y = np.random.default_rng(8).integers(0, 2, size=n, endpoint=False).astype(np.int64)
    meta = pd.DataFrame({"window_end_ts": pd.date_range("2024-01-02", periods=n, freq="15min").astype(str)})

    split_data, _ = _prepare_split(X, y, meta, cfg)

    train_max = int(split_data["idx_train"].max())
    val_min = int(split_data["idx_val"].min())
    val_max = int(split_data["idx_val"].max())
    test_min = int(split_data["idx_test"].min())

    assert (val_min - train_max) >= expected_gap
    assert (test_min - val_max) >= expected_gap


def test_prepare_split_raises_when_embargo_makes_folds_empty(base_cfg):
    cfg = base_cfg
    cfg["evaluation"]["train_ratio"] = 0.6
    cfg["evaluation"]["val_ratio"] = 0.2
    cfg["evaluation"]["test_ratio"] = 0.2
    cfg["evaluation"]["embargo_bars"] = 200
    cfg["evaluation"]["enforce_min_embargo"] = False
    cfg["evaluation"]["allow_zero_embargo_fallback"] = False

    n = 120
    lookback = int(cfg["windowing"]["lookback_bars"])
    X = np.random.default_rng(9).normal(size=(n, lookback, 5)).astype(np.float32)
    y = np.random.default_rng(10).integers(0, 2, size=n, endpoint=False).astype(np.int64)
    meta = pd.DataFrame({"window_end_ts": pd.date_range("2024-01-02", periods=n, freq="15min").astype(str)})

    with pytest.raises(ValueError):
        _prepare_split(X, y, meta, cfg)


def test_causal_smoothing_not_affected_by_future_shock():
    base = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    shocked = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 100.0])

    sm_base = gaussian_smooth_causal(base, bandwidth=2.0)
    sm_shocked = gaussian_smooth_causal(shocked, bandwidth=2.0)

    # Earlier points must remain unchanged when only the final future bar changes.
    assert np.allclose(sm_base[:-1], sm_shocked[:-1], atol=1.0e-9)


def test_positive_augmentation_is_deterministic_and_label_preserving(base_cfg):
    cfg = base_cfg
    cfg["training"]["augmentation"]["enabled"] = True
    cfg["training"]["augmentation"]["n_copies_per_positive"] = 2

    X_train = np.array(
        [
            [[100.0, 101.0, 99.0, 100.5, 1000.0], [101.0, 102.0, 100.0, 101.5, 1100.0]],
            [[102.0, 103.0, 101.0, 102.5, 1200.0], [103.0, 104.0, 102.0, 103.5, 1300.0]],
            [[104.0, 105.0, 103.0, 104.5, 1400.0], [105.0, 106.0, 104.0, 105.5, 1500.0]],
        ],
        dtype=np.float32,
    )
    y_train = np.array([1, 0, 1], dtype=np.int64)

    X_aug_a, y_aug_a = _augment_train_positives(X_train, y_train, cfg=cfg, seed=42)
    X_aug_b, y_aug_b = _augment_train_positives(X_train, y_train, cfg=cfg, seed=42)

    assert np.array_equal(y_aug_a, y_aug_b)
    assert np.allclose(X_aug_a, X_aug_b)
    assert X_aug_a.shape[0] == X_train.shape[0] + 4
    assert y_aug_a.shape[0] == X_aug_a.shape[0]
    assert int((y_aug_a == 1).sum()) == 6
    assert int((y_aug_a == 0).sum()) == 1


def test_train_augmentation_only_changes_training_fold(base_cfg):
    cfg = base_cfg
    cfg["training"]["augmentation"]["enabled"] = True
    cfg["training"]["augmentation"]["n_copies_per_positive"] = 1

    split_data = {
        "X_train": np.ones((2, 3, 5), dtype=np.float32),
        "y_train": np.array([1, 0], dtype=np.int64),
        "X_val": np.full((1, 3, 5), 2.0, dtype=np.float32),
        "y_val": np.array([1], dtype=np.int64),
        "X_test": np.full((1, 3, 5), 3.0, dtype=np.float32),
        "y_test": np.array([0], dtype=np.int64),
        "idx_train": np.array([0, 1], dtype=np.int64),
        "idx_val": np.array([2], dtype=np.int64),
        "idx_test": np.array([3], dtype=np.int64),
    }

    augmented = _apply_train_augmentation(split_data, cfg=cfg, seed=7)

    assert augmented["X_train"].shape[0] == 3
    assert augmented["y_train"].shape[0] == 3
    assert np.array_equal(augmented["X_val"], split_data["X_val"])
    assert np.array_equal(augmented["y_val"], split_data["y_val"])
    assert np.array_equal(augmented["X_test"], split_data["X_test"])
    assert np.array_equal(augmented["y_test"], split_data["y_test"])


def test_non_smoke_run_writes_label_sanity_before_raising(tmp_path, base_cfg):
    cfg = base_cfg
    cfg["project"]["run_name"] = "label_sanity_failfast"
    cfg["model_selection"]["candidate_models"] = ["logreg"]
    cfg["labeling"]["allowed_patterns"] = ["head_shoulders"]
    cfg["optuna"]["enabled"] = False
    cfg["project"]["device"] = "cpu"
    cfg["paths"]["metrics_dir"] = str(tmp_path / "metrics")
    cfg["paths"]["checkpoints_dir"] = str(tmp_path / "checkpoints")
    cfg["paths"]["datasets_dir"] = str(tmp_path / "datasets")
    cfg["paths"]["processed_15m_path"] = str(tmp_path / "spy_15m.csv")

    ts = pd.date_range("2024-01-02 09:30", periods=240, freq="15min", tz="America/New_York")
    close = np.linspace(100.0, 120.0, num=len(ts))
    monotonic = pd.DataFrame(
        {
            "symbol": "SPY",
            "ts_event": ts.astype(str),
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.full(len(ts), 1000.0),
        }
    )
    monotonic.to_csv(cfg["paths"]["processed_15m_path"], index=False)

    with pytest.raises(ValueError, match="Label sanity check failed"):
        run_experiment_suite(cfg=cfg, smoke=False)

    label_sanity_path = tmp_path / "metrics" / "label_sanity_failfast" / "label_sanity.json"
    assert label_sanity_path.exists()
    payload = json.loads(label_sanity_path.read_text(encoding="utf-8"))
    warnings = payload["patterns"]["head_shoulders"]["warnings"]
    assert "zero_positive_events" in warnings
    assert "zero_positive_dataset_labels" in warnings
