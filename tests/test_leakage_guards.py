from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from candlestick.labeling.extrema import gaussian_smooth_causal
from scripts.run_experiment_suite import _downsample_train_negatives, _prepare_split


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

