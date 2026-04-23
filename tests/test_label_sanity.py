from __future__ import annotations

import numpy as np
import pandas as pd

from chart_patterns.eval.label_sanity import (
    label_sanity_issues,
    summarize_pattern_dataset,
    summarize_pattern_events,
    summarize_processed_prices,
    summarize_split_support,
)


def test_summarize_processed_prices_reports_signature():
    ts = pd.date_range("2024-01-02 09:30", periods=4, freq="15min", tz="America/New_York")
    prices = pd.DataFrame(
        {
            "symbol": ["SPY"] * 4,
            "ts_event": ts,
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.5, 2.5, 3.5, 4.5],
            "low": [0.5, 1.5, 2.5, 3.5],
            "close": [1.2, 2.2, 3.2, 4.2],
            "volume": [100, 110, 120, 130],
        }
    )

    summary = summarize_processed_prices(prices)

    assert summary["available"] is True
    assert summary["row_count"] == 4
    assert summary["trading_day_count"] == 1
    assert summary["symbol_count"] == 1


def test_label_sanity_helpers_track_events_dataset_and_splits():
    events = pd.DataFrame(
        {
            "pattern": ["head_shoulders", "head_shoulders", "head_shoulders"],
            "label": [1, 0, 0],
            "reason": ["confirmed_breakout", "failed_breakout", "partial"],
            "anchor_idx": [10, 20, 30],
        }
    )
    X = np.ones((3, 8, 5), dtype=np.float32)
    y = np.array([1, 0, 0], dtype=np.int64)
    meta = pd.DataFrame({"window_end_ts": pd.date_range("2024-01-02", periods=3, freq="15min").astype(str)})
    split_data = {
        "X_train": X[:1],
        "y_train": y[:1],
        "X_val": X[1:2],
        "y_val": y[1:2],
        "X_test": X[2:],
        "y_test": y[2:],
        "idx_train": np.array([0]),
        "idx_val": np.array([1]),
        "idx_test": np.array([2]),
    }
    meta_split = meta.copy()
    meta_split["split"] = ["train", "val", "test"]

    event_summary = summarize_pattern_events("head_shoulders", events)
    dataset_summary = summarize_pattern_dataset("head_shoulders", X, y, meta)
    split_summary = summarize_split_support(split_data, meta_split)
    issues = label_sanity_issues(event_summary, dataset_summary, split_summary)

    assert event_summary["positive_events"] == 1
    assert dataset_summary["positive_labels"] == 1
    assert split_summary["folds"]["train"]["positive"] == 1
    assert "zero_positive_support_val" in issues
    assert "zero_positive_support_test" in issues
