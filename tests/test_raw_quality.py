from __future__ import annotations

import copy

import pandas as pd

from candlestick.data.raw_quality import build_raw_quality_report


def test_raw_quality_report_counts_duplicates_and_timezone_fit(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["data_source"]["timestamp"]["timezone_in_data"] = "America/Denver"
    cfg["data_source"]["timestamp"]["convert_to_timezone"] = "America/New_York"
    cfg["data_source"]["session"]["start"] = "09:30"
    cfg["data_source"]["session"]["end"] = "16:00"

    ts = pd.date_range("2024-01-02 07:30", periods=390, freq="1min")
    day1 = pd.DataFrame(
        {
            "symbol": "SPY",
            "ts_event": ts.astype(str),
            "open": [100.0] * 390,
            "high": [101.0] * 390,
            "low": [99.0] * 390,
            "close": [100.5] * 390,
            "volume": [1000] * 390,
        }
    )
    day2 = day1.copy()
    day2["ts_event"] = pd.date_range("2024-01-03 07:30", periods=390, freq="1min").astype(str)
    duplicated = day2.iloc[:20].copy()

    frame = pd.concat([day1, day2, duplicated], ignore_index=True)
    report = build_raw_quality_report(cfg, frame)

    assert report["duplicate_symbol_timestamp"] == 20
    assert report["duplicate_symbol_timestamp_ohlcv_exact"] == 20
    assert report["conflicting_duplicate_groups_symbol_timestamp"] == 0
    assert report["best_fit_timezone_by_full_session_days"] == "America/Denver"
