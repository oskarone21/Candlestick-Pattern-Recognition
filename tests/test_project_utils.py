from __future__ import annotations

from pathlib import Path

import pandas as pd

from chart_patterns.project_utils import load_processed_prices, timeframe_to_minutes


def test_timeframe_to_minutes_supports_minute_and_hour_aliases():
    assert timeframe_to_minutes("15min") == 15
    assert timeframe_to_minutes("1m") == 1
    assert timeframe_to_minutes("2h") == 120


def test_load_processed_prices_filters_instrument_and_converts_timezone(tmp_path, base_cfg):
    prices_path = tmp_path / "spy_15m.csv"
    frame = pd.DataFrame(
        {
            "symbol": ["SPY", "QQQ"],
            "ts_event": ["2024-01-02T14:30:00Z", "2024-01-02T14:45:00Z"],
            "open": [100.0, 200.0],
            "high": [101.0, 201.0],
            "low": [99.0, 199.0],
            "close": [100.5, 200.5],
            "volume": [1000, 1200],
        }
    )
    frame.to_csv(prices_path, index=False)

    cfg = base_cfg
    cfg["paths"]["processed_15m_path"] = str(prices_path)
    cfg["data_source"]["instrument"] = "SPY"

    loaded = load_processed_prices(cfg)

    assert list(loaded["symbol"]) == ["SPY"]
    assert str(loaded.iloc[0]["ts_event"].tz) == "America/New_York"
    assert loaded.iloc[0]["ts_event"].hour == 9
    assert loaded.iloc[0]["ts_event"].minute == 30
