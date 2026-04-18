from __future__ import annotations

import pandas as pd

from candlestick.data.timezone_audit import normalize_and_audit_timezone


def test_timezone_normalization_captures_dst_offsets(base_cfg):
    df = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY", "SPY"],
            "ts_event": [
                "2024-03-08 14:30:00",
                "2024-03-08 14:31:00",
                "2024-03-11 13:30:00",
                "2024-03-11 13:31:00",
            ],
            "open": [1, 1, 1, 1],
            "high": [1, 1, 1, 1],
            "low": [1, 1, 1, 1],
            "close": [1, 1, 1, 1],
            "volume": [1, 1, 1, 1],
        }
    )

    result = normalize_and_audit_timezone(
        df,
        timezone_in_data="UTC",
        working_timezone="America/New_York",
        fail_on_naive=True,
    )

    assert str(result.frame["ts_event"].dtype).startswith("datetime64[ns, America/New_York]")
    offsets = result.report["offset_distribution_minutes"]
    assert "-300" in offsets
    assert "-240" in offsets
    assert result.report["symbols_with_non_monotonic_time"] == 0


def test_timezone_normalization_from_denver_aligns_to_ny_session():
    df = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "ts_event": ["2024-01-02 07:30:00", "2024-01-02 13:59:00"],
            "open": [1, 1],
            "high": [1, 1],
            "low": [1, 1],
            "close": [1, 1],
            "volume": [1, 1],
        }
    )

    result = normalize_and_audit_timezone(
        df,
        timezone_in_data="America/Denver",
        working_timezone="America/New_York",
        fail_on_naive=True,
    )

    converted = result.frame["ts_event"].dt.strftime("%H:%M").tolist()
    assert converted == ["09:30", "15:59"]
