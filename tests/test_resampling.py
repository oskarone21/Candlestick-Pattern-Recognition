from __future__ import annotations

import pandas as pd

from chart_patterns.data.clean_resample import clean_ohlcv, resample_ohlcv


def test_resample_1m_to_15m_aggregation():
    ts = pd.date_range("2024-01-02 09:30", periods=30, freq="1min", tz="America/New_York")
    close = [100 + i for i in range(30)]
    open_ = [close[0]] + close[:-1]
    high = [max(o, c) + 0.5 for o, c in zip(open_, close)]
    low = [min(o, c) - 0.5 for o, c in zip(open_, close)]
    vol = [100] * 30

    df = pd.DataFrame(
        {
            "symbol": "SPY",
            "ts_event": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )

    cleaned = clean_ohlcv(df).frame
    out = resample_ohlcv(
        cleaned,
        base_timeframe="1min",
        target_timeframe="15min",
        drop_incomplete_bars=True,
    )

    assert len(out) == 2
    first = out.iloc[0]
    assert first["open"] == open_[0]
    assert first["close"] == close[14]
    assert first["high"] == max(high[:15])
    assert first["low"] == min(low[:15])
    assert first["volume"] == 1500
