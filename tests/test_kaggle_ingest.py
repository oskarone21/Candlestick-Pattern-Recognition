from __future__ import annotations

import pandas as pd
import pytest

from candlestick.data.kaggle_ingest import IngestError, normalize_intraday_schema


def test_normalize_intraday_schema_maps_columns(base_cfg):
    df = pd.DataFrame(
        {
            "date": ["2024-01-02 14:30:00+00:00", "2024-01-02 14:31:00+00:00"],
            "open": [100.0, 100.2],
            "high": [100.4, 100.5],
            "low": [99.9, 100.1],
            "close": [100.3, 100.4],
            "volume": [1200, 1300],
        }
    )

    out = normalize_intraday_schema(df, base_cfg)
    assert list(out.columns[:7]) == ["symbol", "ts_event", "open", "high", "low", "close", "volume"]
    assert out["symbol"].nunique() == 1
    assert out["symbol"].iloc[0] == "SPY"


def test_normalize_intraday_schema_raises_on_missing_columns(base_cfg):
    df = pd.DataFrame({"date": ["2024-01-02"], "open": [1.0], "high": [1.1], "low": [0.9]})
    with pytest.raises(IngestError):
        normalize_intraday_schema(df, base_cfg)
