from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def base_cfg() -> dict:
    path = REPO_ROOT / "configs" / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return copy.deepcopy(cfg)


@pytest.fixture
def tiny_price_df() -> pd.DataFrame:
    ts = pd.date_range("2024-01-02 09:30", periods=40, freq="1min", tz="America/New_York")
    close = [100 + (i * 0.1) for i in range(40)]
    open_ = [close[0]] + close[:-1]
    high = [max(o, c) + 0.2 for o, c in zip(open_, close)]
    low = [min(o, c) - 0.2 for o, c in zip(open_, close)]
    vol = [1000 + i for i in range(40)]

    return pd.DataFrame(
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
