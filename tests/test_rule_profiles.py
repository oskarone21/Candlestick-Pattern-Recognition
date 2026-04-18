from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from candlestick.config import load_config
from candlestick.labeling.pattern_rules import detect_pattern_events


@pytest.mark.skipif(
    not Path("data/processed/spy_15m.csv").exists(),
    reason="Prepared SPY 15m dataset not available.",
)
def test_intraday_balanced_profile_yields_positives_for_all_patterns():
    cfg = load_config(
        "configs/config.yaml",
        ["configs/overrides/intraday_balanced.yaml"],
        [],
    )
    df = pd.read_csv("data/processed/spy_15m.csv")
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    df = df.dropna(subset=["ts_event"])
    df = df[df["symbol"].astype(str).str.upper() == "SPY"].sort_values("ts_event").reset_index(drop=True)

    for pattern in cfg["labeling"]["allowed_patterns"]:
        events = detect_pattern_events(df, pattern, cfg)
        assert not events.empty
        assert int((events["label"] == 1).sum()) > 0
