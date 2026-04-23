from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from chart_patterns.config import load_config
from chart_patterns.datasets.window_builder import build_pattern_dataset
from chart_patterns.labeling.pattern_rules import detect_pattern_events
from scripts.run_experiment_suite import _prepare_split


@pytest.mark.skipif(
    not Path("data/processed/spy_15m.csv").exists(),
    reason="Prepared SPY 15m dataset not available.",
)
def test_balanced_profile_yields_positives_for_all_patterns():
    cfg = load_config(
        "configs/config.yaml",
        ["configs/overrides/balanced.yaml"],
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

        X, y, meta = build_pattern_dataset(
            df,
            events,
            lookback_bars=int(cfg["windowing"].get("lookback_bars", 80)),
        )
        assert len(y) > 0
        assert int((y == 1).sum()) > 0

        split_data, _ = _prepare_split(X, y, meta, cfg)
        assert int((split_data["y_train"] == 1).sum()) > 0
        assert int((split_data["y_val"] == 1).sum()) > 0
        assert int((split_data["y_test"] == 1).sum()) > 0
