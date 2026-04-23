from __future__ import annotations

import pandas as pd
import pytest

from chart_patterns.labeling.pattern_rules import detect_pattern_events


def _df_from_close(close_values: list[float]) -> pd.DataFrame:
    ts = pd.date_range("2024-01-02 09:30", periods=len(close_values), freq="15min", tz="America/New_York")
    open_ = [close_values[0]] + close_values[:-1]
    high = [max(o, c) + 0.2 for o, c in zip(open_, close_values)]
    low = [min(o, c) - 0.2 for o, c in zip(open_, close_values)]
    volume = [1200 + (i % 5) * 50 for i in range(len(close_values))]
    return pd.DataFrame(
        {
            "symbol": "SPY",
            "ts_event": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close_values,
            "volume": volume,
        }
    )


def _test_cfg(base_cfg: dict) -> dict:
    cfg = base_cfg
    cfg["labeling"]["extrema_detection"]["smoothing"]["bandwidth"]["default_bandwidth"] = 1.0
    cfg["labeling"]["extrema_detection"]["extrema_validation"]["min_extrema_separation_bars"] = 1
    cfg["labeling"]["confirmation"]["breakout_confirmation_pct"] = 0.01
    cfg["labeling"]["confirmation"]["confirm_break_within_bars"] = 10
    cfg["labeling"]["labeling_policy"]["hard_negative_sampling"]["enabled"] = False

    for p in ["head_shoulders", "inverse_head_shoulders", "double_top", "double_bottom"]:
        cfg["labeling"][p]["volume_rules"]["required"] = False
    cfg["labeling"]["head_shoulders"]["geometry"]["shoulder_symmetry_tolerance_pct"] = 0.12
    cfg["labeling"]["head_shoulders"]["geometry"]["neckline_point_symmetry_tolerance_pct"] = 0.20
    cfg["labeling"]["head_shoulders"]["geometry"]["neckline_max_slope"] = 0.4
    cfg["labeling"]["head_shoulders"]["geometry"]["min_peak_separation_bars"] = 1
    cfg["labeling"]["head_shoulders"]["geometry"]["max_peak_separation_bars"] = 20
    cfg["labeling"]["inverse_head_shoulders"]["geometry"]["shoulder_symmetry_tolerance_pct"] = 0.12
    cfg["labeling"]["inverse_head_shoulders"]["geometry"]["neckline_point_symmetry_tolerance_pct"] = 0.20
    cfg["labeling"]["inverse_head_shoulders"]["geometry"]["neckline_max_slope"] = 0.4
    cfg["labeling"]["inverse_head_shoulders"]["geometry"]["min_trough_separation_bars"] = 1
    cfg["labeling"]["inverse_head_shoulders"]["geometry"]["max_trough_separation_bars"] = 20

    return cfg


@pytest.mark.parametrize(
    "pattern,close_values",
    [
        ("head_shoulders", [9, 10, 12, 11, 10, 11, 14, 12, 10.1, 11, 12.2, 11, 9.4, 9.2, 9.0]),
        ("inverse_head_shoulders", [16, 15, 13, 14, 15, 14, 11, 13, 15, 14, 13.2, 14, 15.6, 15.8, 16.0]),
        ("double_top", [8, 9, 10, 12, 11, 10, 9, 10, 12, 11, 10, 8.8, 8.5, 8.3]),
        ("double_bottom", [16, 15, 14, 12, 13, 14, 15, 14, 12.1, 13, 14, 15.2, 15.5]),
    ],
)
def test_pattern_positive_detection(base_cfg, pattern, close_values):
    cfg = _test_cfg(base_cfg)
    df = _df_from_close(close_values)
    events = detect_pattern_events(df, pattern, cfg)

    assert not events.empty
    assert (events["label"] == 1).any()


def test_pattern_negative_reason_appears(base_cfg):
    cfg = _test_cfg(base_cfg)
    # Double top shape without confirmed breakdown.
    close_values = [8, 9, 10, 12, 11, 10, 9.3, 10, 12, 11.2, 10.6, 10.2, 10.1, 10.0]
    df = _df_from_close(close_values)

    events = detect_pattern_events(df, "double_top", cfg)
    assert not events.empty
    assert set(events["reason"]).intersection({"failed_breakout", "near_miss"})


def test_pattern_outputs_keep_string_values(base_cfg):
    cfg = _test_cfg(base_cfg)
    df = _df_from_close([16, 15, 14, 12, 13, 14, 15, 14, 12.1, 13, 14, 15.2, 15.5])

    events = detect_pattern_events(df, "double_bottom", cfg)

    assert not events.empty
    assert set(events["pattern"]) == {"double_bottom"}
    assert set(events["direction"]) == {"long"}
    assert set(events["reason"]).issubset({"confirmed_breakout", "failed_breakout", "near_miss", "partial"})
