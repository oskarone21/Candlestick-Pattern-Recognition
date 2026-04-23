from __future__ import annotations

import pandas as pd

from chart_patterns.trading.backtest import run_backtest_for_predictions


def test_backtest_arithmetic_hits_tp(base_cfg):
    ts = pd.date_range("2024-01-02 09:30", periods=6, freq="15min", tz="America/New_York")
    px = pd.DataFrame(
        {
            "symbol": "SPY",
            "ts_event": ts,
            "open": [100.0, 100.0, 100.5, 101.0, 101.5, 102.0],
            "high": [100.2, 101.0, 102.5, 102.8, 103.0, 103.2],
            "low": [99.8, 99.7, 100.3, 100.8, 101.2, 101.8],
            "close": [100.0, 100.8, 102.0, 102.5, 102.8, 103.0],
            "volume": [1000, 1200, 1300, 1400, 1500, 1600],
        }
    )

    pred = pd.DataFrame(
        {
            "window_end_idx": [0],
            "window_end_ts": [str(ts[0])],
            "proba": [0.95],
            "label": [1],
            "direction": ["long"],
            "pattern_height": [2.0],
            "target_price": [102.0],
            "stop_price": [98.5],
            "pattern": ["double_bottom"],
        }
    )

    trades, summary = run_backtest_for_predictions(px, pred, base_cfg, threshold=0.5)

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "tp"
    assert trades.iloc[0]["shares"] > 0
    assert summary["trades"] == 1
    assert summary["win_rate"] == 1.0
    assert summary["total_pnl"] > 0
    assert summary["return_on_capital"] > 0


def test_backtest_blocks_same_pattern_overlap(base_cfg):
    base_cfg["backtest"]["allow_same_pattern_overlap"] = False
    base_cfg["backtest"]["max_open_positions"] = 4
    ts = pd.date_range("2024-01-02 09:30", periods=8, freq="15min", tz="America/New_York")
    px = pd.DataFrame(
        {
            "symbol": "SPY",
            "ts_event": ts,
            "open": [100.0, 100.0, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2],
            "high": [100.3, 100.4, 100.6, 100.8, 101.0, 101.2, 101.4, 101.6],
            "low": [99.8, 99.9, 100.0, 100.2, 100.4, 100.6, 100.8, 101.0],
            "close": [100.0, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.4],
            "volume": [1000] * 8,
        }
    )
    pred = pd.DataFrame(
        {
            "window_end_idx": [0, 1],
            "window_end_ts": [str(ts[0]), str(ts[1])],
            "proba": [0.95, 0.94],
            "label": [1, 1],
            "direction": ["long", "long"],
            "stop_price": [99.0, 99.1],
            "pattern": ["double_bottom", "double_bottom"],
        }
    )

    trades, summary = run_backtest_for_predictions(px, pred, base_cfg, threshold=0.5)

    assert len(trades) == 1
    assert summary["rejected_same_pattern_overlap"] == 1


def test_backtest_rejects_long_stop_above_entry(base_cfg):
    ts = pd.date_range("2024-01-02 09:30", periods=6, freq="15min", tz="America/New_York")
    px = pd.DataFrame(
        {
            "symbol": "SPY",
            "ts_event": ts,
            "open": [100.0, 100.0, 100.2, 100.4, 100.6, 100.8],
            "high": [100.3, 100.4, 100.6, 100.8, 101.0, 101.2],
            "low": [99.8, 99.9, 100.0, 100.2, 100.4, 100.6],
            "close": [100.0, 100.2, 100.4, 100.6, 100.8, 101.0],
            "volume": [1000] * 6,
        }
    )
    pred = pd.DataFrame(
        {
            "window_end_idx": [0],
            "window_end_ts": [str(ts[0])],
            "proba": [0.95],
            "label": [0],
            "direction": ["long"],
            "stop_price": [100.5],
            "pattern": ["double_bottom"],
        }
    )

    trades, summary = run_backtest_for_predictions(px, pred, base_cfg, threshold=0.5)

    assert trades.empty
    assert summary["rejected_invalid_stop"] == 1


def test_backtest_rejects_tiny_stop_distance(base_cfg):
    base_cfg["backtest"]["min_stop_distance_bps"] = 10
    ts = pd.date_range("2024-01-02 09:30", periods=6, freq="15min", tz="America/New_York")
    px = pd.DataFrame(
        {
            "symbol": "SPY",
            "ts_event": ts,
            "open": [100.0, 100.0, 100.2, 100.4, 100.6, 100.8],
            "high": [100.3, 100.4, 100.6, 100.8, 101.0, 101.2],
            "low": [99.8, 99.9, 100.0, 100.2, 100.4, 100.6],
            "close": [100.0, 100.2, 100.4, 100.6, 100.8, 101.0],
            "volume": [1000] * 6,
        }
    )
    pred = pd.DataFrame(
        {
            "window_end_idx": [0],
            "window_end_ts": [str(ts[0])],
            "proba": [0.95],
            "label": [0],
            "direction": ["long"],
            "stop_price": [99.95],
            "pattern": ["double_bottom"],
        }
    )

    trades, summary = run_backtest_for_predictions(px, pred, base_cfg, threshold=0.5)

    assert trades.empty
    assert summary["rejected_tiny_stop"] == 1


def test_backtest_caps_gross_exposure(base_cfg):
    base_cfg["backtest"]["risk_per_trade_bps"] = 1000
    base_cfg["backtest"]["max_gross_exposure_multiple"] = 1.0
    ts = pd.date_range("2024-01-02 09:30", periods=6, freq="15min", tz="America/New_York")
    px = pd.DataFrame(
        {
            "symbol": "SPY",
            "ts_event": ts,
            "open": [100.0, 100.0, 100.5, 101.0, 101.5, 102.0],
            "high": [100.2, 101.0, 102.5, 102.8, 103.0, 103.2],
            "low": [99.8, 99.7, 100.3, 100.8, 101.2, 101.8],
            "close": [100.0, 100.8, 102.0, 102.5, 102.8, 103.0],
            "volume": [1000, 1200, 1300, 1400, 1500, 1600],
        }
    )
    pred = pd.DataFrame(
        {
            "window_end_idx": [0],
            "window_end_ts": [str(ts[0])],
            "proba": [0.95],
            "label": [1],
            "direction": ["long"],
            "stop_price": [99.5],
            "pattern": ["double_bottom"],
        }
    )

    trades, summary = run_backtest_for_predictions(px, pred, base_cfg, threshold=0.5)

    assert len(trades) == 1
    assert summary["trades"] == 1
    assert trades.iloc[0]["shares"] == 10000
