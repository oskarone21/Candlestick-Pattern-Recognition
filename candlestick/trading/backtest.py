from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from candlestick.config import ensure_dir


def _atr(df: pd.DataFrame, idx: int, period: int = 14) -> float:
    start = max(1, idx - period + 1)
    window = df.iloc[start : idx + 1]
    prev_close = df["close"].shift(1).iloc[start : idx + 1]
    tr = pd.concat(
        [
            window["high"] - window["low"],
            (window["high"] - prev_close).abs(),
            (window["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(tr.mean()) if len(tr) else 0.0


def _compute_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return 0.0
    return float((np.mean(returns) / std) * np.sqrt(252.0))


def _max_drawdown(cum_pnl: np.ndarray) -> float:
    if len(cum_pnl) == 0:
        return 0.0
    peaks = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - peaks
    return float(drawdowns.min())


def _resolve_stop(
    direction: str,
    entry_price: float,
    pattern_invalidation: float,
    atr_value: float,
    stop_atr_multiple: float,
) -> float:
    atr_stop = entry_price - stop_atr_multiple * atr_value if direction == "long" else entry_price + stop_atr_multiple * atr_value

    if direction == "long":
        pattern_stop = min(pattern_invalidation, entry_price - 1.0e-6)
        return float(max(pattern_stop, atr_stop))

    pattern_stop = max(pattern_invalidation, entry_price + 1.0e-6)
    return float(min(pattern_stop, atr_stop))


def _simulate_exit(
    px: pd.DataFrame,
    entry_idx: int,
    direction: str,
    stop: float,
    tp: float,
    time_stop_bars: int,
) -> tuple[int, str, float]:
    exit_idx = min(len(px) - 1, entry_idx + time_stop_bars)
    exit_reason = "time_stop"
    exit_price = float(px.iloc[exit_idx]["close"])

    for i in range(entry_idx, exit_idx + 1):
        hi = float(px.iloc[i]["high"])
        lo = float(px.iloc[i]["low"])

        if direction == "long":
            hit_stop = lo <= stop
            hit_tp = hi >= tp
        else:
            hit_stop = hi >= stop
            hit_tp = lo <= tp

        if hit_stop and hit_tp:
            hit_tp = False

        if hit_stop:
            return i, "stop", stop
        if hit_tp:
            return i, "tp", tp

    return exit_idx, exit_reason, exit_price


def _daily_return_series(trades_df: pd.DataFrame, capital_base: float) -> tuple[np.ndarray, np.ndarray]:
    if trades_df.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    exits = pd.to_datetime(trades_df["exit_ts"], errors="coerce")
    dates = exits.dt.tz_localize(None).dt.normalize()
    pnl_by_day = (
        pd.DataFrame({"date": dates, "net_pnl": trades_df["net_pnl"]})
        .dropna(subset=["date"])
        .groupby("date")["net_pnl"]
        .sum()
    )
    if pnl_by_day.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    full_index = pd.bdate_range(pnl_by_day.index.min(), pnl_by_day.index.max())
    pnl_by_day = pnl_by_day.reindex(full_index, fill_value=0.0)
    daily_pnl = pnl_by_day.to_numpy(dtype=float)
    return daily_pnl, daily_pnl / max(capital_base, 1.0e-8)


def run_backtest_for_predictions(
    price_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    cfg: dict[str, Any],
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Simulate post-breakout trades with risk-based SPY share sizing."""
    bt_cfg = cfg.get("backtest", {})
    capital_base = float(bt_cfg.get("capital_base", 1_000_000.0))
    transaction_cost_bps = float(bt_cfg.get("transaction_cost_bps_per_side", 1.0))
    slippage_bps = float(bt_cfg.get("slippage_bps_per_side", 0.0))
    time_stop_bars = int(bt_cfg.get("time_stop_bars", 26))
    stop_atr_multiple = float(bt_cfg.get("stop_atr_multiple", bt_cfg.get("atr_stop_multiplier", 1.0)))
    target_r_multiple = float(bt_cfg.get("target_r_multiple", 1.5))
    risk_per_trade_bps = float(bt_cfg.get("risk_per_trade_bps", 25.0))
    max_portfolio_heat_bps = float(bt_cfg.get("max_portfolio_heat_bps", 100.0))
    max_open_positions = int(bt_cfg.get("max_open_positions", 4))
    allow_same_pattern_overlap = bool(bt_cfg.get("allow_same_pattern_overlap", False))

    risk_budget = capital_base * risk_per_trade_bps / 10000.0
    max_heat_dollars = capital_base * max_portfolio_heat_bps / 10000.0

    px = price_df.sort_values("ts_event").reset_index(drop=True).copy()
    trades: list[dict[str, Any]] = []
    rejected = Counter()
    heat_history: list[float] = []
    open_positions: list[dict[str, Any]] = []
    max_simultaneous = 0

    signals = pred_df[pred_df["proba"] >= threshold].copy()
    signals = signals.sort_values(["window_end_idx", "proba"], ascending=[True, False])

    for _, row in signals.iterrows():
        anchor = int(row["window_end_idx"])
        entry_idx = anchor + 1
        if entry_idx >= len(px):
            rejected["no_future_bar"] += 1
            continue

        open_positions = [pos for pos in open_positions if int(pos["exit_idx"]) >= entry_idx]

        pattern = str(row.get("pattern", "unknown"))
        if not allow_same_pattern_overlap and any(pos["pattern"] == pattern for pos in open_positions):
            rejected["same_pattern_overlap"] += 1
            continue

        if len(open_positions) >= max_open_positions:
            rejected["max_open_positions"] += 1
            continue

        direction = str(row["direction"])
        sign = 1.0 if direction == "long" else -1.0
        entry_ts = px.iloc[entry_idx]["ts_event"]
        entry_price = float(px.iloc[entry_idx]["open"])
        atr_value = max(_atr(px, entry_idx), 1.0e-6)

        pattern_invalidation = float(
            row.get(
                "stop_price",
                entry_price - atr_value if direction == "long" else entry_price + atr_value,
            )
        )
        stop = _resolve_stop(
            direction=direction,
            entry_price=entry_price,
            pattern_invalidation=pattern_invalidation,
            atr_value=atr_value,
            stop_atr_multiple=stop_atr_multiple,
        )
        risk_per_share = abs(entry_price - stop)
        if risk_per_share <= 1.0e-8:
            rejected["invalid_stop"] += 1
            continue

        shares = int(np.floor(risk_budget / risk_per_share))
        if shares <= 0:
            rejected["zero_size"] += 1
            continue

        trade_risk = shares * risk_per_share
        current_heat = sum(float(pos["risk_dollars"]) for pos in open_positions)
        if current_heat + trade_risk > max_heat_dollars + 1.0e-9:
            rejected["portfolio_heat"] += 1
            continue

        tp = entry_price + sign * target_r_multiple * risk_per_share
        exit_idx, exit_reason, exit_price = _simulate_exit(
            px=px,
            entry_idx=entry_idx,
            direction=direction,
            stop=stop,
            tp=tp,
            time_stop_bars=time_stop_bars,
        )

        entry_notional = shares * entry_price
        exit_notional = shares * exit_price
        entry_cost = entry_notional * (transaction_cost_bps + slippage_bps) / 10000.0
        exit_cost = exit_notional * (transaction_cost_bps + slippage_bps) / 10000.0
        total_cost = entry_cost + exit_cost

        gross_pnl = sign * (exit_price - entry_price) * shares
        net_pnl = gross_pnl - total_cost
        portfolio_return = net_pnl / max(capital_base, 1.0e-8)
        r_multiple = net_pnl / max(trade_risk, 1.0e-8)
        holding_bars = int(exit_idx - entry_idx)
        heat_after_entry = (current_heat + trade_risk) / max(capital_base, 1.0e-8)

        trades.append(
            {
                "pattern": pattern,
                "anchor_ts": row.get("window_end_ts"),
                "entry_ts": entry_ts,
                "exit_ts": px.iloc[exit_idx]["ts_event"],
                "direction": direction,
                "entry_idx": int(entry_idx),
                "exit_idx": int(exit_idx),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": int(shares),
                "entry_notional": float(entry_notional),
                "exit_notional": float(exit_notional),
                "risk_dollars": float(trade_risk),
                "portfolio_heat_after_entry": float(heat_after_entry),
                "tp": float(tp),
                "stop": float(stop),
                "exit_reason": exit_reason,
                "holding_bars": holding_bars,
                "gross_pnl": float(gross_pnl),
                "net_pnl": float(net_pnl),
                "return": float(portfolio_return),
                "r_multiple": float(r_multiple),
                "proba": float(row["proba"]),
                "proba_raw": float(row.get("proba_raw", row["proba"])),
                "label": int(row.get("label", 0)),
            }
        )

        open_positions.append(
            {
                "pattern": pattern,
                "exit_idx": int(exit_idx),
                "risk_dollars": float(trade_risk),
            }
        )
        heat_history.append(float(heat_after_entry))
        max_simultaneous = max(max_simultaneous, len(open_positions))

    trades_df = pd.DataFrame(trades)
    rejected_total = int(sum(rejected.values()))
    if trades_df.empty:
        summary = {
            "trades": 0,
            "signals_total": int(len(signals)),
            "accepted_signals": 0,
            "rejected_signals": rejected_total,
            "return_on_capital": 0.0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "expectancy": 0.0,
            "max_simultaneous_positions": 0,
            "average_portfolio_heat": 0.0,
            "max_portfolio_heat": 0.0,
            "rejected_same_pattern_overlap": int(rejected["same_pattern_overlap"]),
            "rejected_max_open_positions": int(rejected["max_open_positions"]),
            "rejected_portfolio_heat": int(rejected["portfolio_heat"]),
            "rejected_zero_size": int(rejected["zero_size"]),
            "rejected_invalid_stop": int(rejected["invalid_stop"]),
            "rejected_no_future_bar": int(rejected["no_future_bar"]),
        }
        return trades_df, summary

    pnl = trades_df["net_pnl"].to_numpy(dtype=float)
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    profit_factor = float(wins / abs(losses)) if losses < 0 else float("inf")
    daily_pnl, daily_returns = _daily_return_series(trades_df, capital_base)

    summary = {
        "trades": int(len(trades_df)),
        "signals_total": int(len(signals)),
        "accepted_signals": int(len(trades_df)),
        "rejected_signals": rejected_total,
        "return_on_capital": float(pnl.sum() / max(capital_base, 1.0e-8)),
        "total_pnl": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()),
        "sharpe": _compute_sharpe(daily_returns),
        "profit_factor": profit_factor,
        "max_drawdown": _max_drawdown(np.cumsum(daily_pnl)),
        "expectancy": float(pnl.mean()),
        "max_simultaneous_positions": int(max_simultaneous),
        "average_portfolio_heat": float(np.mean(heat_history)) if heat_history else 0.0,
        "max_portfolio_heat": float(np.max(heat_history)) if heat_history else 0.0,
        "rejected_same_pattern_overlap": int(rejected["same_pattern_overlap"]),
        "rejected_max_open_positions": int(rejected["max_open_positions"]),
        "rejected_portfolio_heat": int(rejected["portfolio_heat"]),
        "rejected_zero_size": int(rejected["zero_size"]),
        "rejected_invalid_stop": int(rejected["invalid_stop"]),
        "rejected_no_future_bar": int(rejected["no_future_bar"]),
    }
    return trades_df, summary


def save_backtest_outputs(
    trades_df: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path,
    pattern: str,
) -> None:
    out = ensure_dir(Path(output_dir) / pattern)
    trades_path = out / "trades.csv"
    summary_path = out / "summary.json"

    trades_df.to_csv(trades_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
