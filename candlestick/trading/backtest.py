from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd

from candlestick.config import ensure_dir
from candlestick.domain import (
    COLUMN_CLOSE,
    COLUMN_HIGH,
    COLUMN_LOW,
    COLUMN_OPEN,
    COLUMN_TS_EVENT,
    COLUMN_PROBA,
    COLUMN_PATTERN,
    COLUMN_DIRECTION,
    COLUMN_LABEL,
    COLUMN_WINDOW_END_IDX,
    COLUMN_WINDOW_END_TS,
    EPSILON_COMPARE,
    EPSILON_SAFE_DIVIDE,
    ExitReason,
    TradeDirection,
    UNKNOWN_PATTERN,
)

REJECT_NO_FUTURE_BAR = "no_future_bar"
REJECT_SAME_PATTERN_OVERLAP = "same_pattern_overlap"
REJECT_MAX_OPEN_POSITIONS = "max_open_positions"
REJECT_PORTFOLIO_HEAT = "portfolio_heat"
REJECT_ZERO_SIZE = "zero_size"
REJECT_INVALID_STOP = "invalid_stop"
REJECT_TINY_STOP = "tiny_stop"

SUMMARY_REJECTION_KEYS = (
    REJECT_SAME_PATTERN_OVERLAP,
    REJECT_MAX_OPEN_POSITIONS,
    REJECT_PORTFOLIO_HEAT,
    REJECT_ZERO_SIZE,
    REJECT_INVALID_STOP,
    REJECT_TINY_STOP,
    REJECT_NO_FUTURE_BAR,
)


class BacktestSummary(TypedDict):
    trades: int
    signals_total: int
    accepted_signals: int
    rejected_signals: int
    return_on_capital: float
    total_pnl: float
    win_rate: float
    sharpe: float
    profit_factor: float
    max_drawdown: float
    expectancy: float
    max_simultaneous_positions: int
    average_portfolio_heat: float
    max_portfolio_heat: float
    rejected_same_pattern_overlap: int
    rejected_max_open_positions: int
    rejected_portfolio_heat: int
    rejected_zero_size: int
    rejected_invalid_stop: int
    rejected_tiny_stop: int
    rejected_no_future_bar: int


def _atr(df: pd.DataFrame, idx: int, period: int = 14) -> float:
    start = max(1, idx - period + 1)
    window = df.iloc[start : idx + 1]
    prev_close = df[COLUMN_CLOSE].shift(1).iloc[start : idx + 1]
    tr = pd.concat(
        [
            window[COLUMN_HIGH] - window[COLUMN_LOW],
            (window[COLUMN_HIGH] - prev_close).abs(),
            (window[COLUMN_LOW] - prev_close).abs(),
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


def _coerce_direction(value: Any) -> TradeDirection:
    return TradeDirection(str(value))


def _resolve_stop(
    direction: TradeDirection,
    entry_price: float,
    pattern_invalidation: float,
    atr_value: float,
    stop_atr_multiple: float,
) -> float:
    if direction is TradeDirection.LONG:
        atr_stop = entry_price - stop_atr_multiple * atr_value
        pattern_stop = min(pattern_invalidation, entry_price - 1.0e-6)
        return float(max(pattern_stop, atr_stop))

    atr_stop = entry_price + stop_atr_multiple * atr_value
    pattern_stop = max(pattern_invalidation, entry_price + 1.0e-6)
    return float(min(pattern_stop, atr_stop))


def _simulate_exit(
    px: pd.DataFrame,
    entry_idx: int,
    direction: TradeDirection,
    stop: float,
    tp: float,
    time_stop_bars: int,
) -> tuple[int, ExitReason, float]:
    exit_idx = min(len(px) - 1, entry_idx + time_stop_bars)
    exit_price = float(px.iloc[exit_idx][COLUMN_CLOSE])

    for idx in range(entry_idx, exit_idx + 1):
        high = float(px.iloc[idx][COLUMN_HIGH])
        low = float(px.iloc[idx][COLUMN_LOW])

        if direction is TradeDirection.LONG:
            hit_stop = low <= stop
            hit_tp = high >= tp
        else:
            hit_stop = high >= stop
            hit_tp = low <= tp

        if hit_stop and hit_tp:
            hit_tp = False
        if hit_stop:
            return idx, ExitReason.STOP, stop
        if hit_tp:
            return idx, ExitReason.TAKE_PROFIT, tp

    return exit_idx, ExitReason.TIME_STOP, exit_price


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


def _summary_from_state(
    signals_total: int,
    trades_df: pd.DataFrame,
    rejected: Counter[str],
    capital_base: float,
    heat_history: list[float],
    max_simultaneous: int,
) -> BacktestSummary:
    rejected_total = int(sum(rejected.values()))
    base_summary: BacktestSummary = {
        "trades": int(len(trades_df)),
        "signals_total": int(signals_total),
        "accepted_signals": int(len(trades_df)),
        "rejected_signals": rejected_total,
        "return_on_capital": 0.0,
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "sharpe": 0.0,
        "profit_factor": 0.0,
        "max_drawdown": 0.0,
        "expectancy": 0.0,
        "max_simultaneous_positions": int(max_simultaneous),
        "average_portfolio_heat": float(np.mean(heat_history)) if heat_history else 0.0,
        "max_portfolio_heat": float(np.max(heat_history)) if heat_history else 0.0,
        "rejected_same_pattern_overlap": int(rejected[REJECT_SAME_PATTERN_OVERLAP]),
        "rejected_max_open_positions": int(rejected[REJECT_MAX_OPEN_POSITIONS]),
        "rejected_portfolio_heat": int(rejected[REJECT_PORTFOLIO_HEAT]),
        "rejected_zero_size": int(rejected[REJECT_ZERO_SIZE]),
        "rejected_invalid_stop": int(rejected[REJECT_INVALID_STOP]),
        "rejected_tiny_stop": int(rejected[REJECT_TINY_STOP]),
        "rejected_no_future_bar": int(rejected[REJECT_NO_FUTURE_BAR]),
    }
    if trades_df.empty:
        return base_summary

    pnl = trades_df["net_pnl"].to_numpy(dtype=float)
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    profit_factor = float(wins / abs(losses)) if losses < 0 else float("inf")
    daily_pnl, daily_returns = _daily_return_series(trades_df, capital_base)

    base_summary.update(
        {
            "return_on_capital": float(pnl.sum() / max(capital_base, 1.0e-8)),
            "total_pnl": float(pnl.sum()),
            "win_rate": float((pnl > 0).mean()),
            "sharpe": _compute_sharpe(daily_returns),
            "profit_factor": profit_factor,
            "max_drawdown": _max_drawdown(np.cumsum(daily_pnl)),
            "expectancy": float(pnl.mean()),
        }
    )
    return base_summary


def run_backtest_for_predictions(
    price_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    cfg: dict[str, Any],
    threshold: float,
) -> tuple[pd.DataFrame, BacktestSummary]:
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
    min_stop_distance_bps = float(bt_cfg.get("min_stop_distance_bps", 5.0))
    max_gross_exposure_multiple = float(bt_cfg.get("max_gross_exposure_multiple", 1.0))

    risk_budget = capital_base * risk_per_trade_bps / 10000.0
    max_heat_dollars = capital_base * max_portfolio_heat_bps / 10000.0

    px = price_df.sort_values(COLUMN_TS_EVENT).reset_index(drop=True).copy()
    trades: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    heat_history: list[float] = []
    open_positions: list[dict[str, Any]] = []
    max_simultaneous = 0

    signals = pred_df[pred_df[COLUMN_PROBA] >= threshold].copy()
    signals = signals.sort_values([COLUMN_WINDOW_END_IDX, COLUMN_PROBA], ascending=[True, False])

    for _, row in signals.iterrows():
        anchor = int(row[COLUMN_WINDOW_END_IDX])
        entry_idx = anchor + 1
        if entry_idx >= len(px):
            rejected[REJECT_NO_FUTURE_BAR] += 1
            continue

        open_positions = [position for position in open_positions if int(position["exit_idx"]) >= entry_idx]

        pattern = str(row.get(COLUMN_PATTERN, UNKNOWN_PATTERN))
        if not allow_same_pattern_overlap and any(position[COLUMN_PATTERN] == pattern for position in open_positions):
            rejected[REJECT_SAME_PATTERN_OVERLAP] += 1
            continue
        if len(open_positions) >= max_open_positions:
            rejected[REJECT_MAX_OPEN_POSITIONS] += 1
            continue

        direction = _coerce_direction(row[COLUMN_DIRECTION])
        sign = 1.0 if direction is TradeDirection.LONG else -1.0
        entry_ts = px.iloc[entry_idx][COLUMN_TS_EVENT]
        entry_price = float(px.iloc[entry_idx][COLUMN_OPEN])
        atr_value = max(_atr(px, entry_idx), EPSILON_COMPARE)
        default_stop = entry_price - atr_value if direction is TradeDirection.LONG else entry_price + atr_value
        supplied_stop = row.get("stop_price")
        if supplied_stop is None or pd.isna(supplied_stop):
            pattern_invalidation = float(default_stop)
        else:
            pattern_invalidation = float(supplied_stop)

        if direction is TradeDirection.LONG and pattern_invalidation >= entry_price:
            rejected[REJECT_INVALID_STOP] += 1
            continue
        if direction is TradeDirection.SHORT and pattern_invalidation <= entry_price:
            rejected[REJECT_INVALID_STOP] += 1
            continue

        stop = _resolve_stop(
            direction=direction,
            entry_price=entry_price,
            pattern_invalidation=pattern_invalidation,
            atr_value=atr_value,
            stop_atr_multiple=stop_atr_multiple,
        )
        risk_per_share = abs(entry_price - stop)
        if risk_per_share <= 1.0e-8:
            rejected[REJECT_INVALID_STOP] += 1
            continue
        min_stop_distance = entry_price * min_stop_distance_bps / 10000.0
        if risk_per_share < max(min_stop_distance, 1.0e-8):
            rejected[REJECT_TINY_STOP] += 1
            continue

        shares = int(np.floor(risk_budget / risk_per_share))
        max_shares = int(np.floor((capital_base * max_gross_exposure_multiple) / max(entry_price, 1.0e-8)))
        shares = min(shares, max_shares)
        if shares <= 0:
            rejected[REJECT_ZERO_SIZE] += 1
            continue

        trade_risk = shares * risk_per_share
        current_heat = sum(float(position["risk_dollars"]) for position in open_positions)
        if current_heat + trade_risk > max_heat_dollars + 1.0e-9:
            rejected[REJECT_PORTFOLIO_HEAT] += 1
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
                "exit_ts": px.iloc[exit_idx][COLUMN_TS_EVENT],
                "direction": direction.value,
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
                "exit_reason": exit_reason.value,
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
    summary = _summary_from_state(
        signals_total=len(signals),
        trades_df=trades_df,
        rejected=rejected,
        capital_base=capital_base,
        heat_history=heat_history,
        max_simultaneous=max_simultaneous,
    )
    return trades_df, summary


def save_backtest_outputs(
    trades_df: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path,
    pattern: str,
) -> None:
    out = ensure_dir(Path(output_dir) / pattern)
    trades_df.to_csv(out / "trades.csv", index=False)
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
