"""Replay-based paper trading for the two-stage product pipeline."""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.trading.inference import predict_sequence_probabilities
from src.trading.trade_labeling import build_live_trade_feature_vector, pattern_direction


def build_stage1_window(df_ohlcv: pd.DataFrame, anchor_bar: int, cfg: dict) -> np.ndarray | None:
    """Extract and normalise one Stage 1 inference window."""
    col_cfg = cfg["data_source"]["columns"]
    lookback = cfg["windowing"]["lookback_bars"]
    cols = [
        col_cfg["open"],
        col_cfg["high"],
        col_cfg["low"],
        col_cfg["close"],
        col_cfg["volume"],
    ]

    start = anchor_bar - lookback + 1
    end = anchor_bar + 1
    if start < 0:
        return None

    window = df_ohlcv[cols].iloc[start:end].to_numpy(dtype=float)
    if window.shape[0] != lookback:
        return None

    w_min = window.min(axis=0)
    w_max = window.max(axis=0)
    denom = np.where((w_max - w_min) > 0, w_max - w_min, 1.0)
    return ((window - w_min) / denom).astype(np.float32)


def run_paper_replay(
    stage1_model: torch.nn.Module,
    stage2_model,
    df_ohlcv: pd.DataFrame,
    feature_frame: pd.DataFrame,
    cfg: dict,
    device: torch.device,
    stage1_threshold: float,
    stage2_threshold: float,
    save_dir: str,
) -> dict:
    """Replay the last N bars as a paper-trading session."""
    os.makedirs(save_dir, exist_ok=True)

    replay_cfg = cfg["product"]["replay"]
    trade_cfg = cfg["product"]["stage2"]["trade_labeling"]
    col_cfg = cfg["data_source"]["columns"]

    bars = int(replay_cfg.get("bars", 96))
    start_capital = float(replay_cfg.get("starting_capital", 100000.0))
    size_per_trade = float(replay_cfg.get("size_per_trade_usd", 10000.0))
    max_positions = int(replay_cfg.get("max_open_positions", 1))

    direction = pattern_direction(cfg["labeling"]["active_pattern"])
    horizon = int(trade_cfg["horizon_bars"])
    take_profit_pct = float(trade_cfg["take_profit_pct"])
    stop_loss_pct = float(trade_cfg["stop_loss_pct"])
    round_trip_cost_pct = float(trade_cfg.get("round_trip_cost_bps", 0.0)) / 10000.0
    same_bar_priority = trade_cfg.get("same_bar_exit_priority", "stop").lower()

    close = df_ohlcv[col_cfg["close"]].to_numpy(dtype=float)
    high = df_ohlcv[col_cfg["high"]].to_numpy(dtype=float)
    low = df_ohlcv[col_cfg["low"]].to_numpy(dtype=float)

    start_bar = max(cfg["windowing"]["lookback_bars"] - 1, len(df_ohlcv) - bars)
    capital = start_capital
    position = None
    trades = []
    equity_curve = []

    for bar in range(start_bar, len(df_ohlcv)):
        timestamp = df_ohlcv.index[bar]

        if position is not None:
            exit_info = _maybe_exit_position(
                position=position,
                bar=bar,
                timestamps=df_ohlcv.index,
                high=high,
                low=low,
                close=close,
                direction=direction,
                horizon=horizon,
                round_trip_cost_pct=round_trip_cost_pct,
                same_bar_priority=same_bar_priority,
            )
            if exit_info is not None:
                trade_return = exit_info["net_return"]
                pnl_usd = size_per_trade * trade_return
                capital += pnl_usd
                trades.append(
                    {
                        **position,
                        **exit_info,
                        "pnl_usd": float(pnl_usd),
                    }
                )
                position = None

        if position is None and max_positions > 0:
            feature_vector = None
            stage1_window = build_stage1_window(df_ohlcv, bar, cfg)
            if stage1_window is not None:
                stage1_prob = float(
                    predict_sequence_probabilities(
                        stage1_model,
                        stage1_window[None, :, :],
                        device=device,
                        batch_size=1,
                    )[0]
                )
                if stage1_prob >= stage1_threshold:
                    feature_vector = build_live_trade_feature_vector(
                        feature_frame=feature_frame,
                        bar_idx=bar,
                        stage1_prob=stage1_prob,
                    )
                    if feature_vector is not None:
                        stage2_prob = float(stage2_model.predict_proba(feature_vector[None, :])[:, 1][0])
                        if stage2_prob >= stage2_threshold:
                            entry_price = close[bar]
                            if direction == 1:
                                take_profit = entry_price * (1.0 + take_profit_pct)
                                stop_loss = entry_price * (1.0 - stop_loss_pct)
                            else:
                                take_profit = entry_price * (1.0 - take_profit_pct)
                                stop_loss = entry_price * (1.0 + stop_loss_pct)

                            position = {
                                "entry_bar": int(bar),
                                "entry_time": str(timestamp),
                                "entry_price": float(entry_price),
                                "stage1_prob": stage1_prob,
                                "stage2_prob": stage2_prob,
                                "take_profit": float(take_profit),
                                "stop_loss": float(stop_loss),
                            }

        equity_curve.append({"timestamp": str(timestamp), "capital": float(capital)})

    if position is not None:
        last_bar = len(df_ohlcv) - 1
        exit_price = close[last_bar]
        gross_return = direction * ((exit_price - position["entry_price"]) / position["entry_price"])
        net_return = gross_return - round_trip_cost_pct
        pnl_usd = size_per_trade * net_return
        capital += pnl_usd
        trades.append(
            {
                **position,
                "exit_bar": int(last_bar),
                "exit_time": str(df_ohlcv.index[last_bar]),
                "exit_price": float(exit_price),
                "exit_reason": "replay_end",
                "gross_return": float(gross_return),
                "net_return": float(net_return),
                "pnl_usd": float(pnl_usd),
            }
        )
        equity_curve.append({"timestamp": str(df_ohlcv.index[last_bar]), "capital": float(capital)})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    trades_path = os.path.join(save_dir, "paper_trades.csv")
    equity_path = os.path.join(save_dir, "paper_equity.csv")
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)

    summary = {
        "starting_capital": start_capital,
        "ending_capital": float(capital),
        "net_pnl_usd": float(capital - start_capital),
        "n_trades": int(len(trades_df)),
        "win_rate_pct": float(100.0 * (trades_df["net_return"] > 0).mean()) if len(trades_df) else 0.0,
        "avg_net_return_pct": float(100.0 * trades_df["net_return"].mean()) if len(trades_df) else 0.0,
    }

    with open(os.path.join(save_dir, "paper_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_equity_curve(equity_df, os.path.join(save_dir, "paper_equity.png"))

    print(
        "[paper] Replay summary — "
        f"trades: {summary['n_trades']} | "
        f"net_pnl_usd: {summary['net_pnl_usd']:.2f} | "
        f"ending_capital: {summary['ending_capital']:.2f}"
    )
    return summary


def _maybe_exit_position(
    position: dict,
    bar: int,
    timestamps: pd.Index,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    direction: int,
    horizon: int,
    round_trip_cost_pct: float,
    same_bar_priority: str,
) -> dict | None:
    """Check whether the open position exits on the current bar."""
    if bar <= position["entry_bar"]:
        return None

    hit_stop = False
    hit_target = False
    if direction == 1:
        hit_stop = low[bar] <= position["stop_loss"]
        hit_target = high[bar] >= position["take_profit"]
    else:
        hit_stop = high[bar] >= position["stop_loss"]
        hit_target = low[bar] <= position["take_profit"]

    exit_price = None
    exit_reason = None

    if hit_stop and hit_target:
        if same_bar_priority == "target":
            exit_price = position["take_profit"]
            exit_reason = "take_profit_same_bar"
        else:
            exit_price = position["stop_loss"]
            exit_reason = "stop_loss_same_bar"
    elif hit_stop:
        exit_price = position["stop_loss"]
        exit_reason = "stop_loss"
    elif hit_target:
        exit_price = position["take_profit"]
        exit_reason = "take_profit"
    elif bar - position["entry_bar"] >= horizon:
        exit_price = close[bar]
        exit_reason = "horizon"

    if exit_price is None or exit_reason is None:
        return None

    gross_return = direction * ((exit_price - position["entry_price"]) / position["entry_price"])
    net_return = gross_return - round_trip_cost_pct
    return {
        "exit_bar": int(bar),
        "exit_time": str(timestamps[bar]),
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "gross_return": float(gross_return),
        "net_return": float(net_return),
    }


def _plot_equity_curve(equity_df: pd.DataFrame, out_path: str) -> None:
    """Save a simple equity-curve plot for the replay session."""
    if equity_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_df["capital"].to_numpy(dtype=float), color="steelblue", lw=2)
    ax.set_title("Paper Trading Equity Curve")
    ax.set_xlabel("Replay Step")
    ax.set_ylabel("Capital")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
