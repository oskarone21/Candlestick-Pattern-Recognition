"""Stage 2 trade labels and causal feature engineering."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


BULLISH_PATTERNS = {"double_bottom", "inverse_head_shoulders"}
BEARISH_PATTERNS = {"double_top", "head_shoulders"}


@dataclass
class TradeOutcome:
    """Forward trade result for one anchor bar."""

    anchor_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    gross_return: float
    net_return: float
    profitable_label: int
    exit_reason: str


def pattern_direction(pattern_name: str) -> int:
    """Return +1 for long patterns and -1 for short patterns."""
    if pattern_name in BULLISH_PATTERNS:
        return 1
    if pattern_name in BEARISH_PATTERNS:
        return -1
    raise ValueError(f"Unsupported pattern for trading direction: {pattern_name}")


def build_causal_feature_frame(df_ohlcv: pd.DataFrame, cfg: dict, pattern_name: str) -> pd.DataFrame:
    """Create causal tabular features available at each bar close."""
    col_cfg = cfg["data_source"]["columns"]
    direction = pattern_direction(pattern_name)

    open_ = df_ohlcv[col_cfg["open"]].astype(float)
    high = df_ohlcv[col_cfg["high"]].astype(float)
    low = df_ohlcv[col_cfg["low"]].astype(float)
    close = df_ohlcv[col_cfg["close"]].astype(float)
    volume = df_ohlcv[col_cfg["volume"]].astype(float)

    returns = close.pct_change()
    tr = pd.concat(
        [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    feature_frame = pd.DataFrame(index=df_ohlcv.index)
    feature_frame["signed_ret_1"] = direction * close.pct_change(1)
    feature_frame["signed_ret_4"] = direction * close.pct_change(4)
    feature_frame["signed_ret_8"] = direction * close.pct_change(8)
    feature_frame["signed_ret_16"] = direction * close.pct_change(16)
    feature_frame["signed_close_vs_ma8"] = direction * (
        (close / close.rolling(8).mean()) - 1.0
    )
    feature_frame["signed_close_vs_ma20"] = direction * (
        (close / close.rolling(20).mean()) - 1.0
    )
    feature_frame["signed_trend_20"] = direction * close.pct_change(20)
    feature_frame["vol_8"] = returns.rolling(8).std()
    feature_frame["vol_20"] = returns.rolling(20).std()
    feature_frame["atr_14_pct"] = tr.rolling(14).mean() / close
    feature_frame["range_pct"] = (high - low) / close
    feature_frame["signed_body_pct"] = direction * ((close - open_) / open_)
    feature_frame["upper_wick_pct"] = (high - np.maximum(open_, close)) / close
    feature_frame["lower_wick_pct"] = (np.minimum(open_, close) - low) / close
    feature_frame["volume_ratio_1_20"] = volume / volume.rolling(20).mean()
    feature_frame["volume_ratio_8_20"] = volume.rolling(8).mean() / volume.rolling(20).mean()
    feature_frame["signed_range_expansion_8"] = direction * (
        close.pct_change(1) / tr.rolling(8).mean()
    )

    return feature_frame.replace([np.inf, -np.inf], np.nan)


def build_live_trade_feature_vector(
    feature_frame: pd.DataFrame,
    bar_idx: int,
    stage1_prob: float,
) -> np.ndarray | None:
    """Build one Stage 2 feature vector for live/paper-trading inference."""
    row = feature_frame.iloc[bar_idx]
    if row.isna().any():
        return None

    base_values = row.to_numpy(dtype=float)
    return np.concatenate(
        [base_values, np.array([stage1_prob, stage1_prob - 0.5], dtype=float)]
    ).astype(np.float32)


def simulate_trade_outcome(
    df_ohlcv: pd.DataFrame,
    anchor_bar: int,
    cfg: dict,
    pattern_name: str,
) -> TradeOutcome | None:
    """Create a Stage 2 profitability label from a forward trade simulation."""
    col_cfg = cfg["data_source"]["columns"]
    trade_cfg = cfg["product"]["stage2"]["trade_labeling"]

    direction = pattern_direction(pattern_name)
    horizon = int(trade_cfg["horizon_bars"])
    take_profit_pct = float(trade_cfg["take_profit_pct"])
    stop_loss_pct = float(trade_cfg["stop_loss_pct"])
    min_net_return_pct = float(trade_cfg.get("min_net_return_pct", 0.0))
    round_trip_cost_pct = float(trade_cfg.get("round_trip_cost_bps", 0.0)) / 10000.0
    same_bar_priority = trade_cfg.get("same_bar_exit_priority", "stop").lower()

    close = df_ohlcv[col_cfg["close"]].to_numpy(dtype=float)
    high = df_ohlcv[col_cfg["high"]].to_numpy(dtype=float)
    low = df_ohlcv[col_cfg["low"]].to_numpy(dtype=float)

    if anchor_bar + 1 >= len(close):
        return None

    expiry_bar = anchor_bar + horizon
    if expiry_bar >= len(close):
        return None

    entry_price = close[anchor_bar]
    if entry_price <= 0:
        return None

    if direction == 1:
        take_profit = entry_price * (1.0 + take_profit_pct)
        stop_loss = entry_price * (1.0 - stop_loss_pct)
    else:
        take_profit = entry_price * (1.0 - take_profit_pct)
        stop_loss = entry_price * (1.0 + stop_loss_pct)

    exit_bar = expiry_bar
    exit_price = close[expiry_bar]
    exit_reason = "horizon"

    for bar in range(anchor_bar + 1, expiry_bar + 1):
        bar_high = high[bar]
        bar_low = low[bar]

        if direction == 1:
            hit_stop = bar_low <= stop_loss
            hit_target = bar_high >= take_profit
        else:
            hit_stop = bar_high >= stop_loss
            hit_target = bar_low <= take_profit

        if hit_stop and hit_target:
            chosen = same_bar_priority
            if chosen == "target":
                exit_price = take_profit
                exit_reason = "take_profit_same_bar"
            else:
                exit_price = stop_loss
                exit_reason = "stop_loss_same_bar"
            exit_bar = bar
            break
        if hit_stop:
            exit_price = stop_loss
            exit_reason = "stop_loss"
            exit_bar = bar
            break
        if hit_target:
            exit_price = take_profit
            exit_reason = "take_profit"
            exit_bar = bar
            break

    gross_return = direction * ((exit_price - entry_price) / entry_price)
    net_return = gross_return - round_trip_cost_pct
    profitable_label = int(net_return >= min_net_return_pct)

    return TradeOutcome(
        anchor_bar=anchor_bar,
        exit_bar=exit_bar,
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        gross_return=float(gross_return),
        net_return=float(net_return),
        profitable_label=int(profitable_label),
        exit_reason=exit_reason,
    )


def build_trade_dataset(
    anchor_metadata: pd.DataFrame,
    df_ohlcv: pd.DataFrame,
    stage1_probs: np.ndarray,
    cfg: dict,
    pattern_name: str,
) -> dict:
    """Build the Stage 2 dataset from full anchor metadata and forward trade labels."""
    if len(anchor_metadata) != len(stage1_probs):
        raise ValueError(
            "anchor_metadata and stage1_probs must have the same length. "
            f"Got {len(anchor_metadata)} and {len(stage1_probs)}."
        )

    feature_frame = build_causal_feature_frame(df_ohlcv, cfg, pattern_name)

    feature_rows = []
    metadata_rows = []
    stage1_probs = np.asarray(stage1_probs, dtype=float)

    for row, stage1_prob in zip(anchor_metadata.itertuples(index=False), stage1_probs):
        anchor_bar = int(row.anchor_bar)
        outcome = simulate_trade_outcome(df_ohlcv, anchor_bar, cfg, pattern_name)
        if outcome is None:
            continue

        features = build_live_trade_feature_vector(feature_frame, anchor_bar, stage1_prob)
        if features is None:
            continue

        feature_rows.append(features)
        metadata_rows.append(
            {
                "anchor_bar": anchor_bar,
                "anchor_time": str(getattr(row, "anchor_time", "")),
                "stage1_target": int(row.stage1_target),
                "stage1_prob": float(stage1_prob),
                "window_reason": str(getattr(row, "label_reason", "background")),
                "label_source": str(getattr(row, "label_source", "background")),
                **asdict(outcome),
            }
        )

    if not feature_rows:
        raise ValueError("Stage 2 dataset is empty after trade labeling.")

    base_feature_names = list(feature_frame.columns)
    feature_names = base_feature_names + ["stage1_prob", "stage1_margin"]
    metadata = pd.DataFrame(metadata_rows)

    X = np.vstack(feature_rows).astype(np.float32)
    y = metadata["profitable_label"].to_numpy(dtype=np.int64)
    net_returns = metadata["net_return"].to_numpy(dtype=float)
    anchors = metadata["anchor_bar"].to_numpy(dtype=int)

    print(
        "[stage2] Trade dataset — "
        f"{len(y)} samples | profitable: {int(y.sum())} | "
        f"non-profitable: {int((y == 0).sum())}"
    )

    return {
        "X": X,
        "y": y,
        "net_returns": net_returns,
        "anchors": anchors,
        "feature_names": feature_names,
        "feature_frame": feature_frame,
        "metadata": metadata,
    }
