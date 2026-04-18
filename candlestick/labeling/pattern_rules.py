from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from candlestick.labeling.extrema import Extremum, detect_extrema_from_close


@dataclass
class PatternEvent:
    symbol: str
    pattern: str
    label: int
    reason: str
    anchor_idx: int
    breakout_idx: int | None
    ts_event: pd.Timestamp
    direction: str
    neckline: float
    pattern_height: float
    stop_price: float
    target_price: float


def _relative_diff(a: float, b: float) -> float:
    denom = max((abs(a) + abs(b)) / 2.0, 1.0e-12)
    return abs(a - b) / denom


def _close(df: pd.DataFrame, idx: int) -> float:
    return float(df.iloc[idx]["close"])


def _volume(df: pd.DataFrame, idx: int) -> float:
    return float(df.iloc[idx]["volume"])


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

    value = float(tr.mean()) if not tr.empty else float((df.iloc[idx]["high"] - df.iloc[idx]["low"]))
    return max(value, 1.0e-8)


def _find_breakout(
    df: pd.DataFrame,
    start_idx: int,
    max_horizon: int,
    neckline_at_idx,
    beta: float,
    direction: str,
    min_volume: float | None,
) -> int | None:
    end_idx = min(len(df) - 1, start_idx + max_horizon)
    for i in range(start_idx + 1, end_idx + 1):
        px = _close(df, i)
        neck = neckline_at_idx(i)
        if direction == "short":
            passed = px < neck * (1.0 - beta)
        else:
            passed = px > neck * (1.0 + beta)

        if not passed:
            continue

        if min_volume is not None and _volume(df, i) < min_volume:
            continue

        return i
    return None


def _event(
    df: pd.DataFrame,
    symbol: str,
    pattern: str,
    label: int,
    reason: str,
    anchor_idx: int,
    breakout_idx: int | None,
    direction: str,
    neckline: float,
    height: float,
    stop_price: float,
) -> PatternEvent:
    price_ref = _close(df, breakout_idx if breakout_idx is not None else anchor_idx)
    target = price_ref + height if direction == "long" else price_ref - height

    return PatternEvent(
        symbol=symbol,
        pattern=pattern,
        label=label,
        reason=reason,
        anchor_idx=anchor_idx,
        breakout_idx=breakout_idx,
        ts_event=df.iloc[anchor_idx]["ts_event"],
        direction=direction,
        neckline=float(neckline),
        pattern_height=float(max(height, 1.0e-8)),
        stop_price=float(stop_price),
        target_price=float(target),
    )


def _detect_head_shoulders_like(
    df: pd.DataFrame,
    extrema: list[Extremum],
    cfg: dict[str, Any],
    inverse: bool,
) -> list[PatternEvent]:
    key = "inverse_head_shoulders" if inverse else "head_shoulders"
    geom = cfg["labeling"][key]["geometry"]
    vol_cfg = cfg["labeling"][key]["volume_rules"]
    conf = cfg["labeling"]["confirmation"]

    tol_shoulder = float(geom.get("shoulder_symmetry_tolerance_pct", 0.015))
    tol_neck = float(geom.get("neckline_point_symmetry_tolerance_pct", 0.015))
    min_sep = int(geom.get("min_peak_separation_bars", geom.get("min_trough_separation_bars", 5)))
    max_sep = int(geom.get("max_peak_separation_bars", geom.get("max_trough_separation_bars", 40)))
    max_slope = float(geom.get("neckline_max_slope", 0.15))

    beta = float(conf.get("breakout_confirmation_pct", 0.04))
    horizon = int(conf.get("confirm_break_within_bars", 8))
    vol_avg_bars = int(conf.get("breakout_volume_average_bars", 20))

    seq = ["min", "max", "min", "max", "min"] if inverse else ["max", "min", "max", "min", "max"]

    events: list[PatternEvent] = []
    for i in range(len(extrema) - 4):
        e = extrema[i : i + 5]
        if [x.kind for x in e] != seq:
            continue

        e1, e2, e3, e4, e5 = e

        span_13 = e3.idx - e1.idx
        span_35 = e5.idx - e3.idx
        if span_13 < min_sep or span_35 < min_sep or span_13 > max_sep or span_35 > max_sep:
            continue

        if inverse:
            head_ok = e3.price < e1.price and e3.price < e5.price
            shoulder_ref = max(e1.price, e5.price)
            stop_level = min(e1.price, e3.price, e5.price)
            direction = "long"
        else:
            head_ok = e3.price > e1.price and e3.price > e5.price
            shoulder_ref = min(e1.price, e5.price)
            stop_level = max(e1.price, e3.price, e5.price)
            direction = "short"

        shoulder_ok = _relative_diff(e1.price, e5.price) <= tol_shoulder
        neckline_sym_ok = _relative_diff(e2.price, e4.price) <= tol_neck

        slope = (e4.price - e2.price) / max(e4.idx - e2.idx, 1)
        slope_ok = abs(slope) <= max_slope

        strict_geom = head_ok and shoulder_ok and neckline_sym_ok and slope_ok
        soft_geom = head_ok and _relative_diff(e1.price, e5.price) <= (tol_shoulder * 1.8)

        if not strict_geom:
            if soft_geom:
                events.append(
                    _event(
                        df=df,
                        symbol=str(df.iloc[0]["symbol"]),
                        pattern=key,
                        label=0,
                        reason="near_miss",
                        anchor_idx=e5.idx,
                        breakout_idx=None,
                        direction=direction,
                        neckline=e4.price,
                        height=abs(e3.price - shoulder_ref),
                        stop_price=stop_level,
                    )
                )
            continue

        vol_ok = True
        if vol_cfg.get("required", True):
            if inverse:
                vol_ok = _volume(df, e2.idx) > _volume(df, e4.idx)
            else:
                vol_ok = _volume(df, e1.idx) > _volume(df, e3.idx) > _volume(df, e5.idx)
        if not vol_ok:
            continue

        vol_multiplier = float(vol_cfg.get("breakout_volume_min_multiplier", 1.2))
        vol_start = max(0, e5.idx - vol_avg_bars + 1)
        rolling_avg = float(df.iloc[vol_start : e5.idx + 1]["volume"].mean())
        min_break_vol = rolling_avg * vol_multiplier if vol_cfg.get("required", True) else None

        neckline_fn = lambda idx: e2.price + slope * (idx - e2.idx)
        breakout_idx = _find_breakout(
            df=df,
            start_idx=e5.idx,
            max_horizon=horizon,
            neckline_at_idx=neckline_fn,
            beta=beta,
            direction=direction,
            min_volume=min_break_vol,
        )

        height = abs(e3.price - neckline_fn(e3.idx))
        if breakout_idx is not None:
            anchor = breakout_idx
            events.append(
                _event(
                    df=df,
                    symbol=str(df.iloc[0]["symbol"]),
                    pattern=key,
                    label=1,
                    reason="confirmed_breakout",
                    anchor_idx=anchor,
                    breakout_idx=breakout_idx,
                    direction=direction,
                    neckline=neckline_fn(breakout_idx),
                    height=height,
                    stop_price=stop_level,
                )
            )
        else:
            events.append(
                _event(
                    df=df,
                    symbol=str(df.iloc[0]["symbol"]),
                    pattern=key,
                    label=0,
                    reason="failed_breakout",
                    anchor_idx=e5.idx,
                    breakout_idx=None,
                    direction=direction,
                    neckline=neckline_fn(e5.idx),
                    height=height,
                    stop_price=stop_level,
                )
            )

    return events


def _detect_double_like(
    df: pd.DataFrame,
    extrema: list[Extremum],
    cfg: dict[str, Any],
    bottom: bool,
) -> list[PatternEvent]:
    key = "double_bottom" if bottom else "double_top"
    geom = cfg["labeling"][key]["geometry"]
    vol_cfg = cfg["labeling"][key]["volume_rules"]
    conf = cfg["labeling"]["confirmation"]

    tol = float(geom.get("trough_tolerance_pct" if bottom else "peak_tolerance_pct", 0.025))
    min_retrace = float(geom.get("min_bounce_pct" if bottom else "min_pullback_pct", 0.05))
    prior_trend = float(geom.get("prior_trend_pct", 0.15))
    min_sep = int(geom.get("min_separation_bars", 5))
    max_sep = int(geom.get("max_separation_bars", 50))

    beta = float(conf.get("breakout_confirmation_pct", 0.04))
    horizon = int(conf.get("confirm_break_within_bars", 8))
    vol_avg_bars = int(conf.get("breakout_volume_average_bars", 20))

    seq = ["min", "max", "min"] if bottom else ["max", "min", "max"]
    direction = "long" if bottom else "short"

    events: list[PatternEvent] = []
    for i in range(len(extrema) - 2):
        e1, e2, e3 = extrema[i : i + 3]
        if [e1.kind, e2.kind, e3.kind] != seq:
            continue

        sep = e3.idx - e1.idx
        if sep < min_sep or sep > max_sep:
            continue

        pair_ok = _relative_diff(e1.price, e3.price) <= tol
        anchor_level = (e1.price + e3.price) / 2.0

        if bottom:
            retrace = (e2.price - anchor_level) / max(anchor_level, 1.0e-12)
            prior_start_idx = max(0, e1.idx - max(20, min_sep))
            p0 = _close(df, prior_start_idx)
            trend_ok = ((p0 - e1.price) / max(abs(p0), 1.0e-12)) >= prior_trend
            stop_level = min(e1.price, e3.price)
        else:
            retrace = (anchor_level - e2.price) / max(anchor_level, 1.0e-12)
            prior_start_idx = max(0, e1.idx - max(20, min_sep))
            p0 = _close(df, prior_start_idx)
            trend_ok = ((e1.price - p0) / max(abs(p0), 1.0e-12)) >= prior_trend
            stop_level = max(e1.price, e3.price)

        retrace_ok = retrace >= min_retrace
        strict_geom = pair_ok and retrace_ok and trend_ok
        soft_geom = _relative_diff(e1.price, e3.price) <= (tol * 1.8)

        if not strict_geom:
            if soft_geom:
                events.append(
                    _event(
                        df=df,
                        symbol=str(df.iloc[0]["symbol"]),
                        pattern=key,
                        label=0,
                        reason="near_miss",
                        anchor_idx=e3.idx,
                        breakout_idx=None,
                        direction=direction,
                        neckline=e2.price,
                        height=abs(anchor_level - e2.price),
                        stop_price=stop_level,
                    )
                )
            continue

        vol_ok = True
        if vol_cfg.get("required", True):
            vol_ok = _volume(df, e1.idx) > _volume(df, e3.idx)

        if not vol_ok:
            continue

        vol_multiplier = float(vol_cfg.get("breakout_volume_min_multiplier", 1.2))
        vol_start = max(0, e3.idx - vol_avg_bars + 1)
        rolling_avg = float(df.iloc[vol_start : e3.idx + 1]["volume"].mean())
        min_break_vol = rolling_avg * vol_multiplier if vol_cfg.get("required", True) else None

        neckline_price = e2.price
        neckline_fn = lambda _: neckline_price
        breakout_idx = _find_breakout(
            df=df,
            start_idx=e3.idx,
            max_horizon=horizon,
            neckline_at_idx=neckline_fn,
            beta=beta,
            direction=direction,
            min_volume=min_break_vol,
        )

        height = abs(anchor_level - neckline_price)
        if breakout_idx is not None:
            events.append(
                _event(
                    df=df,
                    symbol=str(df.iloc[0]["symbol"]),
                    pattern=key,
                    label=1,
                    reason="confirmed_breakout",
                    anchor_idx=breakout_idx,
                    breakout_idx=breakout_idx,
                    direction=direction,
                    neckline=neckline_price,
                    height=height,
                    stop_price=stop_level,
                )
            )
        else:
            events.append(
                _event(
                    df=df,
                    symbol=str(df.iloc[0]["symbol"]),
                    pattern=key,
                    label=0,
                    reason="failed_breakout",
                    anchor_idx=e3.idx,
                    breakout_idx=None,
                    direction=direction,
                    neckline=neckline_price,
                    height=height,
                    stop_price=stop_level,
                )
            )

    return events


def _add_partial_negatives(
    df: pd.DataFrame,
    extrema: list[Extremum],
    pattern: str,
    direction: str,
    existing_anchor_idx: set[int],
    limit: int,
) -> list[PatternEvent]:
    out: list[PatternEvent] = []
    symbol = str(df.iloc[0]["symbol"])
    for ex in extrema:
        if ex.idx in existing_anchor_idx:
            continue
        if ex.idx <= 0 or ex.idx >= len(df) - 1:
            continue

        out.append(
            _event(
                df=df,
                symbol=symbol,
                pattern=pattern,
                label=0,
                reason="partial",
                anchor_idx=ex.idx,
                breakout_idx=None,
                direction=direction,
                neckline=ex.price,
                height=float(_atr(df, ex.idx)),
                stop_price=float(df.iloc[ex.idx]["close"]),
            )
        )
        if len(out) >= limit:
            break
    return out


def detect_pattern_events(df: pd.DataFrame, pattern: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Detect pattern events and return label table with positive and hard-negative rows."""
    if df.empty:
        return pd.DataFrame()

    symbol_df = df.sort_values("ts_event").reset_index(drop=True).copy()
    working_timezone = cfg.get("data_source", {}).get("timestamp", {}).get(
        "convert_to_timezone", "America/New_York"
    )
    symbol_df["ts_event"] = pd.to_datetime(symbol_df["ts_event"], errors="coerce", utc=True).dt.tz_convert(
        working_timezone
    )
    symbol_df = symbol_df.dropna(subset=["ts_event"]).copy()

    extrema = detect_extrema_from_close(symbol_df, cfg)
    if len(extrema) == 0:
        return pd.DataFrame()

    if pattern == "head_shoulders":
        events = _detect_head_shoulders_like(symbol_df, extrema, cfg, inverse=False)
        direction = "short"
    elif pattern == "inverse_head_shoulders":
        events = _detect_head_shoulders_like(symbol_df, extrema, cfg, inverse=True)
        direction = "long"
    elif pattern == "double_top":
        events = _detect_double_like(symbol_df, extrema, cfg, bottom=False)
        direction = "short"
    elif pattern == "double_bottom":
        events = _detect_double_like(symbol_df, extrema, cfg, bottom=True)
        direction = "long"
    else:
        raise ValueError(f"Unsupported pattern: {pattern}")

    # Add partial negatives when requested.
    hard_cfg = cfg["labeling"].get("labeling_policy", {}).get("hard_negative_sampling", {})
    if hard_cfg.get("enabled", True) and hard_cfg.get("include_partial_patterns", True):
        existing = {e.anchor_idx for e in events}
        positives = sum(1 for e in events if e.label == 1)
        target_ratio = hard_cfg.get("target_positive_to_negative_ratio", "1:3")
        try:
            _, neg = target_ratio.split(":")
            neg_factor = max(1, int(neg))
        except Exception:
            neg_factor = 3

        desired_negatives = max(neg_factor * max(positives, 1), 8)
        current_negatives = sum(1 for e in events if e.label == 0)
        add_limit = max(0, desired_negatives - current_negatives)

        if add_limit > 0:
            events.extend(
                _add_partial_negatives(
                    symbol_df,
                    extrema,
                    pattern=pattern,
                    direction=direction,
                    existing_anchor_idx=existing,
                    limit=add_limit,
                )
            )

    if not events:
        return pd.DataFrame()

    # Deduplicate on anchor index, prioritizing confirmed breakouts over negatives.
    ordered = sorted(events, key=lambda e: (e.anchor_idx, -e.label, e.reason))
    dedup: dict[int, PatternEvent] = {}
    for e in ordered:
        current = dedup.get(e.anchor_idx)
        if current is None or e.label > current.label:
            dedup[e.anchor_idx] = e

    records = [asdict(x) for x in dedup.values()]
    out = pd.DataFrame(records).sort_values("anchor_idx").reset_index(drop=True)
    return out


def detect_all_patterns(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    patterns = cfg["labeling"].get(
        "allowed_patterns",
        ["head_shoulders", "inverse_head_shoulders", "double_top", "double_bottom"],
    )
    out: dict[str, pd.DataFrame] = {}
    for pattern in patterns:
        out[pattern] = detect_pattern_events(df, pattern, cfg)
    return out
