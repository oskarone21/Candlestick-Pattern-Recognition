from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from candlestick.domain import (
    CFG_CONFIRMATION,
    CFG_GEOMETRY,
    CFG_LABELING,
    CFG_VOLUME_RULES,
    COLUMN_ANCHOR_IDX,
    COLUMN_CLOSE,
    COLUMN_HIGH,
    COLUMN_LOW,
    COLUMN_SYMBOL,
    COLUMN_TS_EVENT,
    COLUMN_VOLUME,
    DEFAULT_BREAKOUT_CONFIRMATION_PCT,
    DEFAULT_BREAKOUT_VOLUME_AVERAGE_BARS,
    DEFAULT_BREAKOUT_VOLUME_MIN_MULTIPLIER,
    DEFAULT_CONFIRM_BREAK_WITHIN_BARS,
    DEFAULT_POS_NEG_RATIO,
    DEFAULT_SHOULDER_TOLERANCE_PCT,
    EPSILON_SAFE_DIVIDE,
    ExtremumKind,
    PatternEventReason,
    PatternName,
    TradeDirection,
)
from candlestick.labeling.extrema import Extremum, detect_extrema_from_close
from candlestick.project_utils import allowed_patterns_from_cfg, working_timezone_from_cfg


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class PatternSpec:
    name: PatternName
    direction: TradeDirection
    extrema_sequence: tuple[str, ...]


PATTERN_SPECS: dict[PatternName, PatternSpec] = {
    PatternName.HEAD_SHOULDERS: PatternSpec(
        name=PatternName.HEAD_SHOULDERS,
        direction=TradeDirection.SHORT,
        extrema_sequence=("max", "min", "max", "min", "max"),
    ),
    PatternName.INVERSE_HEAD_SHOULDERS: PatternSpec(
        name=PatternName.INVERSE_HEAD_SHOULDERS,
        direction=TradeDirection.LONG,
        extrema_sequence=("min", "max", "min", "max", "min"),
    ),
    PatternName.DOUBLE_TOP: PatternSpec(
        name=PatternName.DOUBLE_TOP,
        direction=TradeDirection.SHORT,
        extrema_sequence=("max", "min", "max"),
    ),
    PatternName.DOUBLE_BOTTOM: PatternSpec(
        name=PatternName.DOUBLE_BOTTOM,
        direction=TradeDirection.LONG,
        extrema_sequence=("min", "max", "min"),
    ),
}


def _relative_diff(a: float, b: float) -> float:
    denom = max((abs(a) + abs(b)) / 2.0, EPSILON_SAFE_DIVIDE)
    return abs(a - b) / denom


def _close(df: pd.DataFrame, idx: int) -> float:
    return float(df.iloc[idx][COLUMN_CLOSE])


def _volume(df: pd.DataFrame, idx: int) -> float:
    return float(df.iloc[idx][COLUMN_VOLUME])


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
    value = float(tr.mean()) if not tr.empty else float(df.iloc[idx][COLUMN_HIGH] - df.iloc[idx][COLUMN_LOW])
    return max(value, EPSILON_SAFE_DIVIDE)


def _find_breakout(
    df: pd.DataFrame,
    start_idx: int,
    max_horizon: int,
    neckline_at_idx,
    beta: float,
    direction: TradeDirection,
    min_volume: float | None,
) -> int | None:
    end_idx = min(len(df) - 1, start_idx + max_horizon)
    for idx in range(start_idx + 1, end_idx + 1):
        price = _close(df, idx)
        neckline = neckline_at_idx(idx)
        if direction is TradeDirection.SHORT:
            passed = price < neckline * (1.0 - beta)
        else:
            passed = price > neckline * (1.0 + beta)

        if not passed:
            continue
        if min_volume is not None and _volume(df, idx) < min_volume:
            continue
        return idx
    return None


def _event(
    df: pd.DataFrame,
    symbol: str,
    pattern: PatternName,
    label: int,
    reason: PatternEventReason,
    anchor_idx: int,
    breakout_idx: int | None,
    direction: TradeDirection,
    neckline: float,
    height: float,
    stop_price: float,
) -> PatternEvent:
    price_ref = _close(df, breakout_idx if breakout_idx is not None else anchor_idx)
    target = price_ref + height if direction is TradeDirection.LONG else price_ref - height
    return PatternEvent(
        symbol=symbol,
        pattern=pattern.value,
        label=label,
        reason=reason.value,
        anchor_idx=anchor_idx,
        breakout_idx=breakout_idx,
        ts_event=df.iloc[anchor_idx][COLUMN_TS_EVENT],
        direction=direction.value,
        neckline=float(neckline),
        pattern_height=float(max(height, EPSILON_SAFE_DIVIDE)),
        stop_price=float(stop_price),
        target_price=float(target),
    )


def _detect_head_shoulders_like(
    df: pd.DataFrame,
    extrema: list[Extremum],
    cfg: dict[str, Any],
    spec: PatternSpec,
) -> list[PatternEvent]:
    geom = cfg["labeling"][spec.name.value]["geometry"]
    vol_cfg = cfg["labeling"][spec.name.value]["volume_rules"]
    conf = cfg["labeling"]["confirmation"]

    tol_shoulder = float(geom.get("shoulder_symmetry_tolerance_pct", 0.015))
    tol_neck = float(geom.get("neckline_point_symmetry_tolerance_pct", 0.015))
    min_sep = int(geom.get("min_peak_separation_bars", geom.get("min_trough_separation_bars", 5)))
    max_sep = int(geom.get("max_peak_separation_bars", geom.get("max_trough_separation_bars", 40)))
    max_slope = float(geom.get("neckline_max_slope", 0.15))

    beta = float(conf.get("breakout_confirmation_pct", 0.04))
    horizon = int(conf.get("confirm_break_within_bars", 8))
    vol_avg_bars = int(conf.get("breakout_volume_average_bars", 20))

    symbol = str(df.iloc[0][COLUMN_SYMBOL])
    is_inverse = spec.direction is TradeDirection.LONG
    events: list[PatternEvent] = []

    for idx in range(len(extrema) - 4):
        e1, e2, e3, e4, e5 = extrema[idx : idx + 5]
        if tuple(ext.kind for ext in (e1, e2, e3, e4, e5)) != spec.extrema_sequence:
            continue

        span_13 = e3.idx - e1.idx
        span_35 = e5.idx - e3.idx
        if span_13 < min_sep or span_35 < min_sep or span_13 > max_sep or span_35 > max_sep:
            continue

        if is_inverse:
            head_ok = e3.price < e1.price and e3.price < e5.price
            shoulder_ref = max(e1.price, e5.price)
            stop_level = min(e1.price, e3.price, e5.price)
        else:
            head_ok = e3.price > e1.price and e3.price > e5.price
            shoulder_ref = min(e1.price, e5.price)
            stop_level = max(e1.price, e3.price, e5.price)

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
                        symbol=symbol,
                        pattern=spec.name,
                        label=0,
                        reason=PatternEventReason.NEAR_MISS,
                        anchor_idx=e5.idx,
                        breakout_idx=None,
                        direction=spec.direction,
                        neckline=e4.price,
                        height=abs(e3.price - shoulder_ref),
                        stop_price=stop_level,
                    )
                )
            continue

        vol_ok = True
        if vol_cfg.get("required", True):
            if is_inverse:
                vol_ok = _volume(df, e2.idx) > _volume(df, e4.idx)
            else:
                vol_ok = _volume(df, e1.idx) > _volume(df, e3.idx) > _volume(df, e5.idx)
        if not vol_ok:
            continue

        vol_multiplier = float(vol_cfg.get("breakout_volume_min_multiplier", 1.2))
        vol_start = max(0, e5.idx - vol_avg_bars + 1)
        rolling_avg = float(df.iloc[vol_start : e5.idx + 1][COLUMN_VOLUME].mean())
        min_break_vol = rolling_avg * vol_multiplier if vol_cfg.get("required", True) else None

        neckline_fn = lambda row_idx: e2.price + slope * (row_idx - e2.idx)
        breakout_idx = _find_breakout(
            df=df,
            start_idx=e5.idx,
            max_horizon=horizon,
            neckline_at_idx=neckline_fn,
            beta=beta,
            direction=spec.direction,
            min_volume=min_break_vol,
        )

        height = abs(e3.price - neckline_fn(e3.idx))
        if breakout_idx is not None:
            events.append(
                _event(
                    df=df,
                    symbol=symbol,
                    pattern=spec.name,
                    label=1,
                    reason=PatternEventReason.CONFIRMED_BREAKOUT,
                    anchor_idx=breakout_idx,
                    breakout_idx=breakout_idx,
                    direction=spec.direction,
                    neckline=neckline_fn(breakout_idx),
                    height=height,
                    stop_price=stop_level,
                )
            )
            continue

        events.append(
            _event(
                df=df,
                symbol=symbol,
                pattern=spec.name,
                label=0,
                reason=PatternEventReason.FAILED_BREAKOUT,
                anchor_idx=e5.idx,
                breakout_idx=None,
                direction=spec.direction,
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
    spec: PatternSpec,
) -> list[PatternEvent]:
    geom = cfg["labeling"][spec.name.value]["geometry"]
    vol_cfg = cfg["labeling"][spec.name.value]["volume_rules"]
    conf = cfg["labeling"]["confirmation"]

    is_bottom = spec.direction is TradeDirection.LONG
    tol_key = "trough_tolerance_pct" if is_bottom else "peak_tolerance_pct"
    retrace_key = "min_bounce_pct" if is_bottom else "min_pullback_pct"
    tol = float(geom.get(tol_key, 0.025))
    min_retrace = float(geom.get(retrace_key, 0.05))
    prior_trend = float(geom.get("prior_trend_pct", 0.15))
    min_sep = int(geom.get("min_separation_bars", 5))
    max_sep = int(geom.get("max_separation_bars", 50))

    beta = float(conf.get("breakout_confirmation_pct", 0.04))
    horizon = int(conf.get("confirm_break_within_bars", 8))
    vol_avg_bars = int(conf.get("breakout_volume_average_bars", 20))

    symbol = str(df.iloc[0][COLUMN_SYMBOL])
    events: list[PatternEvent] = []

    for idx in range(len(extrema) - 2):
        e1, e2, e3 = extrema[idx : idx + 3]
        if tuple(ext.kind for ext in (e1, e2, e3)) != spec.extrema_sequence:
            continue

        separation = e3.idx - e1.idx
        if separation < min_sep or separation > max_sep:
            continue

        pair_ok = _relative_diff(e1.price, e3.price) <= tol
        anchor_level = (e1.price + e3.price) / 2.0
        prior_start_idx = max(0, e1.idx - max(20, min_sep))
        prior_price = _close(df, prior_start_idx)

        if is_bottom:
            retrace = (e2.price - anchor_level) / max(anchor_level, 1.0e-12)
            trend_ok = ((prior_price - e1.price) / max(abs(prior_price), 1.0e-12)) >= prior_trend
            stop_level = min(e1.price, e3.price)
        else:
            retrace = (anchor_level - e2.price) / max(anchor_level, 1.0e-12)
            trend_ok = ((e1.price - prior_price) / max(abs(prior_price), 1.0e-12)) >= prior_trend
            stop_level = max(e1.price, e3.price)

        retrace_ok = retrace >= min_retrace
        strict_geom = pair_ok and retrace_ok and trend_ok
        soft_geom = _relative_diff(e1.price, e3.price) <= (tol * 1.8)

        if not strict_geom:
            if soft_geom:
                events.append(
                    _event(
                        df=df,
                        symbol=symbol,
                        pattern=spec.name,
                        label=0,
                        reason=PatternEventReason.NEAR_MISS,
                        anchor_idx=e3.idx,
                        breakout_idx=None,
                        direction=spec.direction,
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
        rolling_avg = float(df.iloc[vol_start : e3.idx + 1][COLUMN_VOLUME].mean())
        min_break_vol = rolling_avg * vol_multiplier if vol_cfg.get("required", True) else None

        neckline_price = e2.price
        breakout_idx = _find_breakout(
            df=df,
            start_idx=e3.idx,
            max_horizon=horizon,
            neckline_at_idx=lambda _: neckline_price,
            beta=beta,
            direction=spec.direction,
            min_volume=min_break_vol,
        )

        height = abs(anchor_level - neckline_price)
        if breakout_idx is not None:
            events.append(
                _event(
                    df=df,
                    symbol=symbol,
                    pattern=spec.name,
                    label=1,
                    reason=PatternEventReason.CONFIRMED_BREAKOUT,
                    anchor_idx=breakout_idx,
                    breakout_idx=breakout_idx,
                    direction=spec.direction,
                    neckline=neckline_price,
                    height=height,
                    stop_price=stop_level,
                )
            )
            continue

        events.append(
            _event(
                df=df,
                symbol=symbol,
                pattern=spec.name,
                label=0,
                reason=PatternEventReason.FAILED_BREAKOUT,
                anchor_idx=e3.idx,
                breakout_idx=None,
                direction=spec.direction,
                neckline=neckline_price,
                height=height,
                stop_price=stop_level,
            )
        )

    return events


def _add_partial_negatives(
    df: pd.DataFrame,
    extrema: list[Extremum],
    pattern: PatternName,
    direction: TradeDirection,
    existing_anchor_idx: set[int],
    limit: int,
) -> list[PatternEvent]:
    out: list[PatternEvent] = []
    symbol = str(df.iloc[0][COLUMN_SYMBOL])

    for extremum in extrema:
        if extremum.idx in existing_anchor_idx:
            continue
        if extremum.idx <= 0 or extremum.idx >= len(df) - 1:
            continue

        out.append(
            _event(
                df=df,
                symbol=symbol,
                pattern=pattern,
                label=0,
                reason=PatternEventReason.PARTIAL,
                anchor_idx=extremum.idx,
                breakout_idx=None,
                direction=direction,
                neckline=extremum.price,
                height=float(_atr(df, extremum.idx)),
                stop_price=float(df.iloc[extremum.idx][COLUMN_CLOSE]),
            )
        )
        if len(out) >= limit:
            break

    return out


def _detect_events_for_pattern(
    df: pd.DataFrame,
    extrema: list[Extremum],
    cfg: dict[str, Any],
    pattern: PatternName,
) -> list[PatternEvent]:
    spec = PATTERN_SPECS[pattern]
    if pattern in {PatternName.HEAD_SHOULDERS, PatternName.INVERSE_HEAD_SHOULDERS}:
        return _detect_head_shoulders_like(df, extrema, cfg, spec)
    return _detect_double_like(df, extrema, cfg, spec)


def detect_pattern_events(df: pd.DataFrame, pattern: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Return the event table for one pattern."""
    if df.empty:
        return pd.DataFrame()

    try:
        pattern_name = PatternName(pattern)
    except ValueError as exc:
        raise ValueError(f"Unsupported pattern: {pattern}") from exc

    symbol_df = df.sort_values(COLUMN_TS_EVENT).reset_index(drop=True).copy()
    symbol_df[COLUMN_TS_EVENT] = pd.to_datetime(
        symbol_df[COLUMN_TS_EVENT],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(working_timezone_from_cfg(cfg))
    symbol_df = symbol_df.dropna(subset=[COLUMN_TS_EVENT]).copy()

    extrema = detect_extrema_from_close(symbol_df, cfg)
    if not extrema:
        return pd.DataFrame()

    events = _detect_events_for_pattern(symbol_df, extrema, cfg, pattern_name)

    hard_cfg = cfg["labeling"].get("labeling_policy", {}).get("hard_negative_sampling", {})
    if hard_cfg.get("enabled", True) and hard_cfg.get("include_partial_patterns", True):
        existing = {event.anchor_idx for event in events}
        positives = sum(1 for event in events if event.label == 1)
        target_ratio = str(hard_cfg.get("target_positive_to_negative_ratio", "1:3"))
        try:
            _, negative_ratio = target_ratio.split(":")
            neg_factor = max(1, int(negative_ratio))
        except Exception:
            neg_factor = 3

        desired_negatives = max(neg_factor * max(positives, 1), 8)
        current_negatives = sum(1 for event in events if event.label == 0)
        add_limit = max(0, desired_negatives - current_negatives)

        if add_limit > 0:
            events.extend(
                _add_partial_negatives(
                    symbol_df,
                    extrema,
                    pattern=pattern_name,
                    direction=PATTERN_SPECS[pattern_name].direction,
                    existing_anchor_idx=existing,
                    limit=add_limit,
                )
            )

    if not events:
        return pd.DataFrame()

    ordered = sorted(events, key=lambda event: (event.anchor_idx, -event.label, event.reason))
    dedup: dict[int, PatternEvent] = {}
    for event in ordered:
        current = dedup.get(event.anchor_idx)
        if current is None or event.label > current.label:
            dedup[event.anchor_idx] = event

    records = [asdict(event) for event in dedup.values()]
    return pd.DataFrame(records).sort_values("anchor_idx").reset_index(drop=True)


def detect_all_patterns(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    detected: dict[str, pd.DataFrame] = {}
    for pattern in allowed_patterns_from_cfg(cfg):
        detected[pattern] = detect_pattern_events(df, pattern, cfg)
    return detected
