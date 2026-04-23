from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from chart_patterns.domain import (
    COLUMN_CLOSE,
    COLUMN_HIGH,
    COLUMN_LOW,
    COLUMN_OPEN,
    COLUMN_SYMBOL,
    COLUMN_TS_EVENT,
    COLUMN_VOLUME,
    DEFAULT_SESSION_END,
    DEFAULT_SESSION_START,
)
from chart_patterns.project_utils import timeframe_to_minutes


@dataclass
class CleanResult:
    frame: pd.DataFrame
    dropped_rows: int
    invalid_ohlc_rows: int


@dataclass
class DedupResult:
    frame: pd.DataFrame
    dropped_rows: int


def drop_exact_duplicate_ohlcv(df: pd.DataFrame) -> DedupResult:
    """Drop exact duplicate minute bars while preserving canonical sort order."""
    before = len(df)
    out = df.drop_duplicates(
        subset=[COLUMN_SYMBOL, COLUMN_TS_EVENT, COLUMN_OPEN, COLUMN_HIGH, COLUMN_LOW, COLUMN_CLOSE, COLUMN_VOLUME],
        keep="last",
    )
    out = out.sort_values([COLUMN_SYMBOL, COLUMN_TS_EVENT]).reset_index(drop=True)
    return DedupResult(frame=out, dropped_rows=before - len(out))


def clean_ohlcv(df: pd.DataFrame, drop_duplicates: bool = True) -> CleanResult:
    """Clean intraday OHLCV with strict numeric and consistency checks."""
    out = df.copy()

    for col in [COLUMN_OPEN, COLUMN_HIGH, COLUMN_LOW, COLUMN_CLOSE, COLUMN_VOLUME]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    before = len(out)
    out = out.dropna(subset=[COLUMN_SYMBOL, COLUMN_TS_EVENT, COLUMN_OPEN, COLUMN_HIGH, COLUMN_LOW, COLUMN_CLOSE, COLUMN_VOLUME])

    if drop_duplicates:
        out = out.drop_duplicates(subset=[COLUMN_SYMBOL, COLUMN_TS_EVENT], keep="last")

    invalid = (
        (out[COLUMN_HIGH] < out[[COLUMN_OPEN, COLUMN_CLOSE]].max(axis=1))
        | (out[COLUMN_LOW] > out[[COLUMN_OPEN, COLUMN_CLOSE]].min(axis=1))
        | (out[COLUMN_HIGH] < out[COLUMN_LOW])
        | (out[COLUMN_VOLUME] < 0)
    )
    invalid_count = int(invalid.sum())
    out = out.loc[~invalid].copy()

    out = out.sort_values([COLUMN_SYMBOL, COLUMN_TS_EVENT]).reset_index(drop=True)
    dropped = before - len(out)
    return CleanResult(frame=out, dropped_rows=dropped, invalid_ohlc_rows=invalid_count)


def filter_regular_session(
    df: pd.DataFrame,
    start: str = DEFAULT_SESSION_START,
    end: str = DEFAULT_SESSION_END,
    regular_session_only: bool = True,
) -> pd.DataFrame:
    """Filter to weekdays and optional regular US session interval."""
    out = df.copy()

    out = out[out[COLUMN_TS_EVENT].dt.dayofweek < 5]

    if not regular_session_only:
        return out

    pieces: list[pd.DataFrame] = []
    for symbol, grp in out.groupby(COLUMN_SYMBOL, sort=False):
        indexed = grp.set_index(COLUMN_TS_EVENT, drop=False)
        sliced = indexed.between_time(start, end, inclusive="left").copy()
        sliced[COLUMN_SYMBOL] = symbol
        pieces.append(sliced.reset_index(drop=True))

    if not pieces:
        return out.iloc[0:0].copy()
    return pd.concat(pieces, axis=0, ignore_index=True)


def resample_ohlcv(
    df: pd.DataFrame,
    base_timeframe: str,
    target_timeframe: str,
    drop_incomplete_bars: bool = True,
    label: str = "right",
    closed: str = "left",
) -> pd.DataFrame:
    """Resample OHLCV grouped by symbol with standard aggregation rules."""
    if base_timeframe == target_timeframe:
        return df.copy()

    base_minutes = timeframe_to_minutes(base_timeframe)
    target_minutes = timeframe_to_minutes(target_timeframe)
    expected_rows = target_minutes // base_minutes
    if expected_rows <= 0:
        raise ValueError("target_timeframe must be larger than base_timeframe for resampling.")

    out_frames: list[pd.DataFrame] = []
    for symbol, grp in df.groupby(COLUMN_SYMBOL, sort=False):
        indexed = grp.set_index(COLUMN_TS_EVENT)
        ohlcv = indexed.resample(target_timeframe, label=label, closed=closed).agg(
            {
                COLUMN_OPEN: "first",
                COLUMN_HIGH: "max",
                COLUMN_LOW: "min",
                COLUMN_CLOSE: "last",
                COLUMN_VOLUME: "sum",
            }
        )
        counts = indexed[COLUMN_CLOSE].resample(target_timeframe, label=label, closed=closed).count()

        if drop_incomplete_bars:
            ohlcv = ohlcv[counts == expected_rows]

        ohlcv = ohlcv.dropna(subset=[COLUMN_OPEN, COLUMN_HIGH, COLUMN_LOW, COLUMN_CLOSE])
        ohlcv[COLUMN_SYMBOL] = symbol
        out_frames.append(ohlcv.reset_index())

    if not out_frames:
        return df.iloc[0:0].copy()

    out = pd.concat(out_frames, axis=0, ignore_index=True)
    out = out.sort_values([COLUMN_SYMBOL, COLUMN_TS_EVENT]).reset_index(drop=True)
    return out


def prepare_intraday_15m(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Apply session filtering + resampling using config values."""
    data_cfg = cfg["data_source"]
    session_cfg = data_cfg.get("session", {})
    res_cfg = cfg["resampling"]

    session_filtered = filter_regular_session(
        df,
        start=session_cfg.get("start", DEFAULT_SESSION_START),
        end=session_cfg.get("end", DEFAULT_SESSION_END),
        regular_session_only=session_cfg.get("regular_session_only", True),
    )

    out = resample_ohlcv(
        session_filtered,
        base_timeframe=res_cfg.get("base_timeframe", "1min"),
        target_timeframe=res_cfg.get("target_timeframe", "15min"),
        drop_incomplete_bars=res_cfg.get("drop_incomplete_bars", True),
    )
    return out
