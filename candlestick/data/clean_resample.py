from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class CleanResult:
    frame: pd.DataFrame
    dropped_rows: int
    invalid_ohlc_rows: int


@dataclass
class DedupResult:
    frame: pd.DataFrame
    dropped_rows: int


def _timeframe_to_minutes(rule: str) -> int:
    norm = rule.strip().lower()
    if norm.endswith("min"):
        return int(norm[:-3])
    if norm.endswith("m"):
        return int(norm[:-1])
    if norm.endswith("h"):
        return int(norm[:-1]) * 60
    raise ValueError(f"Unsupported timeframe rule: {rule}")


def drop_exact_duplicate_ohlcv(df: pd.DataFrame) -> DedupResult:
    """Drop exact duplicate minute bars while preserving canonical sort order."""
    before = len(df)
    out = df.drop_duplicates(
        subset=["symbol", "ts_event", "open", "high", "low", "close", "volume"],
        keep="last",
    )
    out = out.sort_values(["symbol", "ts_event"]).reset_index(drop=True)
    return DedupResult(frame=out, dropped_rows=before - len(out))


def clean_ohlcv(df: pd.DataFrame, drop_duplicates: bool = True) -> CleanResult:
    """Clean intraday OHLCV with strict numeric and consistency checks."""
    out = df.copy()

    # Standardize dtype.
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    before = len(out)
    out = out.dropna(subset=["symbol", "ts_event", "open", "high", "low", "close", "volume"])

    if drop_duplicates:
        out = out.drop_duplicates(subset=["symbol", "ts_event"], keep="last")

    # Remove impossible bars.
    invalid = (
        (out["high"] < out[["open", "close"]].max(axis=1))
        | (out["low"] > out[["open", "close"]].min(axis=1))
        | (out["high"] < out["low"])
        | (out["volume"] < 0)
    )
    invalid_count = int(invalid.sum())
    out = out.loc[~invalid].copy()

    out = out.sort_values(["symbol", "ts_event"]).reset_index(drop=True)
    dropped = before - len(out)
    return CleanResult(frame=out, dropped_rows=dropped, invalid_ohlc_rows=invalid_count)


def filter_regular_session(
    df: pd.DataFrame,
    start: str = "09:30",
    end: str = "16:00",
    regular_session_only: bool = True,
) -> pd.DataFrame:
    """Filter to weekdays and optional regular US session interval."""
    out = df.copy()

    # Remove weekends first.
    out = out[out["ts_event"].dt.dayofweek < 5]

    if not regular_session_only:
        return out

    # Use timestamp as index for robust between_time filtering.
    pieces: list[pd.DataFrame] = []
    for symbol, grp in out.groupby("symbol", sort=False):
        indexed = grp.set_index("ts_event", drop=False)
        sliced = indexed.between_time(start, end, inclusive="left").copy()
        sliced["symbol"] = symbol
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

    base_minutes = _timeframe_to_minutes(base_timeframe)
    target_minutes = _timeframe_to_minutes(target_timeframe)
    expected_rows = target_minutes // base_minutes
    if expected_rows <= 0:
        raise ValueError("target_timeframe must be larger than base_timeframe for resampling.")

    out_frames: list[pd.DataFrame] = []
    for symbol, grp in df.groupby("symbol", sort=False):
        indexed = grp.set_index("ts_event")
        ohlcv = indexed.resample(target_timeframe, label=label, closed=closed).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        counts = indexed["close"].resample(target_timeframe, label=label, closed=closed).count()

        if drop_incomplete_bars:
            ohlcv = ohlcv[counts == expected_rows]

        ohlcv = ohlcv.dropna(subset=["open", "high", "low", "close"])
        ohlcv["symbol"] = symbol
        out_frames.append(ohlcv.reset_index())

    if not out_frames:
        return df.iloc[0:0].copy()

    out = pd.concat(out_frames, axis=0, ignore_index=True)
    out = out.sort_values(["symbol", "ts_event"]).reset_index(drop=True)
    return out


def prepare_intraday_15m(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Apply session filtering + resampling using config values."""
    data_cfg = cfg["data_source"]
    session_cfg = data_cfg.get("session", {})
    res_cfg = cfg["resampling"]

    session_filtered = filter_regular_session(
        df,
        start=session_cfg.get("start", "09:30"),
        end=session_cfg.get("end", "16:00"),
        regular_session_only=session_cfg.get("regular_session_only", True),
    )

    out = resample_ohlcv(
        session_filtered,
        base_timeframe=res_cfg.get("base_timeframe", "1min"),
        target_timeframe=res_cfg.get("target_timeframe", "15min"),
        drop_incomplete_bars=res_cfg.get("drop_incomplete_bars", True),
    )
    return out
