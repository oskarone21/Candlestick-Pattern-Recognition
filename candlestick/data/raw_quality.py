from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DatetimeTZDtype

from candlestick.config import ensure_dir
from candlestick.domain import (
    COLUMN_CLOSE,
    COLUMN_HIGH,
    COLUMN_LOW,
    COLUMN_OPEN,
    COLUMN_SYMBOL,
    COLUMN_TS_EVENT,
    COLUMN_VOLUME,
    DEFAULT_SESSION_END,
    DEFAULT_SESSION_START,
    DEFAULT_TIMEZONE_IN_DATA,
    DEFAULT_WORKING_TIMEZONE,
)


def _hhmm_to_minute(value: str) -> int:
    hour, minute = [int(x) for x in value.split(":")]
    return hour * 60 + minute


def _daily_count_stats(series: pd.Series) -> dict[str, Any]:
    if series.empty:
        return {"days": 0, "min": 0, "median": 0.0, "max": 0}
    counts = series.value_counts()
    return {
        "days": int(len(counts)),
        "min": int(counts.min()),
        "median": float(counts.median()),
        "max": int(counts.max()),
    }


def _timezone_fit_metrics(
    frame: pd.DataFrame,
    timezone_in_data: str,
    working_timezone: str,
    session_start: str,
    session_end: str,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows_in_session": 0,
            "duplicates_in_session": 0,
            "daily_counts": {"days": 0, "min": 0, "median": 0.0, "max": 0},
            "full_session_days": 0,
            "expected_minutes_per_day": _hhmm_to_minute(session_end) - _hhmm_to_minute(session_start),
        }

    ts = frame[COLUMN_TS_EVENT]
    if isinstance(ts.dtype, DatetimeTZDtype):
        local = ts.dt.tz_convert(timezone_in_data)
    else:
        local = ts.dt.tz_localize(timezone_in_data, ambiguous="NaT", nonexistent="shift_forward")

    converted = local.dropna().dt.tz_convert(working_timezone)
    start_minute = _hhmm_to_minute(session_start)
    end_minute = _hhmm_to_minute(session_end)
    minute_of_day = converted.dt.hour * 60 + converted.dt.minute
    in_session = converted[(converted.dt.dayofweek < 5) & (minute_of_day >= start_minute) & (minute_of_day < end_minute)]

    if in_session.empty:
        return {
            "rows_in_session": 0,
            "duplicates_in_session": 0,
            "daily_counts": {"days": 0, "min": 0, "median": 0.0, "max": 0},
            "full_session_days": 0,
            "expected_minutes_per_day": end_minute - start_minute,
        }

    daily = in_session.dt.date
    daily_counts = daily.value_counts()
    expected = end_minute - start_minute

    return {
        "rows_in_session": int(len(in_session)),
        "duplicates_in_session": int(pd.Series(in_session.astype(str)).duplicated().sum()),
        "daily_counts": _daily_count_stats(daily),
        "full_session_days": int((daily_counts == expected).sum()),
        "expected_minutes_per_day": expected,
    }


def build_raw_quality_report(cfg: dict[str, Any], raw_df: pd.DataFrame) -> dict[str, Any]:
    out = raw_df.copy()
    out[COLUMN_TS_EVENT] = pd.to_datetime(out[COLUMN_TS_EVENT], errors="coerce")
    out = out.dropna(subset=[COLUMN_TS_EVENT]).copy()

    session_cfg = cfg.get("data_source", {}).get("session", {})
    ts_cfg = cfg.get("data_source", {}).get("timestamp", {})
    session_start = session_cfg.get("start", DEFAULT_SESSION_START)
    session_end = session_cfg.get("end", DEFAULT_SESSION_END)
    timezone_in_data = ts_cfg.get("timezone_in_data", DEFAULT_TIMEZONE_IN_DATA)
    working_timezone = ts_cfg.get("convert_to_timezone", DEFAULT_WORKING_TIMEZONE)

    duplicate_symbol_ts = int(out.duplicated(subset=[COLUMN_SYMBOL, COLUMN_TS_EVENT]).sum())
    duplicate_symbol_ts_ohlcv = int(
        out.duplicated(
            subset=[
                COLUMN_SYMBOL,
                COLUMN_TS_EVENT,
                COLUMN_OPEN,
                COLUMN_HIGH,
                COLUMN_LOW,
                COLUMN_CLOSE,
                COLUMN_VOLUME,
            ]
        ).sum()
    )

    dedup_exact = out.drop_duplicates(
        subset=[COLUMN_SYMBOL, COLUMN_TS_EVENT, COLUMN_OPEN, COLUMN_HIGH, COLUMN_LOW, COLUMN_CLOSE, COLUMN_VOLUME],
        keep="last",
    )

    grouped = dedup_exact.groupby([COLUMN_SYMBOL, COLUMN_TS_EVENT], sort=False).size()
    conflicting_groups = int((grouped > 1).sum())

    dedup_symbol_ts = out.drop_duplicates(subset=[COLUMN_SYMBOL, COLUMN_TS_EVENT], keep="last").copy()
    daily_unique_minutes = dedup_symbol_ts.groupby(dedup_symbol_ts[COLUMN_TS_EVENT].dt.date)[COLUMN_TS_EVENT].nunique()
    regime_counts = daily_unique_minutes.value_counts().sort_values(ascending=False)

    candidate_timezones = []
    for tz in [
        timezone_in_data,
        DEFAULT_WORKING_TIMEZONE,
        "America/Chicago",
        DEFAULT_TIMEZONE_IN_DATA,
        "America/Los_Angeles",
        "UTC",
    ]:
        if tz not in candidate_timezones:
            candidate_timezones.append(tz)

    timezone_fit: dict[str, Any] = {}
    for tz in candidate_timezones:
        timezone_fit[tz] = _timezone_fit_metrics(
            frame=dedup_symbol_ts,
            timezone_in_data=tz,
            working_timezone=working_timezone,
            session_start=session_start,
            session_end=session_end,
        )

    best_timezone = None
    best_full_days = -1
    for tz, metrics in timezone_fit.items():
        full_days = int(metrics.get("full_session_days", 0))
        if full_days > best_full_days:
            best_timezone = tz
            best_full_days = full_days

    report = {
        "rows": int(len(raw_df)),
        "rows_with_parseable_timestamp": int(len(out)),
        "rows_dropped_for_null_timestamp": int(len(raw_df) - len(out)),
        "range_start_raw": str(out[COLUMN_TS_EVENT].min()) if not out.empty else None,
        "range_end_raw": str(out[COLUMN_TS_EVENT].max()) if not out.empty else None,
        "duplicate_symbol_timestamp": duplicate_symbol_ts,
        "duplicate_symbol_timestamp_ohlcv_exact": duplicate_symbol_ts_ohlcv,
        "conflicting_duplicate_groups_symbol_timestamp": conflicting_groups,
        "daily_unique_minutes_distribution_top": {
            str(int(k)): int(v) for k, v in regime_counts.head(15).to_dict().items()
        },
        "timezone_fit": timezone_fit,
        "best_fit_timezone_by_full_session_days": best_timezone,
        "configured_timezone_in_data": timezone_in_data,
        "configured_working_timezone": working_timezone,
    }
    return report


def write_raw_quality_report(report: dict[str, Any], output_path: str | Path) -> Path:
    out = Path(output_path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out
