from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd
from pandas import DatetimeTZDtype

from chart_patterns.domain import (
    COLUMN_SYMBOL,
    DEFAULT_EARLY_CLOSE_END,
    DEFAULT_MARKET_CALENDAR,
    DEFAULT_SESSION_END,
    DEFAULT_SESSION_START,
)


def _hhmm_to_minute(value: str) -> int:
    hour, minute = [int(x) for x in value.split(":")]
    return hour * 60 + minute


def _minute_to_hhmm(value: int) -> str:
    return f"{value // 60:02d}:{value % 60:02d}"


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    shift = (weekday - first.weekday()) % 7
    return first + timedelta(days=shift + (n - 1) * 7)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    if month == 12:
        pivot = date(year + 1, 1, 1)
    else:
        pivot = date(year, month + 1, 1)
    candidate = pivot - timedelta(days=1)
    shift = (candidate.weekday() - weekday) % 7
    return candidate - timedelta(days=shift)


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    d = date(year, month, day)
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _observed_new_year_holiday(year: int) -> date:
    d = date(year, 1, 1)
    # NYSE does not observe a Friday holiday when Jan 1 falls on Saturday.
    if d.weekday() == 5:
        return d
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _easter_sunday(year: int) -> date:
    # Meeus/Jones/Butcher Gregorian algorithm.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


_XNYS_SPECIAL_FULL_DAY_CLOSURES: set[date] = {
    date(2012, 10, 29),  # Hurricane Sandy
    date(2012, 10, 30),  # Hurricane Sandy
    date(2018, 12, 5),   # National Day of Mourning (G.H.W. Bush)
}


def xnys_holidays_for_year(year: int) -> set[date]:
    holidays = {
        _observed_new_year_holiday(year),         # New Year
        _nth_weekday_of_month(year, 1, 0, 3),     # MLK Day
        _nth_weekday_of_month(year, 2, 0, 3),     # Presidents Day
        _easter_sunday(year) - timedelta(days=2), # Good Friday
        _last_weekday_of_month(year, 5, 0),       # Memorial Day
        _observed_fixed_holiday(year, 7, 4),      # Independence Day
        _nth_weekday_of_month(year, 9, 0, 1),     # Labor Day
        _nth_weekday_of_month(year, 11, 3, 4),    # Thanksgiving
        _observed_fixed_holiday(year, 12, 25),    # Christmas
    }

    # Juneteenth became a US market holiday from 2022 onward.
    if year >= 2022:
        holidays.add(_observed_fixed_holiday(year, 6, 19))

    holidays.update(d for d in _XNYS_SPECIAL_FULL_DAY_CLOSURES if d.year == year)
    return holidays


def xnys_early_closes_for_year(year: int) -> set[date]:
    early_closes: set[date] = set()

    # Day after Thanksgiving (Friday).
    thanksgiving = _nth_weekday_of_month(year, 11, 3, 4)
    early_closes.add(thanksgiving + timedelta(days=1))

    # Christmas Eve (when it lands Monday-Thursday).
    christmas_eve = date(year, 12, 24)
    if christmas_eve.weekday() <= 3:
        early_closes.add(christmas_eve)

    # Independence Day eve when July 4 falls Tuesday-Friday.
    july_fourth = date(year, 7, 4)
    if july_fourth.weekday() in {1, 2, 3, 4}:
        early_closes.add(july_fourth - timedelta(days=1))

    # Remove anything that is not a weekday or is a full-day market holiday.
    holidays = xnys_holidays_for_year(year)
    return {d for d in early_closes if d.weekday() < 5 and d not in holidays}


def build_xnys_schedule(
    start_date: date,
    end_date: date,
    timezone: str,
    regular_start: str = DEFAULT_SESSION_START,
    regular_end: str = DEFAULT_SESSION_END,
    early_close_end: str = DEFAULT_EARLY_CLOSE_END,
    keep_official_early_closes: bool = True,
) -> pd.DataFrame:
    if start_date > end_date:
        return pd.DataFrame(
            columns=[
                "date",
                "session_open",
                "session_close",
                "expected_open_minute",
                "expected_close_minute",
                "expected_minutes",
                "is_official_early_close",
            ]
        )

    start_minute = _hhmm_to_minute(regular_start)
    regular_end_minute = _hhmm_to_minute(regular_end)
    early_end_minute = _hhmm_to_minute(early_close_end)

    business_days = pd.date_range(start_date, end_date, freq="B").date
    years = range(start_date.year, end_date.year + 1)
    holidays = set().union(*(xnys_holidays_for_year(year) for year in years))
    early_closes = set().union(*(xnys_early_closes_for_year(year) for year in years))

    records: list[dict[str, Any]] = []
    for day in business_days:
        if day in holidays:
            continue

        is_early = keep_official_early_closes and day in early_closes
        close_minute = early_end_minute if is_early else regular_end_minute
        records.append(
            {
                "date": day,
                "session_open": pd.Timestamp(f"{day} {regular_start}", tz=timezone),
                "session_close": pd.Timestamp(f"{day} {_minute_to_hhmm(close_minute)}", tz=timezone),
                "expected_open_minute": start_minute,
                "expected_close_minute": close_minute,
                "expected_minutes": close_minute - start_minute,
                "is_official_early_close": bool(is_early),
            }
        )

    return pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)


@dataclass
class SessionValidationResult:
    frame: pd.DataFrame
    coverage: pd.DataFrame
    summary: dict[str, Any]


def validate_intraday_by_session_calendar(
    df: pd.DataFrame,
    calendar: str = DEFAULT_MARKET_CALENDAR,
    regular_start: str = DEFAULT_SESSION_START,
    regular_end: str = DEFAULT_SESSION_END,
    early_close_end: str = DEFAULT_EARLY_CLOSE_END,
    keep_official_early_closes: bool = True,
    drop_anomalous_partial_days: bool = True,
) -> SessionValidationResult:
    if calendar.upper() != DEFAULT_MARKET_CALENDAR:
        raise ValueError(f"Unsupported session calendar '{calendar}'. Only XNYS is currently implemented.")

    if df.empty:
        empty_cov = pd.DataFrame(
            columns=[
                COLUMN_SYMBOL,
                "date",
                "is_trading_day",
                "is_official_early_close",
                "expected_minutes",
                "observed_unique_minutes",
                "missing_minutes",
                "extra_minutes",
                "status",
                "drop_day",
                "first_observed",
                "last_observed",
                "missing_minutes_sample",
                "extra_minutes_sample",
            ]
        )
        return SessionValidationResult(
            frame=df.copy(),
            coverage=empty_cov,
            summary={
                "calendar": DEFAULT_MARKET_CALENDAR,
                "rows_in": 0,
                "rows_out": 0,
                "symbols": 0,
                "trading_days_expected": 0,
                "valid_days": 0,
                "missing_days": 0,
                "anomalous_days": 0,
                "dropped_days": 0,
                "dropped_day_reasons": {},
            },
        )

    if not isinstance(df["ts_event"].dtype, DatetimeTZDtype):
        raise ValueError(
            "validate_intraday_by_session_calendar requires timezone-aware ts_event values "
            "(already converted to the working exchange timezone)."
        )

    out = df.copy()
    out = out[out["ts_event"].dt.dayofweek < 5].copy()

    regular_start_minute = _hhmm_to_minute(regular_start)
    regular_end_minute = _hhmm_to_minute(regular_end)
    out["day"] = out["ts_event"].dt.date
    out["minute_of_day"] = out["ts_event"].dt.hour * 60 + out["ts_event"].dt.minute
    out = out[
        (out["minute_of_day"] >= regular_start_minute) & (out["minute_of_day"] < regular_end_minute)
    ].copy()

    timezone_name = str(out["ts_event"].dt.tz)
    coverage_rows: list[dict[str, Any]] = []
    kept_chunks: list[pd.DataFrame] = []

    for symbol, symbol_df in out.groupby("symbol", sort=False):
        symbol_df = symbol_df.sort_values("ts_event").copy()
        observed_days = set(symbol_df["day"].unique().tolist())
        if not observed_days:
            continue

        start_day = min(observed_days)
        end_day = max(observed_days)
        schedule = build_xnys_schedule(
            start_date=start_day,
            end_date=end_day,
            timezone=timezone_name,
            regular_start=regular_start,
            regular_end=regular_end,
            early_close_end=early_close_end,
            keep_official_early_closes=keep_official_early_closes,
        )
        schedule_map = {row["date"]: row for row in schedule.to_dict(orient="records")}
        day_groups = {d: g.copy() for d, g in symbol_df.groupby("day", sort=True)}
        all_days = sorted(set(schedule_map.keys()) | observed_days)

        for day in all_days:
            day_frame = day_groups.get(day)
            observed_minutes_set: set[int] = set()
            observed_count = 0
            first_observed: str | None = None
            last_observed: str | None = None
            if day_frame is not None and not day_frame.empty:
                observed_minutes_set = set(day_frame["minute_of_day"].astype(int).unique().tolist())
                observed_count = len(observed_minutes_set)
                first_observed = str(day_frame["ts_event"].min())
                last_observed = str(day_frame["ts_event"].max())

            sched = schedule_map.get(day)
            if sched is None:
                status = "non_trading_day_data" if observed_count > 0 else "non_trading_day"
                missing_count = 0
                extra_count = observed_count
                expected_minutes_set: set[int] = set()
                is_trading_day = False
                is_early_close = False
                expected_minutes = 0
                expected_open_minute = regular_start_minute
                expected_close_minute = regular_end_minute
            else:
                expected_open_minute = int(sched["expected_open_minute"])
                expected_close_minute = int(sched["expected_close_minute"])
                expected_minutes_set = set(range(expected_open_minute, expected_close_minute))
                missing_minutes_set = expected_minutes_set - observed_minutes_set
                extra_minutes_set = observed_minutes_set - expected_minutes_set
                missing_count = len(missing_minutes_set)
                extra_count = len(extra_minutes_set)
                is_trading_day = True
                is_early_close = bool(sched["is_official_early_close"])
                expected_minutes = int(sched["expected_minutes"])

                if observed_count == 0:
                    status = "missing_day"
                elif missing_count == 0 and extra_count == 0:
                    status = "valid"
                elif missing_count > 0 and extra_count > 0:
                    status = "missing_and_extra_minutes"
                elif missing_count > 0:
                    status = "missing_minutes"
                else:
                    status = "extra_minutes"

            drop_day = bool(
                observed_count > 0
                and status != "valid"
                and status != "non_trading_day"
                and drop_anomalous_partial_days
            )

            missing_sample: list[str] = []
            extra_sample: list[str] = []
            if sched is not None:
                missing_sample = [
                    _minute_to_hhmm(x)
                    for x in sorted(expected_minutes_set - observed_minutes_set)[:10]
                ]
                extra_sample = [
                    _minute_to_hhmm(x)
                    for x in sorted(observed_minutes_set - expected_minutes_set)[:10]
                ]
            else:
                extra_sample = [_minute_to_hhmm(x) for x in sorted(observed_minutes_set)[:10]]

            coverage_rows.append(
                {
                    "symbol": symbol,
                    "date": str(day),
                    "is_trading_day": bool(is_trading_day),
                    "is_official_early_close": bool(is_early_close),
                    "expected_open": _minute_to_hhmm(expected_open_minute),
                    "expected_close": _minute_to_hhmm(expected_close_minute),
                    "expected_minutes": int(expected_minutes),
                    "observed_unique_minutes": int(observed_count),
                    "missing_minutes": int(missing_count),
                    "extra_minutes": int(extra_count),
                    "status": status,
                    "drop_day": bool(drop_day),
                    "first_observed": first_observed,
                    "last_observed": last_observed,
                    "missing_minutes_sample": missing_sample,
                    "extra_minutes_sample": extra_sample,
                }
            )

            if day_frame is None or day_frame.empty:
                continue

            in_expected_window = (
                (day_frame["minute_of_day"] >= expected_open_minute)
                & (day_frame["minute_of_day"] < expected_close_minute)
            )
            if status == "valid":
                kept_chunks.append(day_frame.loc[in_expected_window].copy())
            elif not drop_anomalous_partial_days and status != "non_trading_day_data":
                kept_chunks.append(day_frame.loc[in_expected_window].copy())

    coverage = pd.DataFrame.from_records(coverage_rows).sort_values(["symbol", "date"]).reset_index(drop=True)

    if kept_chunks:
        cleaned = pd.concat(kept_chunks, axis=0, ignore_index=True)
        cleaned = cleaned.sort_values(["symbol", "ts_event"]).reset_index(drop=True)
    else:
        cleaned = out.iloc[0:0].copy()

    cleaned = cleaned.drop(columns=["day", "minute_of_day"], errors="ignore")

    summary = {
        "calendar": DEFAULT_MARKET_CALENDAR,
        "rows_in": int(len(df)),
        "rows_out": int(len(cleaned)),
        "symbols": int(df["symbol"].nunique()) if "symbol" in df.columns else 0,
        "trading_days_expected": int(coverage["is_trading_day"].sum()) if not coverage.empty else 0,
        "valid_days": int((coverage["status"] == "valid").sum()) if not coverage.empty else 0,
        "official_early_close_days_valid": int(
            ((coverage["status"] == "valid") & coverage["is_official_early_close"]).sum()
        )
        if not coverage.empty
        else 0,
        "missing_days": int((coverage["status"] == "missing_day").sum()) if not coverage.empty else 0,
        "anomalous_days": int(
            coverage["status"].isin(
                {"missing_minutes", "extra_minutes", "missing_and_extra_minutes", "non_trading_day_data"}
            ).sum()
        )
        if not coverage.empty
        else 0,
        "dropped_days": int(coverage["drop_day"].sum()) if not coverage.empty else 0,
        "dropped_day_reasons": (
            coverage.loc[coverage["drop_day"], "status"].value_counts().to_dict() if not coverage.empty else {}
        ),
    }

    return SessionValidationResult(frame=cleaned, coverage=coverage, summary=summary)
