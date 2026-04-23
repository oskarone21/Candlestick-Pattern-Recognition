from __future__ import annotations

import pandas as pd

from chart_patterns.data.clean_resample import drop_exact_duplicate_ohlcv
from chart_patterns.data.session_quality import build_xnys_schedule, validate_intraday_by_session_calendar


def _make_intraday_frame(day: str, start: str, periods: int) -> pd.DataFrame:
    ts = pd.date_range(start=f"{day} {start}", periods=periods, freq="1min", tz="America/New_York")
    close = [100.0 + (i * 0.01) for i in range(periods)]
    open_ = [close[0]] + close[:-1]
    high = [max(o, c) + 0.05 for o, c in zip(open_, close)]
    low = [min(o, c) - 0.05 for o, c in zip(open_, close)]
    return pd.DataFrame(
        {
            "symbol": "SPY",
            "ts_event": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1000] * periods,
        }
    )


def test_drop_exact_duplicate_ohlcv_preserves_non_conflicting_rows():
    base = _make_intraday_frame("2024-01-02", "09:30", 3)
    exact_duplicate = base.iloc[[1]].copy()
    conflicting_same_minute = base.iloc[[2]].copy()
    conflicting_same_minute["close"] = conflicting_same_minute["close"] + 0.1

    frame = pd.concat([base, exact_duplicate, conflicting_same_minute], ignore_index=True)
    dedup = drop_exact_duplicate_ohlcv(frame)

    assert dedup.dropped_rows == 1
    # The conflicting same-minute row is intentionally retained for downstream handling.
    assert len(dedup.frame) == 4
    assert dedup.frame["ts_event"].is_monotonic_increasing


def test_session_calendar_keeps_valid_early_close_and_drops_anomalous_days():
    full_day = _make_intraday_frame("2019-12-23", "09:30", 390)
    valid_early_close = _make_intraday_frame("2019-12-24", "09:30", 210)

    # Official early-close day with extra minutes (09:30 -> 13:14) should be dropped.
    extra_minutes_early_close = _make_intraday_frame("2019-11-29", "09:30", 225)
    # Standard full day with missing opening session (10:08 -> 15:59) should be dropped.
    missing_open_partial = _make_intraday_frame("2019-12-20", "10:08", 352)

    frame = pd.concat(
        [full_day, valid_early_close, extra_minutes_early_close, missing_open_partial],
        ignore_index=True,
    )
    result = validate_intraday_by_session_calendar(
        frame,
        calendar="XNYS",
        regular_start="09:30",
        regular_end="16:00",
        early_close_end="13:00",
        keep_official_early_closes=True,
        drop_anomalous_partial_days=True,
    )

    kept_days = pd.Series(result.frame["ts_event"].dt.date.unique()).astype(str).sort_values().tolist()
    assert kept_days == ["2019-12-23", "2019-12-24"]
    assert len(result.frame) == 600

    coverage = result.coverage.set_index("date")
    assert coverage.loc["2019-11-29", "status"] == "extra_minutes"
    assert bool(coverage.loc["2019-11-29", "drop_day"])
    assert coverage.loc["2019-12-20", "status"] == "missing_minutes"
    assert bool(coverage.loc["2019-12-20", "drop_day"])
    assert result.summary["dropped_days"] >= 2


def test_xnys_schedule_keeps_dec_31_when_new_year_is_saturday():
    schedule = build_xnys_schedule(
        start_date=pd.Timestamp("2010-12-30").date(),
        end_date=pd.Timestamp("2011-01-03").date(),
        timezone="America/New_York",
    )
    days = set(schedule["date"].astype(str).tolist())
    assert "2010-12-31" in days
