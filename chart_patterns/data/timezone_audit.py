from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DatetimeTZDtype

from chart_patterns.config import ensure_dir


class TimezoneError(RuntimeError):
    """Raised when timestamp timezone requirements are not satisfied."""


@dataclass
class TimezoneResult:
    frame: pd.DataFrame
    report: dict[str, Any]


def _is_tz_aware(series: pd.Series) -> bool:
    return isinstance(series.dtype, DatetimeTZDtype)


def _offset_distribution_minutes(ts: pd.Series) -> dict[str, int]:
    if not _is_tz_aware(ts):
        return {}
    offsets = ts.map(lambda x: int(x.utcoffset().total_seconds() // 60) if pd.notna(x) else None)
    counts = offsets.value_counts(dropna=True)
    return {str(int(k)): int(v) for k, v in counts.items()}


def _monotonic_violations(df: pd.DataFrame) -> int:
    violations = 0
    for _, grp in df.groupby("symbol", sort=False):
        if not grp["ts_event"].is_monotonic_increasing:
            violations += 1
    return violations


def _dst_transition_markers(ts: pd.Series) -> list[str]:
    if not _is_tz_aware(ts):
        return []

    offsets = ts.map(lambda x: x.utcoffset() if pd.notna(x) else None)
    changed = offsets.ne(offsets.shift())
    markers = ts[changed.fillna(False)]
    # Skip the first row transition marker (always different from NaN)
    markers = markers.iloc[1:]
    return [str(x) for x in markers.head(10).tolist()]


def normalize_and_audit_timezone(
    df: pd.DataFrame,
    timezone_in_data: str | None,
    working_timezone: str,
    fail_on_naive: bool = True,
) -> TimezoneResult:
    """Normalize ts_event timezone and return report with quality checks."""
    out = df.copy()
    out["ts_event"] = pd.to_datetime(out["ts_event"], errors="coerce")

    null_ts = int(out["ts_event"].isna().sum())
    if null_ts > 0:
        out = out.dropna(subset=["ts_event"]).reset_index(drop=True)

    tz_aware_before = _is_tz_aware(out["ts_event"])
    if not tz_aware_before:
        if not timezone_in_data:
            if fail_on_naive:
                raise TimezoneError(
                    "Input timestamps are naive and timezone_in_data is missing. "
                    "Set data_source.timestamp.timezone_in_data explicitly."
                )
        else:
            out["ts_event"] = out["ts_event"].dt.tz_localize(
                timezone_in_data,
                ambiguous="NaT",
                nonexistent="shift_forward",
            )
            out = out.dropna(subset=["ts_event"]).reset_index(drop=True)

    out["ts_event"] = out["ts_event"].dt.tz_convert(working_timezone)

    report = {
        "rows": int(len(out)),
        "null_timestamps_dropped": null_ts,
        "timezone_aware_before": tz_aware_before,
        "timezone_in_data_config": timezone_in_data,
        "working_timezone": working_timezone,
        "timestamp_dtype": str(out["ts_event"].dtype),
        "offset_distribution_minutes": _offset_distribution_minutes(out["ts_event"]),
        "dst_transition_samples": _dst_transition_markers(out["ts_event"]),
        "duplicate_timestamps": int(out.duplicated(subset=["symbol", "ts_event"]).sum()),
        "symbols_with_non_monotonic_time": _monotonic_violations(out),
        "range_start": str(out["ts_event"].min()) if not out.empty else None,
        "range_end": str(out["ts_event"].max()) if not out.empty else None,
    }

    return TimezoneResult(frame=out, report=report)


def write_timezone_report(report: dict[str, Any], output_path: str | Path) -> Path:
    out = Path(output_path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out
