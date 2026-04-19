from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from candlestick.domain import COLUMN_LABEL, COLUMN_REASON, COLUMN_SPLIT, COLUMN_SYMBOL, COLUMN_TS_EVENT, COLUMN_WINDOW_END_TS, PatternEventReason, SplitName, SPLIT_NAMES

SPLIT_NAMES = ("train", "val", "test")


def _count_by_value(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    counts = series.astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.to_dict().items()}


def _normalize_timestamps(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(values, errors="coerce", utc=True)


def summarize_processed_prices(prices: pd.DataFrame) -> dict[str, Any]:
    if prices.empty:
        return {
            "available": False,
            "row_count": 0,
            "symbol_count": 0,
            "symbols": [],
            "first_ts": None,
            "last_ts": None,
            "trading_day_count": 0,
            "bars_per_day_distribution": {},
        }

    ts = _normalize_timestamps(prices[COLUMN_TS_EVENT])
    ts_valid = ts.dropna()
    trading_days = ts_valid.dt.strftime("%Y-%m-%d")

    symbols = []
    if COLUMN_SYMBOL in prices.columns:
        symbols = sorted(str(symbol) for symbol in prices[COLUMN_SYMBOL].dropna().astype(str).unique())

    return {
        "available": True,
        "row_count": int(len(prices)),
        "symbol_count": int(len(symbols)),
        "symbols": symbols,
        "first_ts": ts_valid.iloc[0].isoformat() if not ts_valid.empty else None,
        "last_ts": ts_valid.iloc[-1].isoformat() if not ts_valid.empty else None,
        "trading_day_count": int(trading_days.nunique()) if not trading_days.empty else 0,
        "bars_per_day_distribution": _count_by_value(trading_days.value_counts()),
    }


def summarize_pattern_events(pattern: str, events: pd.DataFrame | None, source: str = "live_generated") -> dict[str, Any]:
    if events is None:
        return {
            "pattern": pattern,
            "source": source,
            "available": False,
            "event_count": 0,
            "positive_events": None,
            "negative_events": None,
            "reason_counts": {},
        }

    reason_counts = {reason.value: 0 for reason in PatternEventReason}
    if not events.empty and "reason" in events.columns:
        observed = _count_by_value(events["reason"])
        reason_counts.update({key: int(value) for key, value in observed.items()})

    positive_events = int((events["label"] == 1).sum()) if "label" in events.columns else 0
    negative_events = int((events["label"] == 0).sum()) if "label" in events.columns else 0

    return {
        "pattern": pattern,
        "source": source,
        "available": True,
        "event_count": int(len(events)),
        "positive_events": positive_events,
        "negative_events": negative_events,
        "reason_counts": reason_counts,
    }


def summarize_pattern_dataset(
    pattern: str,
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame | None,
    source: str = "live_generated",
) -> dict[str, Any]:
    meta = meta if meta is not None else pd.DataFrame()
    y_arr = np.asarray(y, dtype=np.int64)
    return {
        "pattern": pattern,
        "source": source,
        "available": True,
        "samples": int(len(y_arr)),
        "positive_labels": int((y_arr == 1).sum()) if len(y_arr) else 0,
        "negative_labels": int((y_arr == 0).sum()) if len(y_arr) else 0,
        "sequence_length": int(X.shape[1]) if X.ndim >= 2 and len(y_arr) else 0,
        "feature_count": int(X.shape[2]) if X.ndim >= 3 and len(y_arr) else 0,
        "metadata_rows": int(len(meta)),
        "first_window_end_ts": str(meta["window_end_ts"].iloc[0]) if not meta.empty and "window_end_ts" in meta else None,
        "last_window_end_ts": str(meta["window_end_ts"].iloc[-1]) if not meta.empty and "window_end_ts" in meta else None,
    }


def summarize_split_support(split_data: dict[str, np.ndarray] | None, meta_split: pd.DataFrame | None) -> dict[str, Any]:
    if split_data is None:
        return {
            "available": False,
            "folds": {},
        }

    folds: dict[str, dict[str, int]] = {}
    for split_name in SPLIT_NAMES:
        y_fold = np.asarray(split_data[f"y_{split_name}"], dtype=np.int64)
        folds[split_name] = {
            "samples": int(len(y_fold)),
            "positive": int((y_fold == 1).sum()) if len(y_fold) else 0,
            "negative": int((y_fold == 0).sum()) if len(y_fold) else 0,
        }

    if meta_split is not None and not meta_split.empty and "split" in meta_split.columns:
        folds["metadata_split_rows"] = _count_by_value(meta_split["split"])

    return {
        "available": True,
        "folds": folds,
    }


def label_sanity_issues(
    event_summary: dict[str, Any] | None,
    dataset_summary: dict[str, Any] | None,
    split_summary: dict[str, Any] | None,
    *,
    split_error: str | None = None,
) -> list[str]:
    issues: list[str] = []

    if event_summary and event_summary.get("available") and int(event_summary.get("positive_events") or 0) == 0:
        issues.append("zero_positive_events")

    if dataset_summary and int(dataset_summary.get("positive_labels") or 0) == 0:
        issues.append("zero_positive_dataset_labels")

    if split_summary and split_summary.get("available"):
        folds = split_summary.get("folds", {})
        for split_name in SPLIT_NAMES:
            fold = folds.get(split_name, {})
            if fold and int(fold.get("positive", 0)) == 0:
                issues.append(f"zero_positive_support_{split_name}")

    if split_error:
        issues.append("split_error")

    return issues
