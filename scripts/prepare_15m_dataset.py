from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from scripts._bootstrap import ensure_repo_root
except ImportError:
    from _bootstrap import ensure_repo_root

ensure_repo_root()

from chart_patterns.config import ensure_dir, load_config
from chart_patterns.data.clean_resample import clean_ohlcv, drop_exact_duplicate_ohlcv, resample_ohlcv
from chart_patterns.data.kaggle_ingest import load_kaggle_dataframe, save_raw_csv
from chart_patterns.data.session_quality import validate_intraday_by_session_calendar
from chart_patterns.data.timezone_audit import normalize_and_audit_timezone, write_timezone_report
from chart_patterns.domain import (
    DEFAULT_EARLY_CLOSE_END,
    DEFAULT_INSTRUMENT,
    DEFAULT_MARKET_CALENDAR,
    DEFAULT_SESSION_END,
    DEFAULT_SESSION_START,
)
from chart_patterns.eval.label_sanity import summarize_processed_prices
from chart_patterns.project_utils import timeframe_to_minutes, working_timezone_from_cfg


def _read_or_download_raw(cfg: dict, refresh_raw: bool = False) -> pd.DataFrame:
    raw_path = Path(cfg["paths"].get("raw_1m_path", "data/raw/spy_1m.csv"))
    if raw_path.exists() and not refresh_raw:
        df = pd.read_csv(raw_path)
        return df

    df = load_kaggle_dataframe(cfg)
    symbol_filter = cfg["data_source"].get("instrument", DEFAULT_INSTRUMENT)
    df = df[df["symbol"].astype(str).str.upper() == str(symbol_filter).upper()].copy()
    save_raw_csv(df, raw_path)
    return df


def _write_session_coverage_report(coverage: pd.DataFrame, output_path: str | Path) -> Path:
    out = Path(output_path)
    ensure_dir(out.parent)

    export = coverage.copy()
    if "missing_minutes_sample" in export.columns:
        export["missing_minutes_sample"] = export["missing_minutes_sample"].apply(
            lambda x: ",".join(x) if isinstance(x, list) else x
        )
    if "extra_minutes_sample" in export.columns:
        export["extra_minutes_sample"] = export["extra_minutes_sample"].apply(
            lambda x: ",".join(x) if isinstance(x, list) else x
        )

    if out.suffix.lower() == ".json":
        with out.open("w", encoding="utf-8") as f:
            json.dump(export.to_dict(orient="records"), f, indent=2)
    else:
        export.to_csv(out, index=False)
    return out


def _distribution_from_series(values: pd.Series) -> dict[str, int]:
    if values.empty:
        return {}
    counts = values.value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in counts.to_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, audit timezone, and resample SPY intraday data to 15m")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--set", dest="set_overrides", action="append", default=[])
    parser.add_argument("--refresh-raw", action="store_true", help="Re-download raw data from Kaggle")
    args = parser.parse_args()

    cfg = load_config(args.config, args.config_override, args.set_overrides)

    raw_df = _read_or_download_raw(cfg, refresh_raw=args.refresh_raw)

    ts_cfg = cfg["data_source"]["timestamp"]
    timezone_in_data = ts_cfg.get("timezone_in_data")
    working_timezone = working_timezone_from_cfg(cfg)

    tz_result = normalize_and_audit_timezone(
        raw_df,
        timezone_in_data=timezone_in_data,
        working_timezone=working_timezone,
        fail_on_naive=True,
    )

    dedup_result = drop_exact_duplicate_ohlcv(tz_result.frame)

    clean_result = clean_ohlcv(
        dedup_result.frame,
        drop_duplicates=cfg["data_source"].get("quality", {}).get("drop_duplicates", True),
    )

    session_cfg = cfg["data_source"].get("session", {})
    session_result = validate_intraday_by_session_calendar(
        clean_result.frame,
        calendar=session_cfg.get("calendar", DEFAULT_MARKET_CALENDAR),
        regular_start=session_cfg.get("start", DEFAULT_SESSION_START),
        regular_end=session_cfg.get("end", DEFAULT_SESSION_END),
        early_close_end=session_cfg.get("early_close_end", DEFAULT_EARLY_CLOSE_END),
        keep_official_early_closes=session_cfg.get("keep_official_early_closes", True),
        drop_anomalous_partial_days=session_cfg.get("drop_anomalous_partial_days", True),
    )

    cleaned_1m_path = Path(cfg["paths"].get("cleaned_1m_path", "data/processed/spy_1m_cleaned.csv"))
    ensure_dir(cleaned_1m_path.parent)
    session_result.frame.to_csv(cleaned_1m_path, index=False)

    prepared = resample_ohlcv(
        session_result.frame,
        base_timeframe=cfg["resampling"].get("base_timeframe", "1min"),
        target_timeframe=cfg["resampling"].get("target_timeframe", "15min"),
        drop_incomplete_bars=cfg["resampling"].get("drop_incomplete_bars", True),
    )

    processed_path = Path(cfg["paths"].get("processed_15m_path", "data/processed/spy_15m.csv"))
    ensure_dir(processed_path.parent)
    prepared.to_csv(processed_path, index=False)

    tz_report_path = cfg["paths"].get("timezone_report", "outputs/data_quality/timezone_report.json")
    write_timezone_report(tz_result.report, tz_report_path)

    quality = {
        "raw_rows": int(len(raw_df)),
        "post_timezone_rows": int(len(tz_result.frame)),
        "exact_duplicate_rows_removed_before_clean": dedup_result.dropped_rows,
        "post_clean_rows": int(len(clean_result.frame)),
        "post_session_validation_rows": int(len(session_result.frame)),
        "post_resample_rows": int(len(prepared)),
        "dropped_rows_total": int(len(raw_df) - len(session_result.frame)),
        "invalid_ohlc_rows": clean_result.invalid_ohlc_rows,
        "timezone_report": tz_result.report,
        "session_calendar_summary": session_result.summary,
        "processed_price_signature": summarize_processed_prices(prepared),
    }

    target_minutes = timeframe_to_minutes(cfg["resampling"].get("target_timeframe", "15min"))

    coverage_df = session_result.coverage.copy()
    coverage_trading = coverage_df[coverage_df["is_trading_day"]].copy() if not coverage_df.empty else coverage_df
    if not coverage_trading.empty:
        coverage_trading["expected_target_bars"] = (
            coverage_trading["expected_minutes"].astype(int) // target_minutes
        )
    else:
        coverage_trading["expected_target_bars"] = pd.Series(dtype="int64")

    prepared_daily = prepared.copy()
    if not prepared_daily.empty:
        prepared_daily["date"] = prepared_daily["ts_event"].astype(str).str.slice(0, 10)
        observed_15m_per_day = prepared_daily.groupby("date").size()
    else:
        observed_15m_per_day = pd.Series(dtype="int64")

    if not coverage_trading.empty:
        coverage_trading["observed_target_bars"] = (
            coverage_trading["date"].map(observed_15m_per_day).fillna(0).astype(int)
        )
        coverage_trading["target_bars_match"] = (
            coverage_trading["observed_target_bars"] == coverage_trading["expected_target_bars"]
        )

    quality["coverage"] = {
        "session_1m": session_result.summary,
        "expected_target_bars_distribution": _distribution_from_series(
            coverage_trading["expected_target_bars"] if "expected_target_bars" in coverage_trading else pd.Series(dtype=int)
        ),
        "observed_target_bars_distribution": _distribution_from_series(
            coverage_trading["observed_target_bars"] if "observed_target_bars" in coverage_trading else pd.Series(dtype=int)
        ),
        "target_bar_mismatch_days": int(
            (~coverage_trading["target_bars_match"]).sum()
        )
        if "target_bars_match" in coverage_trading
        else 0,
    }

    coverage_path = Path(
        cfg["paths"].get("session_coverage_report", "outputs/data_quality/session_coverage_report.csv")
    )
    coverage_written = _write_session_coverage_report(session_result.coverage, coverage_path)
    quality_path = Path(cfg["paths"].get("data_quality_report", "outputs/data_quality/prep_quality_report.json"))
    ensure_dir(quality_path.parent)
    with quality_path.open("w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    print(f"Processed dataset saved to: {processed_path}")
    print(f"Canonical cleaned 1m dataset saved to: {cleaned_1m_path}")
    print(f"Timezone report saved to: {tz_report_path}")
    print(f"Session coverage report saved to: {coverage_written}")
    print(f"Quality report saved to: {quality_path}")
    print(f"Rows after resample: {len(prepared)}")


if __name__ == "__main__":
    main()
