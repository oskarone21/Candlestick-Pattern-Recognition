from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


def _make_raw_rows(day: str, start: str, periods: int) -> pd.DataFrame:
    ts = pd.date_range(start=f"{day} {start}", periods=periods, freq="1min")
    close = [100.0 + (i * 0.01) for i in range(periods)]
    open_ = [close[0]] + close[:-1]
    high = [max(o, c) + 0.05 for o, c in zip(open_, close)]
    low = [min(o, c) - 0.05 for o, c in zip(open_, close)]
    return pd.DataFrame(
        {
            "symbol": "SPY",
            "ts_event": ts.astype(str),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1000] * periods,
        }
    )


def test_prepare_dataset_pipeline_creates_cleaned_outputs(tmp_path, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    raw_path = tmp_path / "raw_spy_1m.csv"
    cleaned_path = tmp_path / "spy_1m_cleaned.csv"
    processed_path = tmp_path / "spy_15m.csv"
    tz_report_path = tmp_path / "timezone_report.json"
    quality_path = tmp_path / "prep_quality_report.json"
    session_coverage_path = tmp_path / "session_coverage_report.csv"

    cfg["paths"]["raw_1m_path"] = str(raw_path)
    cfg["paths"]["cleaned_1m_path"] = str(cleaned_path)
    cfg["paths"]["processed_15m_path"] = str(processed_path)
    cfg["paths"]["timezone_report"] = str(tz_report_path)
    cfg["paths"]["data_quality_report"] = str(quality_path)
    cfg["paths"]["session_coverage_report"] = str(session_coverage_path)

    # Construct synthetic source data in naive America/Denver timestamps.
    full_day = _make_raw_rows("2019-12-23", "07:30", 390)
    valid_early_close = _make_raw_rows("2019-12-24", "07:30", 210)
    # Extra minutes on early-close day (should be dropped by anomaly policy).
    extra_minutes_early_close = _make_raw_rows("2019-11-29", "07:30", 225)
    # Missing opening block on regular day (should be dropped).
    missing_open_partial = _make_raw_rows("2019-12-20", "08:08", 352)
    duplicated_rows = full_day.iloc[:5].copy()
    raw = pd.concat(
        [full_day, valid_early_close, extra_minutes_early_close, missing_open_partial, duplicated_rows],
        ignore_index=True,
    )
    raw.to_csv(raw_path, index=False)

    cfg_path = tmp_path / "prep_config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "scripts/prepare_15m_dataset.py",
        "--config",
        str(cfg_path),
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    assert cleaned_path.exists()
    assert processed_path.exists()
    assert tz_report_path.exists()
    assert quality_path.exists()
    assert session_coverage_path.exists()

    prepared = pd.read_csv(processed_path)
    assert int(prepared.duplicated(subset=["symbol", "ts_event"]).sum()) == 0

    per_day = prepared["ts_event"].str.slice(0, 10).value_counts().to_dict()
    assert per_day == {"2019-12-23": 26, "2019-12-24": 14}

    with quality_path.open("r", encoding="utf-8") as f:
        quality = json.load(f)
    assert quality["session_calendar_summary"]["dropped_days"] >= 2
    assert quality["processed_price_signature"]["row_count"] == 40
    assert quality["processed_price_signature"]["trading_day_count"] == 2
