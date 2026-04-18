from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from candlestick.config import load_config
from candlestick.data.kaggle_ingest import load_kaggle_dataframe, save_raw_csv
from candlestick.data.raw_quality import build_raw_quality_report, write_raw_quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SPY 1-minute data from Kaggle via kagglehub")
    parser.add_argument("--config", default="configs/config.yaml", help="Base config path")
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="Optional override YAML path (can be repeated)",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        help="Dot-notation override (e.g., data_source.kaggle_file_path=spy.csv)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.config_override, args.set_overrides)

    df = load_kaggle_dataframe(cfg)
    symbol_filter = cfg["data_source"].get("instrument", "SPY")
    if "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.upper() == str(symbol_filter).upper()].copy()

    output_path = cfg["paths"].get("raw_1m_path", "data/raw/spy_1m.csv")
    out = save_raw_csv(df, output_path)
    quality_path = cfg["paths"].get("raw_1m_quality_report", "outputs/data_quality/raw_1m_quality_report.json")
    quality_report = build_raw_quality_report(cfg, df)
    quality_out = write_raw_quality_report(quality_report, quality_path)

    print(f"Saved normalized raw intraday dataset: {out}")
    print(f"Raw quality report saved to: {quality_out}")
    print(f"Rows: {len(df)}")
    if not df.empty:
        print(f"Time range: {df['ts_event'].min()} -> {df['ts_event'].max()}")
        print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
