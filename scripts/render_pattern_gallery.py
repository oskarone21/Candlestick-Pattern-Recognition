from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from candlestick.config import ensure_dir, load_config
from candlestick.viz.pattern_gallery import render_pattern_gallery


def main() -> None:
    parser = argparse.ArgumentParser(description="Render TP/FP/FN chart galleries for champion predictions")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--set", dest="set_overrides", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.config_override, args.set_overrides)

    run_name = cfg["project"].get("run_name", "pattern_suite")
    metrics_root = Path(cfg["paths"].get("metrics_dir", "outputs/metrics")) / run_name
    champions_path = metrics_root / "champions.csv"

    if not champions_path.exists():
        raise FileNotFoundError(
            f"Champion file not found: {champions_path}. Run scripts/run_experiment_suite.py first."
        )

    prices_path = Path(cfg["paths"].get("processed_15m_path", "data/processed/spy_15m.csv"))
    prices = pd.read_csv(prices_path)
    working_timezone = cfg.get("data_source", {}).get("timestamp", {}).get(
        "convert_to_timezone", "America/New_York"
    )
    prices["ts_event"] = pd.to_datetime(prices["ts_event"], errors="coerce", utc=True).dt.tz_convert(
        working_timezone
    )
    prices = prices.dropna(subset=["ts_event"]).copy()
    prices = prices[prices["symbol"].astype(str).str.upper() == "SPY"].sort_values("ts_event").reset_index(drop=True)

    gallery_root = ensure_dir(Path(cfg["paths"].get("gallery_dir", "outputs/gallery")) / run_name)
    champions = pd.read_csv(champions_path)

    counts = {}
    for _, row in champions.iterrows():
        pattern = str(row["pattern"])
        pred_path = Path(row["predictions_path"])
        threshold = float(row.get("threshold", 0.5))

        if not pred_path.exists():
            continue

        pred_df = pd.read_csv(pred_path)
        result = render_pattern_gallery(
            price_df=prices,
            pred_df=pred_df,
            threshold=threshold,
            output_dir=gallery_root,
            pattern=pattern,
            lookback=int(cfg["windowing"].get("lookback_bars", 80)),
            lookforward=int(cfg.get("chart_images", {}).get("gallery_lookforward_bars", 20)),
            max_per_bucket=int(cfg.get("chart_images", {}).get("gallery_max_per_bucket", 12)),
        )
        counts[pattern] = result

    with (Path(gallery_root) / "gallery_summary.json").open("w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)

    print(f"Pattern gallery generated at: {gallery_root}")


if __name__ == "__main__":
    main()
