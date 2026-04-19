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

from candlestick.config import ensure_dir, load_config
from candlestick.project_utils import load_processed_prices, run_name_from_cfg
from candlestick.viz.pattern_gallery import render_pattern_gallery


def main() -> None:
    parser = argparse.ArgumentParser(description="Render TP/FP/FN chart galleries for champion predictions")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--set", dest="set_overrides", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.config_override, args.set_overrides)

    run_name = run_name_from_cfg(cfg)
    metrics_root = Path(cfg["paths"].get("metrics_dir", "outputs/metrics")) / run_name
    champions_path = metrics_root / "champions.csv"

    if not champions_path.exists():
        raise FileNotFoundError(
            f"Champion file not found: {champions_path}. Run scripts/run_experiment_suite.py first."
        )

    prices = load_processed_prices(cfg)

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
