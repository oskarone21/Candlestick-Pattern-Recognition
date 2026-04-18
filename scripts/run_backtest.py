from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from candlestick.config import ensure_dir, load_config
from candlestick.trading.backtest import run_backtest_for_predictions, save_backtest_outputs


def _load_prices(cfg: dict) -> pd.DataFrame:
    prices_path = Path(cfg["paths"].get("processed_15m_path", "data/processed/spy_15m.csv"))
    if not prices_path.exists():
        raise FileNotFoundError(f"Processed prices not found: {prices_path}")

    prices = pd.read_csv(prices_path)
    working_timezone = cfg.get("data_source", {}).get("timestamp", {}).get(
        "convert_to_timezone", "America/New_York"
    )
    prices["ts_event"] = pd.to_datetime(prices["ts_event"], errors="coerce", utc=True).dt.tz_convert(
        working_timezone
    )
    prices = prices.dropna(subset=["ts_event"]).copy()
    prices = prices[prices["symbol"].astype(str).str.upper() == "SPY"].copy()
    return prices.sort_values("ts_event").reset_index(drop=True)


def run_backtest(cfg: dict) -> dict[str, object]:
    run_name = cfg["project"].get("run_name", "pattern_suite")
    metrics_root = Path(cfg["paths"].get("metrics_dir", "outputs/metrics")) / run_name
    champions_path = metrics_root / "champions.csv"

    if not champions_path.exists():
        raise FileNotFoundError(
            f"Champion file not found: {champions_path}. Run scripts/run_experiment_suite.py first."
        )

    prices = _load_prices(cfg)
    bt_root = ensure_dir(Path(cfg["paths"].get("backtest_dir", "outputs/backtest")) / run_name)
    try:
        champions = pd.read_csv(champions_path)
    except EmptyDataError:
        champions = pd.DataFrame(columns=["pattern", "threshold", "predictions_path"])

    if champions.empty:
        summary_df = pd.DataFrame(
            columns=[
                "pattern",
                "trades",
                "signals_total",
                "accepted_signals",
                "rejected_signals",
                "return_on_capital",
                "total_pnl",
                "win_rate",
                "sharpe",
                "profit_factor",
                "max_drawdown",
                "expectancy",
            ]
        )
        summary_df.to_csv(Path(bt_root) / "backtest_summary.csv", index=False)
        with (Path(bt_root) / "backtest_summary.json").open("w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        print(f"Backtest skipped: no champion rows found in {champions_path}")
        print(f"Outputs at: {bt_root}")
        return {"backtest_root": bt_root, "summary_df": summary_df}

    summary_rows = []
    for _, row in champions.iterrows():
        pattern = str(row["pattern"])
        pred_path = Path(row["predictions_path"])
        threshold = float(row.get("threshold", 0.5))

        if not pred_path.exists():
            continue

        pred_df = pd.read_csv(pred_path)
        trades, summary = run_backtest_for_predictions(
            price_df=prices,
            pred_df=pred_df,
            cfg=cfg,
            threshold=threshold,
        )
        save_backtest_outputs(trades, summary, bt_root, pattern)

        summary_row = dict(summary)
        summary_row["pattern"] = pattern
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(Path(bt_root) / "backtest_summary.csv", index=False)

    with (Path(bt_root) / "backtest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"Backtest finished. Outputs at: {bt_root}")
    return {"backtest_root": bt_root, "summary_df": summary_df}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest on champion model predictions")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--set", dest="set_overrides", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.config_override, args.set_overrides)
    run_backtest(cfg)


if __name__ == "__main__":
    main()
