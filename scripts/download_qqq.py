#!/usr/bin/env python3
"""Download QQQ 15-min OHLCV data via yfinance and save to data/qqq_15min.csv.

yfinance limitation: 15-min data is capped at ~60 days of history.
For longer history, use --interval 1h (up to ~730 days).

Usage:
    python scripts/download_qqq.py
    python scripts/download_qqq.py --interval 1h --period 2y
    python scripts/download_qqq.py --interval 15m --period 60d
"""

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INTERVAL_TO_FILENAME = {
    "1m":  "qqq_1min.csv",
    "2m":  "qqq_2min.csv",
    "5m":  "qqq_5min.csv",
    "15m": "qqq_15min.csv",
    "30m": "qqq_30min.csv",
    "1h":  "qqq_1h.csv",
    "1d":  "qqq_1d.csv",
}

# yfinance max period per interval
INTERVAL_MAX_PERIOD = {
    "1m":  "7d",
    "2m":  "60d",
    "5m":  "60d",
    "15m": "60d",
    "30m": "60d",
    "1h":  "730d",
    "1d":  "max",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download QQQ OHLCV via yfinance")
    p.add_argument("--ticker",   default="QQQ",  help="Yahoo Finance ticker (default: QQQ)")
    p.add_argument("--interval", default="15m",  choices=list(INTERVAL_TO_FILENAME), help="Bar interval")
    p.add_argument("--period",   default=None,   help="Lookback period (e.g. 60d, 2y, max). Defaults to max for the interval.")
    p.add_argument("--output",   default=None,   help="Output CSV path. Defaults to data/<ticker>_<interval>.csv")
    return p.parse_args()


def download(ticker: str, interval: str, period: str) -> pd.DataFrame:
    print(f"Downloading {ticker} | interval={interval} | period={period} …")
    raw = yf.download(
        tickers=ticker,
        interval=interval,
        period=period,
        auto_adjust=True,
        progress=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned an empty DataFrame — check ticker / period / interval.")
    return raw


def to_project_format(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance returns MultiIndex columns when downloading a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Ensure timezone-aware index (UTC)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "ts_event"
    df = df.reset_index()
    return df


def main() -> None:
    args = parse_args()

    period = args.period or INTERVAL_MAX_PERIOD.get(args.interval, "60d")

    raw = download(args.ticker, args.interval, period)
    df = to_project_format(raw)

    if args.output:
        out_path = Path(args.output)
    else:
        fname = INTERVAL_TO_FILENAME.get(args.interval, f"qqq_{args.interval}.csv")
        if args.ticker.upper() != "QQQ":
            fname = f"{args.ticker.lower()}_{args.interval}.csv"
        out_path = DATA_DIR / fname

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df):,} bars → {out_path}")
    print(f"  Date range : {df['ts_event'].iloc[0]}  →  {df['ts_event'].iloc[-1]}")
    print(f"  Columns    : {list(df.columns)}")


if __name__ == "__main__":
    main()
