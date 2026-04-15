"""Download QQQ daily OHLCV data from Yahoo Finance via yfinance.

Usage:
    python scripts/download_qqq.py                  # default: QQQ, 10 years, data/QQQ_1d.csv
    python scripts/download_qqq.py --ticker SPY --years 5 --output data/SPY_1d.csv

The output CSV uses the column schema expected by the project pipeline:
    ts_event,Open,High,Low,Close,Volume
Timestamps are UTC-normalised.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


def download_qqq(
    ticker: str = "QQQ",
    years: int = 10,
    output: str | Path = "data/QQQ_1d.csv",
) -> Path:
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit(
            "yfinance is not installed. Run: pip install yfinance"
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now(tz=timezone.utc)
    start_date = end_date - timedelta(days=years * 365)

    print(f"Downloading {ticker} daily data from {start_date.date()} to {end_date.date()} ...")

    df: pd.DataFrame = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
    )

    if df.empty:
        raise SystemExit(f"No data returned for {ticker}. Check the ticker symbol and try again.")

    df = df.reset_index()

    if "Date" in df.columns:
        df = df.rename(columns={"Date": "ts_event"})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "ts_event"})

    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)

    flat_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            flat_columns.append(col[0] if col[0] != "" else col[1])
        else:
            flat_columns.append(col)
    df.columns = flat_columns

    df = df[["ts_event", "Open", "High", "Low", "Close", "Volume"]]
    df = df.sort_values("ts_event").reset_index(drop=True)

    df.to_csv(output, index=False)

    print(f"Downloaded {len(df)} rows -> {output}")
    print(f"  Date range : {df['ts_event'].iloc[0]}  ->  {df['ts_event'].iloc[-1]}")
    print(f"  Columns    : {list(df.columns)}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Download QQQ daily data via yfinance")
    parser.add_argument("--ticker", default="QQQ", help="Yahoo Finance ticker symbol (default: QQQ)")
    parser.add_argument("--years", type=int, default=10, help="How many years of history to download (default: 10)")
    parser.add_argument("--output", default="data/QQQ_1d.csv", help="Output CSV path (default: data/QQQ_1d.csv)")
    args = parser.parse_args()

    download_qqq(ticker=args.ticker, years=args.years, output=args.output)


if __name__ == "__main__":
    main()