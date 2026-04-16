"""
clean_spy_1min.py — Deduplicate and normalise 1_min_SPY_2008-2021.csv

Problems found in raw data
--------------------------
1. 638,054 exact duplicate rows  (same timestamp + same OHLCV)
2. 305,438 conflicting rows      (same timestamp, different OHLCV)
   → total 943,492 affected rows out of 2,070,834 (45.5%)
3. 33,789 zero-volume bars       (pre/post-market noise)

Dedup strategy
--------------
Step 1 — Drop exact duplicates (ts + all OHLCV identical).
Step 2 — For remaining timestamp conflicts, aggregate per 1-min bar:
         open  = first entry's open   (opening tick of that minute)
         high  = max across all rows  (true high)
         low   = min across all rows  (true low)
         close = last entry's close   (closing tick of that minute)
         volume= max across all rows  (largest reported volume wins)
Step 3 — Drop bars where volume == 0.
Step 4 — OHLCV sanity check; drop any malformed bars.

Output format (pipeline-compatible)
------------------------------------
Columns : ts_event, Open, High, Low, Close, Volume
Timezone: UTC  (timestamps stored as "2008-01-22 14:31:00+00:00")
Source TZ assumed: US/Eastern  (SPY trades NYSE, ET stamps)

Usage
-----
  python scripts/clean_spy_1min.py
  python scripts/clean_spy_1min.py --input /path/to/1_min_SPY_2008-2021.csv
  python scripts/clean_spy_1min.py --input ... --output data/spy_1min_clean.csv
  python scripts/clean_spy_1min.py --keep-zero-volume   # skip zero-vol filter
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── default paths ─────────────────────────────────────────────────────────────
_DEFAULT_INPUT  = Path.home() / ".cache/kagglehub/datasets" \
    / "gratefuldata/intraday-stock-data-1-min-sp-500-200821" \
    / "versions/1/1_min_SPY_2008-2021.csv"
_DEFAULT_OUTPUT = Path("data/spy_1min_clean.csv")

SOURCE_TIMEZONE = "US/Eastern"   # assumed; change if data proves otherwise


def _parse_args():
    p = argparse.ArgumentParser(description="Clean 1_min_SPY_2008-2021.csv")
    p.add_argument("--input",  default=str(_DEFAULT_INPUT),
                   help="Path to raw Kaggle CSV")
    p.add_argument("--output", default=str(_DEFAULT_OUTPUT),
                   help="Path for cleaned output CSV")
    p.add_argument("--keep-zero-volume", action="store_true",
                   help="Do not drop zero-volume bars")
    return p.parse_args()


def load_raw(path: str) -> pd.DataFrame:
    print(f"Loading {path} …")
    df = pd.read_csv(
        path,
        usecols=["date", "open", "high", "low", "close", "volume"],
        dtype={"open": float, "high": float, "low": float,
               "close": float, "volume": float},
    )
    print(f"  Raw rows : {len(df):,}")
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse 'YYYYMMDD  HH:MM:SS' → tz-aware UTC datetime."""
    # The raw date string uses two spaces between date and time
    df["ts"] = pd.to_datetime(df["date"], format="%Y%m%d  %H:%M:%S")
    # Localise to source timezone then convert to UTC
    df["ts"] = (
        df["ts"]
        .dt.tz_localize(SOURCE_TIMEZONE, ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )
    bad_ts = df["ts"].isna().sum()
    if bad_ts:
        print(f"  Warning: {bad_ts:,} unparseable timestamps dropped (DST fold ambiguity)")
        df = df[df["ts"].notna()].copy()
    df = df.drop(columns=["date"])
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)

    # Step 1 — exact duplicates
    df = df.drop_duplicates(subset=["ts", "open", "high", "low", "close", "volume"])
    n_after_exact = len(df)
    print(f"  After exact-dup drop : {n_after_exact:,}  "
          f"(removed {n_before - n_after_exact:,})")

    # Step 2 — conflicting timestamps → aggregate
    n_conflicts = df["ts"].duplicated(keep=False).sum()
    if n_conflicts:
        print(f"  Conflicting timestamps remaining: {n_conflicts:,} rows — aggregating …")
        df = (
            df.sort_values("ts")              # ensure chronological order within each minute
              .groupby("ts", sort=False)
              .agg(
                  open   = ("open",   "first"),
                  high   = ("high",   "max"),
                  low    = ("low",    "min"),
                  close  = ("close",  "last"),
                  volume = ("volume", "max"),
              )
              .reset_index()
        )
    print(f"  After conflict merge : {len(df):,}")
    return df


def filter_zero_volume(df: pd.DataFrame) -> pd.DataFrame:
    n = (df["volume"] == 0).sum()
    df = df[df["volume"] > 0].copy()
    print(f"  Zero-volume bars dropped: {n:,}")
    return df


def sanity_check(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any bar where OHLCV relationships are violated."""
    bad = (
        (df["high"] < df["low"]) |
        (df["close"] > df["high"]) |
        (df["close"] < df["low"]) |
        (df["open"] > df["high"]) |
        (df["open"] < df["low"]) |
        (df[["open", "high", "low", "close", "volume"]].isna().any(axis=1))
    )
    n_bad = bad.sum()
    if n_bad:
        print(f"  Warning: {n_bad:,} OHLCV-invalid bars dropped")
        df = df[~bad].copy()
    else:
        print("  OHLCV sanity : all bars valid ✓")
    return df


def rename_for_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to match pipeline format (ts_event, Open…Volume)."""
    df = df.rename(columns={
        "ts":     "ts_event",
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "volume": "Volume",
    })
    return df[["ts_event", "Open", "High", "Low", "Close", "Volume"]]


def print_summary(df: pd.DataFrame) -> None:
    print("\n── Cleaned dataset summary ──────────────────────────────")
    print(f"  Rows        : {len(df):,}")
    print(f"  Date range  : {df['ts_event'].min()}  →  {df['ts_event'].max()}")
    days = df["ts_event"].dt.date.nunique()
    print(f"  Trading days: {days:,}")
    print(f"  Avg bars/day: {len(df)/days:.0f}")
    print(f"  Price range : {df['Close'].min():.2f} – {df['Close'].max():.2f}")
    print(f"  Nulls       : {df.isnull().sum().sum()}")
    bpd = df.groupby(df["ts_event"].dt.date).size()
    print(f"  Bars/day    : min={bpd.min()}  median={bpd.median():.0f}  max={bpd.max()}")


def main():
    args = _parse_args()

    # ── load ──────────────────────────────────────────────────────────────────
    if not Path(args.input).exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = load_raw(args.input)

    # ── clean ─────────────────────────────────────────────────────────────────
    print("\nStep 1/4 — parsing timestamps …")
    df = parse_timestamps(df)

    print("\nStep 2/4 — deduplicating …")
    df = deduplicate(df)

    if not args.keep_zero_volume:
        print("\nStep 3/4 — dropping zero-volume bars …")
        df = filter_zero_volume(df)
    else:
        print("\nStep 3/4 — zero-volume filter skipped (--keep-zero-volume)")

    print("\nStep 4/4 — OHLCV sanity check …")
    df = sanity_check(df)

    # ── rename & sort ─────────────────────────────────────────────────────────
    df = rename_for_pipeline(df)
    df = df.sort_values("ts_event").reset_index(drop=True)

    # ── summary ───────────────────────────────────────────────────────────────
    print_summary(df)

    # ── save ──────────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    size_mb = out.stat().st_size / 1e6
    print(f"\nSaved → {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
