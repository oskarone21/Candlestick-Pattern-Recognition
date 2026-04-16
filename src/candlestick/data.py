"""Data loading and OHLCV resampling — TEAM_STANDARDS §3.

All parameters are read from the resolved config dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_raw_ohlcv(cfg: dict) -> pd.DataFrame:
    """Load raw 1-min OHLCV CSV as specified in *cfg['data_source']*.

    Returns a DataFrame indexed by timezone-aware datetime in the
    target timezone (``data_source.timestamp.convert_to_timezone``).
    """
    ds = cfg["data_source"]
    file_path = Path(ds["file_path"])
    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}. "
            "Place nq_1min.csv in data/ as described in README.md."
        )

    df = pd.read_csv(
        file_path,
        delimiter=ds.get("delimiter", ","),
        header=0 if ds.get("has_header", True) else None,
    )

    # --- timestamp handling (TEAM_STANDARDS §3) ---
    ts_cfg = ds["timestamp"]
    ts_col = ts_cfg["column"]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    target_tz = ts_cfg["convert_to_timezone"]
    df[ts_col] = df[ts_col].dt.tz_convert(target_tz)
    if ts_cfg.get("sort_ascending", True):
        df = df.sort_values(ts_col).reset_index(drop=True)
    df = df.set_index(ts_col)

    # --- column rename to canonical names ---
    col_map = ds["columns"]
    rename = {v: k for k, v in col_map.items()}  # CSV name → canonical
    df = df.rename(columns=rename)

    # --- quality checks ---
    quality = ds.get("quality", {})
    if quality.get("drop_duplicates", True):
        df = df[~df.index.duplicated(keep="first")]
    if quality.get("enforce_numeric_ohlcv", True):
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if quality.get("dropna_ohlcv", True):
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    return df


def resample_ohlcv(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Resample from base timeframe to target timeframe.

    Follows TEAM_STANDARDS §3 resampling rules.
    """
    rs_cfg = cfg["resampling"]
    target_tf = rs_cfg["target_timeframe"]
    agg_rules = rs_cfg["ohlcv_agg"]

    resampled = df.resample(target_tf).agg(agg_rules)

    if rs_cfg.get("drop_incomplete_bars", True):
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])

    # Volume can be zero at market close but should not be NaN
    resampled["volume"] = resampled["volume"].fillna(0).astype(np.float64)

    return resampled


def generate_synthetic_ohlcv(
    n_bars: int = 50_000,
    seed: int = 42,
    freq: str = "1min",
) -> pd.DataFrame:
    """Generate synthetic 1-min OHLCV for development without real data.

    Produces a DataFrame with the same schema expected by
    :func:`load_raw_ohlcv` after loading (datetime index, canonical
    column names, timezone-aware timestamps in America/New_York).
    """
    rng = np.random.default_rng(seed)
    # Geometric Brownian Motion for close prices with realistic volatility
    dt = 1 / (6.5 * 60 * 252)  # ~1 min in trading-year fractions
    mu, sigma = 0.05, 0.35  # higher vol for more realistic extrema density
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_bars)
    # Add mean-reverting microstructure noise (Ornstein-Uhlenbeck)
    ou_noise = np.zeros(n_bars)
    theta, ou_sigma = 0.1, 0.0003
    for i in range(1, n_bars):
        ou_noise[i] = ou_noise[i - 1] - theta * ou_noise[i - 1] + ou_sigma * rng.standard_normal()
    log_returns += ou_noise
    close = 15_000.0 * np.exp(np.cumsum(log_returns))

    # Derive OHLV from close
    noise = rng.uniform(0.0001, 0.002, size=n_bars)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.integers(500, 5000, size=n_bars).astype(float)

    # Inject synthetic patterns for testing
    close, high, low, volume = _inject_synthetic_patterns(
        close, high, low, volume, rng
    )

    # Build datetime index — weekday trading hours only
    dates = pd.bdate_range(
        start="2022-01-03", periods=n_bars // 390 + 2, freq="B"
    )
    timestamps = []
    for d in dates:
        trading_start = d + pd.Timedelta(hours=9, minutes=30)
        intraday = pd.date_range(trading_start, periods=390, freq="1min")
        timestamps.extend(intraday.tolist())
        if len(timestamps) >= n_bars:
            break
    timestamps = timestamps[:n_bars]

    idx = pd.DatetimeIndex(timestamps, tz="America/New_York")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "ts_event"
    return df


def _inject_synthetic_patterns(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inject deterministic H&S and double-top/bottom shapes for dev testing.

    Designed to survive 15-min resampling and NW smoothing while meeting
    the strict geometry + volume rules in schema_v2.
    """
    n = len(close)

    # --- Head and Shoulders (bearish) ---
    # Needs: E1(peak) E2(trough) E3(peak>E1,E5) E4(trough~E2) E5(peak~E1)
    # With symmetric shoulders and volume decay V(E1)>V(E3)>V(E5)
    hs_starts = [1500, 7000, 14000, 22000, 33000, 40000]
    for ps in hs_starts:
        pat_len = 450  # ~30 bars at 15-min after resampling
        if ps + pat_len >= n:
            continue
        base = close[ps]
        # Build shape with clear peaks and troughs
        shape = np.concatenate([
            np.linspace(0.00, 0.04, 60),     # rise to left shoulder (E1)
            np.linspace(0.04, 0.015, 45),    # dip to neckline (E2)
            np.linspace(0.015, 0.065, 75),   # rise to head (E3) — higher than E1
            np.linspace(0.065, 0.015, 75),   # back to neckline (E4 ~ E2)
            np.linspace(0.015, 0.04, 45),    # rise to right shoulder (E5 ~ E1)
            np.linspace(0.04, -0.03, 60),    # breakdown through neckline
            np.linspace(-0.03, -0.05, 90),   # continued decline + breakout confirm
        ])
        end = ps + len(shape)
        if end > n:
            shape = shape[:n - ps]
            end = n
        close[ps:end] = base * (1 + shape)
        high[ps:end] = close[ps:end] * (1 + rng.uniform(0.0005, 0.0015, size=end - ps))
        low[ps:end] = close[ps:end] * (1 - rng.uniform(0.0005, 0.0015, size=end - ps))
        # Volume decay: E1 region > E3 region > E5 region
        volume[ps:ps + 60] = rng.integers(4000, 6000, size=60)         # E1 peak region
        volume[ps + 60:ps + 105] = rng.integers(2000, 3000, size=45)   # trough
        volume[ps + 105:ps + 180] = rng.integers(3000, 4500, size=75)  # E3 (head) — less than E1
        volume[ps + 180:ps + 255] = rng.integers(2000, 3000, size=75)  # trough
        volume[ps + 255:ps + 300] = rng.integers(2000, 3000, size=45)  # E5 — less than E3
        # Breakout volume spike
        if end - ps > 360:
            volume[ps + 360:end] = rng.integers(5000, 8000, size=end - ps - 360)

    # --- Double Top (bearish) ---
    dt_starts = [4000, 11000, 18000, 28000, 37000]
    for ps in dt_starts:
        pat_len = 400
        if ps + pat_len >= n:
            continue
        base = close[ps]
        # Need prior uptrend (>15%) then two peaks at similar level
        shape = np.concatenate([
            np.linspace(-0.18, 0.0, 80),     # prior uptrend (satisfies 15% requirement)
            np.linspace(0.0, 0.05, 50),       # rise to E1 (first peak)
            np.linspace(0.05, -0.02, 60),     # pullback to E2 (trough, >5% pullback)
            np.linspace(-0.02, 0.05, 60),     # rise to E3 (second peak ~ E1)
            np.linspace(0.05, -0.08, 80),     # breakdown
            np.linspace(-0.08, -0.12, 70),    # breakout confirm below E2
        ])
        end = ps + len(shape)
        if end > n:
            shape = shape[:n - ps]
            end = n
        close[ps:end] = base * (1 + shape)
        high[ps:end] = close[ps:end] * (1 + rng.uniform(0.0005, 0.001, size=end - ps))
        low[ps:end] = close[ps:end] * (1 - rng.uniform(0.0005, 0.001, size=end - ps))
        # V(E1) > V(E3)
        volume[ps + 80:ps + 130] = rng.integers(4500, 6500, size=50)   # E1 peak
        volume[ps + 130:ps + 190] = rng.integers(2000, 3000, size=60)  # trough
        volume[ps + 190:ps + 250] = rng.integers(3000, 4000, size=60)  # E3 (less than E1)
        if end - ps > 330:
            volume[ps + 330:end] = rng.integers(5000, 8000, size=end - ps - 330)

    # --- Double Bottom (bullish) ---
    db_starts = [5500, 12500, 20000, 30000]
    for ps in db_starts:
        pat_len = 400
        if ps + pat_len >= n:
            continue
        base = close[ps]
        shape = np.concatenate([
            np.linspace(0.18, 0.0, 80),       # prior downtrend
            np.linspace(0.0, -0.05, 50),       # fall to E1 (first trough)
            np.linspace(-0.05, 0.02, 60),      # bounce to E2 (peak, >5% bounce)
            np.linspace(0.02, -0.05, 60),      # fall to E3 (second trough ~ E1)
            np.linspace(-0.05, 0.08, 80),      # rally
            np.linspace(0.08, 0.14, 70),       # breakout confirm above E2
        ])
        end = ps + len(shape)
        if end > n:
            shape = shape[:n - ps]
            end = n
        close[ps:end] = base * (1 + shape)
        high[ps:end] = close[ps:end] * (1 + rng.uniform(0.0005, 0.001, size=end - ps))
        low[ps:end] = close[ps:end] * (1 - rng.uniform(0.0005, 0.001, size=end - ps))
        volume[ps + 80:ps + 130] = rng.integers(4500, 6500, size=50)   # E1 trough
        volume[ps + 190:ps + 250] = rng.integers(3000, 4000, size=60)  # E3 (less than E1)
        if end - ps > 330:
            volume[ps + 330:end] = rng.integers(6000, 9000, size=end - ps - 330)

    return close, high, low, volume
