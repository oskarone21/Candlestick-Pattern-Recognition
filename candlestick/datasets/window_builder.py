from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from candlestick.config import ensure_dir
from candlestick.domain import (
    COLUMN_LABEL,
    COLUMN_TS_EVENT,
    COLUMN_ANCHOR_IDX,
    COLUMN_WINDOW_END_IDX,
    COLUMN_WINDOW_END_TS,
    DEFAULT_POS_NEG_RATIO,
    DEFAULT_SEED,
    OHLCV_COLUMNS,
)

FEATURE_COLUMNS = list(OHLCV_COLUMNS)


def _parse_ratio(ratio: str) -> tuple[int, int]:
    try:
        a, b = ratio.split(":")
        return int(a), int(b)
    except Exception:
        return 1, 3


def sample_negatives(
    events: pd.DataFrame,
    target_ratio: str = "1:3",
    seed: int = 42,
) -> pd.DataFrame:
    """Downsample negatives to target positive:negative ratio."""
    if events.empty:
        return events

    pos = events[events["label"] == 1]
    neg = events[events["label"] == 0]
    if pos.empty or neg.empty:
        return events

    pos_n, neg_n = _parse_ratio(target_ratio)
    target_neg = max(int(len(pos) * (neg_n / max(pos_n, 1))), len(pos))

    if len(neg) <= target_neg:
        return events

    sampled_neg = neg.sample(n=target_neg, random_state=seed)
    out = pd.concat([pos, sampled_neg], axis=0).sort_values("anchor_idx").reset_index(drop=True)
    return out


def build_pattern_dataset(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    lookback_bars: int,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create fixed-length sequence windows anchored on event rows."""
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    if events.empty:
        return np.zeros((0, lookback_bars, len(feature_columns))), np.zeros((0,), dtype=int), events

    df = prices.sort_values(COLUMN_TS_EVENT).reset_index(drop=True).copy()
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    meta_rows: list[dict[str, Any]] = []

    for _, row in events.iterrows():
        anchor_idx = int(row[COLUMN_ANCHOR_IDX])
        start_idx = anchor_idx - lookback_bars + 1
        if start_idx < 0 or anchor_idx >= len(df):
            continue

        window = df.iloc[start_idx : anchor_idx + 1][feature_columns]
        if len(window) != lookback_bars:
            continue

        X_list.append(window.to_numpy(dtype=np.float32))
        y_list.append(int(row[COLUMN_LABEL]))

        meta = row.to_dict()
        meta["window_start_idx"] = start_idx
        meta[COLUMN_WINDOW_END_IDX] = anchor_idx
        meta["window_start_ts"] = str(df.iloc[start_idx][COLUMN_TS_EVENT])
        meta[COLUMN_WINDOW_END_TS] = str(df.iloc[anchor_idx][COLUMN_TS_EVENT])
        meta_rows.append(meta)

    if not X_list:
        return np.zeros((0, lookback_bars, len(feature_columns))), np.zeros((0,), dtype=int), pd.DataFrame()

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    meta_df = pd.DataFrame(meta_rows)
    return X, y, meta_df


def save_pattern_dataset(
    output_dir: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    pattern: str,
) -> Path:
    """Persist pattern dataset as NPZ plus metadata CSV/JSON manifest."""
    out_dir = ensure_dir(Path(output_dir) / pattern)

    npz_path = out_dir / "dataset.npz"
    np.savez_compressed(npz_path, X=X, y=y)

    meta_path = out_dir / "metadata.csv"
    meta.to_csv(meta_path, index=False)

    manifest = {
        "pattern": pattern,
        "samples": int(len(y)),
        "positives": int((y == 1).sum()) if len(y) else 0,
        "negatives": int((y == 0).sum()) if len(y) else 0,
        "sequence_length": int(X.shape[1]) if len(X) else 0,
        "feature_count": int(X.shape[2]) if len(X) else 0,
        "npz": str(npz_path),
        "metadata": str(meta_path),
    }

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out_dir
