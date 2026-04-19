from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from candlestick.domain import (
    COLUMN_SPLIT,
    COLUMN_WINDOW_END_IDX,
    COLUMN_WINDOW_END_TS,
    EPSILON_COMPARE,
    SplitName,
)


@dataclass
class SplitIndex:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def _apply_embargo(idx: np.ndarray, left: int = 0, right: int = 0) -> np.ndarray:
    start = min(max(left, 0), len(idx))
    end = len(idx) - min(max(right, 0), len(idx))
    if end < start:
        return np.array([], dtype=int)
    return idx[start:end]


def _purge_before_boundary(
    meta: pd.DataFrame,
    idx: np.ndarray,
    boundary_bar: float,
    embargo_bars: int,
) -> np.ndarray:
    if len(idx) == 0:
        return idx
    bars = pd.to_numeric(meta.loc[idx, COLUMN_WINDOW_END_IDX], errors="coerce")
    keep = (bars <= (boundary_bar - embargo_bars)).fillna(False).to_numpy()
    return idx[keep]


def _purge_after_boundary(
    meta: pd.DataFrame,
    idx: np.ndarray,
    boundary_bar: float,
    embargo_bars: int,
) -> np.ndarray:
    if len(idx) == 0:
        return idx
    bars = pd.to_numeric(meta.loc[idx, COLUMN_WINDOW_END_IDX], errors="coerce")
    keep = (bars >= (boundary_bar + embargo_bars)).fillna(False).to_numpy()
    return idx[keep]


def time_based_split(
    meta: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    embargo_bars: int = 0,
) -> SplitIndex:
    """Split by window_end_ts ordering with optional embargo to reduce leakage."""
    if meta.empty:
        empty = np.array([], dtype=int)
        return SplitIndex(train=empty, val=empty, test=empty)

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > EPSILON_COMPARE:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    meta_sorted = meta.copy()
    meta_sorted["_tmp_ts"] = pd.to_datetime(meta_sorted[COLUMN_WINDOW_END_TS], errors="coerce", utc=True)
    meta_sorted = meta_sorted.dropna(subset=["_tmp_ts"])
    if meta_sorted.empty:
        empty = np.array([], dtype=int)
        return SplitIndex(train=empty, val=empty, test=empty)
    order = meta_sorted.sort_values("_tmp_ts").index.to_numpy()

    n = len(order)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val : n_train + n_val + n_test]

    if embargo_bars > 0:
        # Prefer bar-distance purge using window_end_idx (bars), not sample counts.
        if COLUMN_WINDOW_END_IDX in meta.columns:
            val_start_bar = None
            if len(val_idx) > 0:
                val_start_bar = pd.to_numeric(meta.loc[val_idx[0], COLUMN_WINDOW_END_IDX], errors="coerce")
            test_start_bar = None
            if len(test_idx) > 0:
                test_start_bar = pd.to_numeric(meta.loc[test_idx[0], COLUMN_WINDOW_END_IDX], errors="coerce")

            if val_start_bar is not None and not pd.isna(val_start_bar):
                train_idx = _purge_before_boundary(meta, train_idx, float(val_start_bar), embargo_bars)
            else:
                train_idx = _apply_embargo(train_idx, right=embargo_bars)

            if test_start_bar is not None and not pd.isna(test_start_bar):
                test_idx = _purge_after_boundary(meta, test_idx, float(test_start_bar), embargo_bars)
            else:
                test_idx = _apply_embargo(test_idx, left=embargo_bars)
        else:
            train_idx = _apply_embargo(train_idx, right=embargo_bars)
            test_idx = _apply_embargo(test_idx, left=embargo_bars)

    return SplitIndex(
        train=np.asarray(train_idx, dtype=int),
        val=np.asarray(val_idx, dtype=int),
        test=np.asarray(test_idx, dtype=int),
    )


def assign_split_column(meta: pd.DataFrame, split: SplitIndex) -> pd.DataFrame:
    out = meta.copy()
    out[COLUMN_SPLIT] = SplitName.UNUSED.value
    out.loc[split.train, COLUMN_SPLIT] = SplitName.TRAIN.value
    out.loc[split.val, COLUMN_SPLIT] = SplitName.VAL.value
    out.loc[split.test, COLUMN_SPLIT] = SplitName.TEST.value
    return out
