"""Purged time-based split helpers for trading experiments."""

from __future__ import annotations

import numpy as np


def purged_time_split_indices(anchors: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """Split sorted anchors into train/val/test with an optional embargo gap.

    The embargo is expressed in bars. Validation and test start points are pushed
    forward until their first anchor is at least `embargo_bars` away from the
    previous split's last anchor.
    """
    eval_cfg = cfg["evaluation"]
    purge_cfg = cfg.get("product", {}).get("purged_split", {})

    train_ratio = eval_cfg["train_ratio"]
    val_ratio = eval_cfg["val_ratio"]
    use_purge = purge_cfg.get("enabled", False)
    embargo_bars = (
        int(purge_cfg.get("embargo_bars", cfg["windowing"]["lookback_bars"]))
        if use_purge
        else 0
    )

    anchors = np.asarray(anchors, dtype=int)
    n = len(anchors)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    raw_train_end = n_train
    raw_val_end = min(n, raw_train_end + n_val)

    train_idx = np.arange(0, raw_train_end, dtype=int)

    val_start = _advance_until_gap(
        anchors,
        start_idx=raw_train_end,
        last_anchor=anchors[train_idx[-1]] if len(train_idx) else None,
        min_gap=embargo_bars,
    )
    val_idx = np.arange(val_start, raw_val_end, dtype=int)

    test_boundary_anchor = None
    if len(val_idx):
        test_boundary_anchor = anchors[val_idx[-1]]
    elif len(train_idx):
        test_boundary_anchor = anchors[train_idx[-1]]

    test_start = _advance_until_gap(
        anchors,
        start_idx=raw_val_end,
        last_anchor=test_boundary_anchor,
        min_gap=embargo_bars,
    )
    test_idx = np.arange(test_start, n, dtype=int)

    print(
        "[split] Purged split — "
        f"train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)} | "
        f"embargo_bars: {embargo_bars}"
    )

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _advance_until_gap(
    anchors: np.ndarray,
    start_idx: int,
    last_anchor: int | None,
    min_gap: int,
) -> int:
    """Move a split start forward until the first anchor is far enough away."""
    idx = start_idx
    if last_anchor is None or min_gap <= 0:
        return idx

    while idx < len(anchors) and (anchors[idx] - last_anchor) < min_gap:
        idx += 1
    return idx

