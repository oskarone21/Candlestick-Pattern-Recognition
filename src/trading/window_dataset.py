"""Full rolling-window dataset builders for the trading product."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_full_anchor_stage1_dataset(
    df_ohlcv: pd.DataFrame,
    labeled_windows: list,
    cfg: dict,
) -> dict:
    """Create one Stage 1 sample for every valid lookback anchor bar.

    Positive labels are assigned only to anchor bars that the rule-based labeler
    marked as confirmed breakouts. Every other anchor remains a negative sample.
    Anchors that were previously sampled as rule-based negatives are still kept
    as negatives, but their origin is tracked in the metadata.
    """
    col_cfg = cfg["data_source"]["columns"]
    lookback = int(cfg["windowing"]["lookback_bars"])
    cols = [
        col_cfg["open"],
        col_cfg["high"],
        col_cfg["low"],
        col_cfg["close"],
        col_cfg["volume"],
    ]

    values = df_ohlcv[cols].to_numpy(dtype=np.float32)
    n_bars = len(values)
    if n_bars < lookback:
        raise ValueError(
            f"Need at least {lookback} bars to build rolling windows, got {n_bars}."
        )

    first_anchor = lookback - 1
    anchors = np.arange(first_anchor, n_bars, dtype=np.int64)
    y = np.zeros(len(anchors), dtype=np.int64)
    reasons = np.full(len(anchors), "background", dtype=object)
    label_sources = np.full(len(anchors), "background", dtype=object)

    for lw in labeled_windows:
        anchor = int(lw.anchor_bar)
        if anchor < first_anchor or anchor >= n_bars:
            continue
        idx = anchor - first_anchor
        if int(lw.label) == 1:
            y[idx] = 1
            reasons[idx] = lw.reason
            label_sources[idx] = "rule_positive"
            continue

        if y[idx] == 0 and label_sources[idx] == "background":
            reasons[idx] = lw.reason
            label_sources[idx] = "rule_negative"

    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=lookback, axis=0)
    windows = np.moveaxis(windows, -1, 1)

    w_min = windows.min(axis=1, keepdims=True)
    w_max = windows.max(axis=1, keepdims=True)
    denom = np.where((w_max - w_min) > 0, (w_max - w_min), 1.0)
    X = np.ascontiguousarray((windows - w_min) / denom, dtype=np.float32)

    metadata = pd.DataFrame(
        {
            "anchor_bar": anchors,
            "anchor_time": [str(df_ohlcv.index[bar]) for bar in anchors],
            "stage1_target": y,
            "label_reason": reasons,
            "label_source": label_sources,
        }
    )

    label_source_counts = metadata["label_source"].value_counts().to_dict()
    print(
        "[stage1] Full rolling dataset — "
        f"{len(y)} anchors | positives: {int(y.sum())} | negatives: {int((y == 0).sum())} | "
        f"rule_negative_anchors: {int((metadata['label_source'] == 'rule_negative').sum())}"
    )

    return {
        "X": X,
        "y": y,
        "anchors": anchors,
        "metadata": metadata,
        "summary": {
            "total_windows": int(len(y)),
            "positive_windows": int(y.sum()),
            "negative_windows": int((y == 0).sum()),
            "label_source_counts": {str(k): int(v) for k, v in label_source_counts.items()},
        },
    }
