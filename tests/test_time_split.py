from __future__ import annotations

import pandas as pd

from candlestick.split.time_split import time_based_split


def test_time_based_split_with_embargo_has_no_overlap():
    ts = pd.date_range("2024-01-01", periods=120, freq="15min")
    meta = pd.DataFrame(
        {
            "window_end_ts": ts.astype(str),
            "window_end_idx": range(120),
        }
    )

    split = time_based_split(
        meta,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        embargo_bars=5,
    )

    train_set = set(split.train.tolist())
    val_set = set(split.val.tolist())
    test_set = set(split.test.tolist())

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    assert len(split.train) > 0
    assert len(split.val) > 0
    assert len(split.test) > 0
