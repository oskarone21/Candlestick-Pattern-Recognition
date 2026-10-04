"""Guards for the SL-TCN pipeline: split leakage and metric definitions."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from candlestick.dataset import build_windows  # noqa: E402
from candlestick.evaluation import threshold_metrics  # noqa: E402

LOOKBACK = 20
CFG = {
    "project": {"seed": 0},
    "windowing": {"lookback_bars": LOOKBACK, "stride": 1},
    "labeling": {"labeling_policy": {"hard_negative_sampling": {
        "enabled": True, "target_positive_to_negative_ratio": "1:3"}}},
    "augmentation": {"enabled": False},
}


def _cand(label, extrema, breakout):
    return SimpleNamespace(label=label, extrema_indices=extrema, breakout_bar=breakout)


def test_windows_never_cross_split_boundary():
    n = 400
    ohlcv = np.tile(np.arange(n, dtype=float)[:, None], (1, 5))  # value == bar index
    bar_lo, bar_hi = 200, 300
    cands = [
        _cand(1, [190, 200, 205], 210),  # look-back starts at 191 < bar_lo -> must be dropped
        _cand(1, [240, 250, 260], 270),  # fully inside -> kept
        _cand(0, [280, 290, 305], None),  # ends after bar_hi -> must be dropped
    ]
    X, y = build_windows(ohlcv, np.zeros(n), cands, CFG, augment=False,
                         bar_lo=bar_lo, bar_hi=bar_hi)
    assert len(X) > 0
    assert X[:, :, 0].min() >= bar_lo
    assert X[:, :, 0].max() < bar_hi
    assert y.sum() == 1


def test_positive_f1_is_not_inflated_by_negatives():
    y_true = np.array([0] * 90 + [1] * 10)
    y_prob = np.r_[np.zeros(90), np.array([0.9] * 3 + [0.1] * 7)]
    m = threshold_metrics(y_true, y_prob, 0.5)
    assert m["recall"] == 0.3
    assert m["f1"] < m["f1_macro"]
