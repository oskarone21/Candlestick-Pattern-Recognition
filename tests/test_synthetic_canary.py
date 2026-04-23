from __future__ import annotations

import numpy as np
import pytest

from chart_patterns.benchmarks.synthetic_canary import SYNTHETIC_CANARY_TIERS, build_synthetic_canary_dataset


@pytest.mark.parametrize(
    "pattern",
    [
        "head_shoulders",
        "inverse_head_shoulders",
        "double_top",
        "double_bottom",
    ],
)
def test_synthetic_canary_dataset_has_expected_counts(pattern):
    base = np.zeros((6, 80, 5), dtype=np.float32)
    base[:, :, 0] = 100.0
    base[:, :, 1] = 100.3
    base[:, :, 2] = 99.7
    base[:, :, 3] = 100.0
    base[:, :, 4] = 1000.0

    X, y, meta = build_synthetic_canary_dataset(base, pattern, count_per_tier=2, seed=42)

    assert X.shape == (len(SYNTHETIC_CANARY_TIERS) * 4, 80, 5)
    assert y.shape[0] == X.shape[0]
    assert int((y == 1).sum()) == len(SYNTHETIC_CANARY_TIERS) * 2
    assert int((y == 0).sum()) == len(SYNTHETIC_CANARY_TIERS) * 2
    assert set(meta["tier"]) == set(SYNTHETIC_CANARY_TIERS)
    assert set(meta["label"]) == {0, 1}
