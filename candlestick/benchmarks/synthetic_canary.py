from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from candlestick.datasets.window_builder import FEATURE_COLUMNS
from candlestick.domain import (
    COLUMN_CLOSE,
    COLUMN_HIGH,
    COLUMN_LOW,
    COLUMN_OPEN,
    COLUMN_VOLUME,
    COLUMN_LABEL,
    COLUMN_PATTERN,
    DEFAULT_SEED,
    PatternName,
)

SYNTHETIC_CANARY_TIERS = ("easy", "medium", "hard")


def _tier_profile(tier: str) -> dict[str, float]:
    normalized = str(tier).lower()
    if normalized == "easy":
        return {"amplitude": 1.55, "noise": 0.015, "asymmetry": 0.02, "breakout": 0.55}
    if normalized == "medium":
        return {"amplitude": 1.35, "noise": 0.03, "asymmetry": 0.06, "breakout": 0.4}
    if normalized == "hard":
        return {"amplitude": 1.15, "noise": 0.05, "asymmetry": 0.1, "breakout": 0.28}
    raise ValueError(f"Unsupported synthetic canary tier: {tier}")


def _interpolate_series(length: int, points: list[tuple[int, float]]) -> np.ndarray:
    x = np.arange(length, dtype=np.float32)
    key_x = np.asarray([idx for idx, _ in points], dtype=np.float32)
    key_y = np.asarray([value for _, value in points], dtype=np.float32)
    return np.interp(x, key_x, key_y).astype(np.float32)


def _inject_close_path(window: np.ndarray, pattern: str, tier: str, rng: np.random.Generator) -> np.ndarray:
    seq_len = int(window.shape[0])
    close = window[:, FEATURE_COLUMNS.index(COLUMN_CLOSE)].astype(np.float32)
    center_price = float(np.median(close))
    scale = float(max(np.std(close), abs(center_price) * 0.003, 0.35))
    profile = _tier_profile(tier)
    amp = np.float32(scale * profile["amplitude"])
    asym = np.float32(scale * profile["asymmetry"])
    breakout = np.float32(scale * profile["breakout"])

    points_by_pattern: dict[str, list[tuple[int, float]]] = {
        PatternName.HEAD_SHOULDERS.value: [
            (0, center_price - 0.55 * amp),
            (int(seq_len * 0.42), center_price + 0.75 * amp),
            (int(seq_len * 0.54), center_price + 0.1 * amp + asym),
            (int(seq_len * 0.66), center_price + 1.45 * amp),
            (int(seq_len * 0.78), center_price + 0.14 * amp - asym),
            (int(seq_len * 0.9), center_price + 0.68 * amp),
            (seq_len - 1, center_price - breakout),
        ],
        PatternName.INVERSE_HEAD_SHOULDERS.value: [
            (0, center_price + 0.55 * amp),
            (int(seq_len * 0.42), center_price - 0.75 * amp),
            (int(seq_len * 0.54), center_price - 0.1 * amp - asym),
            (int(seq_len * 0.66), center_price - 1.45 * amp),
            (int(seq_len * 0.78), center_price - 0.14 * amp + asym),
            (int(seq_len * 0.9), center_price - 0.68 * amp),
            (seq_len - 1, center_price + breakout),
        ],
        PatternName.DOUBLE_TOP.value: [
            (0, center_price - 0.45 * amp),
            (int(seq_len * 0.55), center_price + 1.0 * amp),
            (int(seq_len * 0.7), center_price + 0.08 * amp + asym),
            (int(seq_len * 0.86), center_price + 0.95 * amp - asym),
            (seq_len - 1, center_price - breakout),
        ],
        PatternName.DOUBLE_BOTTOM.value: [
            (0, center_price + 0.45 * amp),
            (int(seq_len * 0.55), center_price - 1.0 * amp),
            (int(seq_len * 0.7), center_price - 0.08 * amp - asym),
            (int(seq_len * 0.86), center_price - 0.95 * amp + asym),
            (seq_len - 1, center_price + breakout),
        ],
    }
    close_path = _interpolate_series(seq_len, points_by_pattern[pattern])
    noise = rng.normal(0.0, float(scale) * profile["noise"], size=seq_len).astype(np.float32)
    close_path = (close_path + noise).astype(np.float32)
    close_path[-1] = np.float32(points_by_pattern[pattern][-1][1])
    return close_path


def _shape_volume(base_volume: np.ndarray, pattern: str, seq_len: int) -> np.ndarray:
    volume = np.asarray(base_volume, dtype=np.float32).copy()
    baseline = float(max(np.median(volume), 1.0))

    if pattern == PatternName.HEAD_SHOULDERS.value:
        volume[int(seq_len * 0.42)] = baseline * 1.35
        volume[int(seq_len * 0.66)] = baseline * 1.18
        volume[int(seq_len * 0.9)] = baseline * 1.03
    elif pattern == PatternName.INVERSE_HEAD_SHOULDERS.value:
        volume[int(seq_len * 0.54)] = baseline * 1.25
        volume[int(seq_len * 0.78)] = baseline * 1.05
    else:
        volume[int(seq_len * 0.55)] = baseline * 1.3
        volume[int(seq_len * 0.86)] = baseline * 1.08

    volume[-1] = baseline * 1.65
    return np.maximum(volume, 1.0).astype(np.float32)


def inject_textbook_pattern(
    window: np.ndarray,
    pattern: str,
    tier: str,
    rng: np.random.Generator,
) -> np.ndarray:
    injected = np.asarray(window, dtype=np.float32).copy()
    seq_len = int(injected.shape[0])
    close_path = _inject_close_path(injected, pattern=pattern, tier=tier, rng=rng)

    median_range = float(np.median(np.maximum(injected[:, FEATURE_COLUMNS.index(COLUMN_HIGH)] - injected[:, FEATURE_COLUMNS.index(COLUMN_LOW)], 1.0e-3)))
    wick = max(median_range * 0.35, float(np.std(close_path)) * 0.12, 0.02)

    open_series = close_path.copy()
    open_series[1:] = close_path[:-1]
    open_series += rng.normal(0.0, wick * 0.1, size=seq_len).astype(np.float32)

    high = np.maximum(open_series, close_path) + wick
    low = np.minimum(open_series, close_path) - wick
    volume = _shape_volume(injected[:, FEATURE_COLUMNS.index(COLUMN_VOLUME)], pattern=pattern, seq_len=seq_len)

    injected[:, FEATURE_COLUMNS.index(COLUMN_OPEN)] = open_series
    injected[:, FEATURE_COLUMNS.index(COLUMN_HIGH)] = high.astype(np.float32)
    injected[:, FEATURE_COLUMNS.index(COLUMN_LOW)] = low.astype(np.float32)
    injected[:, FEATURE_COLUMNS.index(COLUMN_CLOSE)] = close_path.astype(np.float32)
    injected[:, FEATURE_COLUMNS.index(COLUMN_VOLUME)] = volume
    return injected


def build_synthetic_canary_dataset(
    negative_windows: np.ndarray,
    pattern: str,
    *,
    count_per_tier: int = 16,
    tiers: tuple[str, ...] = SYNTHETIC_CANARY_TIERS,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    windows = np.asarray(negative_windows, dtype=np.float32)
    if windows.ndim != 3 or windows.shape[0] == 0:
        raise ValueError("negative_windows must be a non-empty 3D array of shape (n, seq, features).")

    rng = np.random.default_rng(seed)
    generated_windows: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict[str, Any]] = []

    replace = windows.shape[0] < count_per_tier
    for tier in tiers:
        for example_idx in range(count_per_tier):
            source_idx = int(rng.choice(windows.shape[0], replace=replace))
            base_window = windows[source_idx].copy()
            positive_window = inject_textbook_pattern(base_window, pattern=pattern, tier=tier, rng=rng)

            generated_windows.append(positive_window)
            labels.append(1)
            rows.append(
                {
                    "pattern": pattern,
                    "tier": tier,
                    "label": 1,
                    "source_window_index": source_idx,
                    "example_index": example_idx,
                }
            )

            generated_windows.append(base_window)
            labels.append(0)
            rows.append(
                {
                    "pattern": pattern,
                    "tier": tier,
                    "label": 0,
                    "source_window_index": source_idx,
                    "example_index": example_idx,
                }
            )

    X = np.stack(generated_windows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    meta = pd.DataFrame(rows)
    return X, y, meta
