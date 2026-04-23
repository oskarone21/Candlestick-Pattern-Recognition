from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from chart_patterns.domain import COLUMN_CLOSE, CFG_LABELING, CFG_SMOOTHING, EPSILON_COMPARE, ExtremumKind

DEFAULT_GAUSSIAN_WINDOW_MULTIPLIER = 6
DEFAULT_MIN_GAUSSIAN_WINDOW = 5
DEFAULT_MIN_EXTREMA_SEPARATION_BARS = 2
DEFAULT_MIN_BANDWIDTH = 2.0
DEFAULT_MAX_BANDWIDTH = 30.0
DEFAULT_FIXED_BANDWIDTH = 6.0
DEFAULT_MIN_SECOND_DERIVATIVE_ABS = 1.0e-7


@dataclass
class Extremum:
    idx: int
    kind: str
    price: float


def gaussian_smooth(series: pd.Series, bandwidth: float) -> np.ndarray:
    """Smooth a series using a centered Gaussian kernel convolution."""
    values = series.to_numpy(dtype=float)
    if len(values) == 0:
        return values

    if bandwidth <= 0:
        return values.copy()

    window = int(max(DEFAULT_MIN_GAUSSIAN_WINDOW, round(bandwidth * DEFAULT_GAUSSIAN_WINDOW_MULTIPLIER)))
    if window % 2 == 0:
        window += 1

    x = np.arange(window) - window // 2
    kernel = np.exp(-(x**2) / (2 * bandwidth**2))
    kernel /= kernel.sum()

    padded = np.pad(values, pad_width=window // 2, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def gaussian_smooth_causal(series: pd.Series, bandwidth: float) -> np.ndarray:
    """Smooth a series using trailing Gaussian weights (past/current only)."""
    values = series.to_numpy(dtype=float)
    if len(values) == 0:
        return values

    if bandwidth <= 0:
        return values.copy()

    window = int(max(DEFAULT_MIN_GAUSSIAN_WINDOW, round(bandwidth * DEFAULT_GAUSSIAN_WINDOW_MULTIPLIER)))
    if window % 2 == 0:
        window += 1

    lags = np.arange(window, dtype=float)
    kernel = np.exp(-(lags**2) / (2 * bandwidth**2))
    kernel /= kernel.sum()

    smoothed = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        usable = min(i + 1, window)
        past = values[i - usable + 1 : i + 1][::-1]
        w = kernel[:usable]
        smoothed[i] = float(np.dot(past, w) / w.sum())
    return smoothed


def _merge_by_separation(extrema: list[Extremum], min_separation: int) -> list[Extremum]:
    if not extrema:
        return []

    merged: list[Extremum] = [extrema[0]]
    for ex in extrema[1:]:
        prev = merged[-1]
        if ex.idx - prev.idx < min_separation:
            if ex.kind == prev.kind:
                if ex.kind == ExtremumKind.MAX.value and ex.price > prev.price:
                    merged[-1] = ex
                elif ex.kind == ExtremumKind.MIN.value and ex.price < prev.price:
                    merged[-1] = ex
            else:
                strength_prev = abs(prev.price)
                strength_new = abs(ex.price)
                if strength_new >= strength_prev:
                    merged[-1] = ex
        else:
            merged.append(ex)
    return merged


def _enforce_alternation(extrema: list[Extremum]) -> list[Extremum]:
    if not extrema:
        return []

    out: list[Extremum] = [extrema[0]]
    for ex in extrema[1:]:
        prev = out[-1]
        if ex.kind != prev.kind:
            out.append(ex)
            continue

        if ex.kind == ExtremumKind.MAX.value and ex.price > prev.price:
            out[-1] = ex
        elif ex.kind == ExtremumKind.MIN.value and ex.price < prev.price:
            out[-1] = ex
    return out


def find_local_extrema(
    smoothed: np.ndarray,
    min_separation: int = DEFAULT_MIN_EXTREMA_SEPARATION_BARS,
    derivative_epsilon: float = 1.0e-6,
    min_second_derivative_abs: float = DEFAULT_MIN_SECOND_DERIVATIVE_ABS,
    enforce_alternation: bool = True,
) -> list[Extremum]:
    """Detect local extrema using derivative conditions."""
    if len(smoothed) < 5:
        return []

    grad = np.gradient(smoothed)
    second = np.gradient(grad)

    extrema: list[Extremum] = []
    for i in range(1, len(smoothed) - 1):
        is_zero_grad = abs(grad[i]) <= derivative_epsilon

        sign_change_max = grad[i - 1] > 0 and grad[i + 1] < 0
        sign_change_min = grad[i - 1] < 0 and grad[i + 1] > 0

        if (is_zero_grad or sign_change_max) and second[i] < -min_second_derivative_abs:
            extrema.append(Extremum(idx=i, kind=ExtremumKind.MAX.value, price=float(smoothed[i])))
        elif (is_zero_grad or sign_change_min) and second[i] > min_second_derivative_abs:
            extrema.append(Extremum(idx=i, kind=ExtremumKind.MIN.value, price=float(smoothed[i])))

    extrema = _merge_by_separation(extrema, min_separation=min_separation)
    if enforce_alternation:
        extrema = _enforce_alternation(extrema)
    return extrema


def find_local_extrema_causal(
    smoothed: np.ndarray,
    min_separation: int = DEFAULT_MIN_EXTREMA_SEPARATION_BARS,
    derivative_epsilon: float = 1.0e-6,
    enforce_alternation: bool = True,
) -> list[Extremum]:
    """Detect extrema with one-bar confirmation using only data up to current bar."""
    if len(smoothed) < 3:
        return []

    diffs = np.diff(smoothed)
    extrema: list[Extremum] = []

    for t in range(2, len(smoothed)):
        prev_diff = diffs[t - 2]
        curr_diff = diffs[t - 1]
        idx = t - 1

        if prev_diff > derivative_epsilon and curr_diff < -derivative_epsilon:
            extrema.append(Extremum(idx=idx, kind=ExtremumKind.MAX.value, price=float(smoothed[idx])))
        elif prev_diff < -derivative_epsilon and curr_diff > derivative_epsilon:
            extrema.append(Extremum(idx=idx, kind=ExtremumKind.MIN.value, price=float(smoothed[idx])))

    extrema = _merge_by_separation(extrema, min_separation=min_separation)
    if enforce_alternation:
        extrema = _enforce_alternation(extrema)
    return extrema


def detect_extrema_from_close(df: pd.DataFrame, cfg: dict) -> list[Extremum]:
    """Apply configured smoothing and extrema extraction to a symbol dataframe."""
    ext_cfg = cfg[CFG_LABELING]["extrema_detection"]
    bw_cfg = ext_cfg[CFG_SMOOTHING]["bandwidth"]

    if bw_cfg.get("selection_method", "aicc") == "aicc":
        bandwidth = float(
            bw_cfg.get(
                "default_bandwidth",
                (
                    bw_cfg.get("min_bandwidth", DEFAULT_MIN_BANDWIDTH)
                    + bw_cfg.get("max_bandwidth", DEFAULT_MAX_BANDWIDTH)
                )
                / 2,
            )
        )
    else:
        bandwidth = float(bw_cfg.get("default_bandwidth", DEFAULT_FIXED_BANDWIDTH))

    smoothing_cfg = ext_cfg.get(CFG_SMOOTHING, {})
    price_field = ext_cfg.get("price_field", COLUMN_CLOSE)
    if bool(smoothing_cfg.get("causal", True)):
        smoothed = gaussian_smooth_causal(df[price_field], bandwidth=bandwidth)
    else:
        smoothed = gaussian_smooth(df[price_field], bandwidth=bandwidth)

    val_cfg = ext_cfg.get("extrema_validation", {})
    if bool(val_cfg.get("causal", True)):
        extrema = find_local_extrema_causal(
            smoothed,
            min_separation=int(val_cfg.get("min_extrema_separation_bars", DEFAULT_MIN_EXTREMA_SEPARATION_BARS)),
            derivative_epsilon=float(val_cfg.get("zero_crossing_epsilon", EPSILON_COMPARE)),
            enforce_alternation=bool(val_cfg.get("enforce_alternation", True)),
        )
    else:
        extrema = find_local_extrema(
            smoothed,
            min_separation=int(val_cfg.get("min_extrema_separation_bars", DEFAULT_MIN_EXTREMA_SEPARATION_BARS)),
            derivative_epsilon=float(val_cfg.get("zero_crossing_epsilon", EPSILON_COMPARE)),
            min_second_derivative_abs=float(
                val_cfg.get("min_second_derivative_abs", DEFAULT_MIN_SECOND_DERIVATIVE_ABS)
            ),
            enforce_alternation=bool(val_cfg.get("enforce_alternation", True)),
        )
    return extrema
