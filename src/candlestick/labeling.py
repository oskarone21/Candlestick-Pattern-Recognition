"""Pattern labeling engine — PATTERNS_EXPLAINED.md §2–§8.

Implements schema_version 2: geometry + volume + breakout confirmation
for Head & Shoulders, Inverse H&S, Double Top, Double Bottom.

All thresholds come from the resolved config dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PatternCandidate:
    """A candidate pattern found by geometric scan."""
    pattern_type: str                 # head_shoulders | inverse_head_shoulders | double_top | double_bottom
    extrema_indices: list[int]        # bar indices of the key extrema
    extrema_values: list[float]       # smoothed prices at those extrema
    neckline_fn: Any = None           # callable(t) → neckline value
    breakout_bar: int | None = None   # bar index where breakout confirmed
    label: int = 0                    # 0=negative, 1=positive
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Volume helpers
# ---------------------------------------------------------------------------

def _local_volume(volume: np.ndarray, bar: int, window: int = 3) -> float:
    """Average volume in a small window centred on *bar*."""
    lo = max(0, bar - window)
    hi = min(len(volume), bar + window + 1)
    return float(volume[lo:hi].mean())


def _rolling_mean_volume(volume: np.ndarray, bar: int, lookback: int) -> float:
    """Rolling average volume ending at *bar*."""
    lo = max(0, bar - lookback)
    return float(volume[lo:bar].mean()) if bar > lo else float(volume[:bar + 1].mean())


# ---------------------------------------------------------------------------
# Geometry checks per pattern
# ---------------------------------------------------------------------------

def _check_head_shoulders(
    extrema: list[dict],
    start: int,
    cfg_hs: dict,
    cfg_confirm: dict,
    volume: np.ndarray,
    close: np.ndarray,
) -> PatternCandidate | None:
    """Check 5 consecutive extrema starting at *start* for H&S.

    Expects: peak, trough, peak, trough, peak.
    """
    if start + 4 >= len(extrema):
        return None

    pts = extrema[start:start + 5]
    expected = ["max", "min", "max", "min", "max"]
    if [p["type"] for p in pts] != expected:
        return None

    E1, E2, E3, E4, E5 = [p["value"] for p in pts]
    t1, t2, t3, t4, t5 = [p["bar"] for p in pts]

    geom = cfg_hs["geometry"]
    sep_min = geom.get("min_peak_separation_bars", 5)
    sep_max = geom.get("max_peak_separation_bars", 40)

    # Temporal separation
    if not (sep_min <= t3 - t1 <= sep_max) or not (sep_min <= t5 - t3 <= sep_max):
        return None

    # Head prominence: E3 must be above both shoulders (with optional tolerance).
    # head_prominence_tolerance_pct=0 → strict (default).
    # e.g. 0.02 allows head to be up to 2% below a shoulder.
    head_tol = geom.get("head_prominence_tolerance_pct", 0.0)
    if E3 < E1 * (1 - head_tol) or E3 < E5 * (1 - head_tol):
        return None

    # Shoulder symmetry
    sym_tol = geom["shoulder_symmetry_tolerance_pct"]
    if abs(E1 - E5) / (0.5 * (E1 + E5)) > sym_tol:
        return None

    # Neckline-point symmetry
    nl_tol = geom["neckline_point_symmetry_tolerance_pct"]
    if abs(E2 - E4) / (0.5 * (E2 + E4)) > nl_tol:
        return None

    # Neckline slope check
    slope = (E4 - E2) / (t4 - t2) if t4 != t2 else 0.0
    max_slope = geom.get("neckline_max_slope", 0.15)
    if abs(slope) > max_slope:
        return None

    def neckline(t: float) -> float:
        return E2 + slope * (t - t2)

    # --- Volume rules (mandatory) ---
    vr = cfg_hs["volume_rules"]
    if vr.get("required", True):
        V_E1 = _local_volume(volume, t1)
        V_E3 = _local_volume(volume, t3)
        V_E5 = _local_volume(volume, t5)
        order = vr.get("peak_volume_order", "V_E1_gt_V_E3_gt_V_E5")
        if order == "V_E1_gt_V_E3_gt_V_E5":
            if not (V_E1 > V_E3 > V_E5):
                return None

    # --- Breakout confirmation ---
    beta = cfg_confirm["breakout_confirmation_pct"]
    confirm_bars = cfg_confirm.get("confirm_break_within_bars", 8)
    avg_bars = cfg_confirm.get("breakout_volume_average_bars", 20)
    vol_mult = vr.get("breakout_volume_min_multiplier", 1.2)

    breakout_bar = None
    scan_start = t5 + 1
    scan_end = min(scan_start + confirm_bars, len(close))
    for b in range(scan_start, scan_end):
        nl_val = neckline(b)
        if close[b] < nl_val * (1 - beta):
            mu_v = _rolling_mean_volume(volume, b, avg_bars)
            if volume[b] >= mu_v * vol_mult:
                breakout_bar = b
                break

    # Optional: label formation as positive even without confirmed breakout.
    # Enabled by setting labeling.head_shoulders.require_breakout_confirmation: false
    require_breakout = cfg_hs.get("require_breakout_confirmation", True)
    if not require_breakout:
        label = 1
        if breakout_bar is None:
            breakout_bar = min(t5 + 1, len(close) - 1)
    else:
        label = 1 if breakout_bar is not None else 0

    candidate = PatternCandidate(
        pattern_type="head_shoulders",
        extrema_indices=[t1, t2, t3, t4, t5],
        extrema_values=[E1, E2, E3, E4, E5],
        neckline_fn=neckline,
        breakout_bar=breakout_bar,
        label=label,
        meta={"slope": slope},
    )
    return candidate


def _scan_hs_nonconsecutive(
    extrema: list[dict],
    cfg_hs: dict,
    cfg_confirm: dict,
    volume: np.ndarray,
    close: np.ndarray,
) -> list[PatternCandidate]:
    """Scan for H&S using non-consecutive peak triplets.

    For each combination of (left_shoulder, head, right_shoulder) peaks,
    the deepest trough in each inter-peak interval serves as the neckline
    anchor.  Enabled by ``labeling.head_shoulders.non_consecutive_scan: true``.
    """
    geom = cfg_hs["geometry"]
    sep_min = geom.get("min_peak_separation_bars", 5)
    sep_max = geom.get("max_peak_separation_bars", 250)
    head_tol = geom.get("head_prominence_tolerance_pct", 0.0)
    sym_tol = geom["shoulder_symmetry_tolerance_pct"]
    nl_tol = geom["neckline_point_symmetry_tolerance_pct"]
    max_slope = geom.get("neckline_max_slope", 0.15)
    vr = cfg_hs["volume_rules"]
    beta = cfg_confirm["breakout_confirmation_pct"]
    confirm_bars_n = cfg_confirm.get("confirm_break_within_bars", 8)
    avg_bars = cfg_confirm.get("breakout_volume_average_bars", 20)
    vol_mult = vr.get("breakout_volume_min_multiplier", 1.2)
    require_breakout = cfg_hs.get("require_breakout_confirmation", True)

    peaks = [e for e in extrema if e["type"] == "max"]
    # Pre-index troughs by bar position for fast interval lookup
    troughs = [e for e in extrema if e["type"] == "min"]

    def _deepest_trough(bar_lo: int, bar_hi: int) -> dict | None:
        """Return the trough with the lowest value strictly between bar_lo and bar_hi."""
        candidates_t = [t for t in troughs if bar_lo < t["bar"] < bar_hi]
        if not candidates_t:
            return None
        return min(candidates_t, key=lambda e: e["value"])

    results: list[PatternCandidate] = []
    n_peaks = len(peaks)

    for li in range(n_peaks - 2):
        ls = peaks[li]
        t1, E1 = ls["bar"], ls["value"]

        for hi in range(li + 1, n_peaks - 1):
            head = peaks[hi]
            t3, E3 = head["bar"], head["value"]
            if not (sep_min <= t3 - t1 <= sep_max):
                if t3 - t1 > sep_max:
                    break      # farther peaks only get more distant — skip remaining
                continue
            # Head must be strictly above left shoulder (within tolerance)
            if E3 < E1 * (1 - head_tol):
                continue

            for ri in range(hi + 1, n_peaks):
                rs = peaks[ri]
                t5, E5 = rs["bar"], rs["value"]
                if not (sep_min <= t5 - t3 <= sep_max):
                    if t5 - t3 > sep_max:
                        break
                    continue
                # Head must be strictly above right shoulder (within tolerance)
                if E3 < E5 * (1 - head_tol):
                    continue

                # Neckline anchors: deepest trough in each interval
                nl1 = _deepest_trough(t1, t3)
                nl2 = _deepest_trough(t3, t5)
                if nl1 is None or nl2 is None:
                    continue
                E2, t2 = nl1["value"], nl1["bar"]
                E4, t4 = nl2["value"], nl2["bar"]

                # Shoulder symmetry
                if abs(E1 - E5) / (0.5 * (E1 + E5)) > sym_tol:
                    continue
                # Neckline point symmetry
                if abs(E2 - E4) / (0.5 * (E2 + E4)) > nl_tol:
                    continue
                # Neckline slope
                slope = (E4 - E2) / (t4 - t2) if t4 != t2 else 0.0
                if abs(slope) > max_slope:
                    continue

                # Capture loop variables for closure
                _E2, _slope, _t2 = E2, slope, t2
                def neckline(t: float, e2=_E2, s=_slope, t2_=_t2) -> float:
                    return e2 + s * (t - t2_)

                # Volume rules
                if vr.get("required", True):
                    V_E1 = _local_volume(volume, t1)
                    V_E3 = _local_volume(volume, t3)
                    V_E5 = _local_volume(volume, t5)
                    order = vr.get("peak_volume_order", "V_E1_gt_V_E3_gt_V_E5")
                    if order == "V_E1_gt_V_E3_gt_V_E5":
                        if not (V_E1 > V_E3 > V_E5):
                            continue

                # Breakout confirmation
                breakout_bar = None
                scan_start = t5 + 1
                scan_end = min(scan_start + confirm_bars_n, len(close))
                for b in range(scan_start, scan_end):
                    nl_val = neckline(b)
                    if close[b] < nl_val * (1 - beta):
                        mu_v = _rolling_mean_volume(volume, b, avg_bars)
                        if volume[b] >= mu_v * vol_mult:
                            breakout_bar = b
                            break

                if not require_breakout:
                    label = 1
                    if breakout_bar is None:
                        breakout_bar = min(t5 + 1, len(close) - 1)
                else:
                    label = 1 if breakout_bar is not None else 0

                results.append(PatternCandidate(
                    pattern_type="head_shoulders",
                    extrema_indices=[t1, t2, t3, t4, t5],
                    extrema_values=[E1, E2, E3, E4, E5],
                    neckline_fn=neckline,
                    breakout_bar=breakout_bar,
                    label=label,
                    meta={"slope": slope, "non_consecutive": True},
                ))

    return results


def _scan_ihs_nonconsecutive(
    extrema: list[dict],
    cfg_ihs: dict,
    cfg_confirm: dict,
    volume: np.ndarray,
    close: np.ndarray,
) -> list[PatternCandidate]:
    """Scan for Inverse H&S using non-consecutive trough triplets.

    Mirror of _scan_hs_nonconsecutive for bullish reversal pattern.
    Enabled by ``labeling.inverse_head_shoulders.non_consecutive_scan: true``.
    """
    geom = cfg_ihs["geometry"]
    sep_min = geom.get("min_trough_separation_bars", 5)
    sep_max = geom.get("max_trough_separation_bars", 250)
    sym_tol = geom["shoulder_symmetry_tolerance_pct"]
    nl_tol = geom["neckline_point_symmetry_tolerance_pct"]
    max_slope = geom.get("neckline_max_slope", 0.15)
    vr = cfg_ihs["volume_rules"]
    beta = cfg_confirm["breakout_confirmation_pct"]
    confirm_bars_n = cfg_confirm.get("confirm_break_within_bars", 8)
    avg_bars = cfg_confirm.get("breakout_volume_average_bars", 20)
    vol_mult = vr.get("breakout_volume_min_multiplier", 1.2)

    troughs = [e for e in extrema if e["type"] == "min"]
    peaks = [e for e in extrema if e["type"] == "max"]

    def _highest_peak(bar_lo: int, bar_hi: int) -> dict | None:
        candidates_p = [p for p in peaks if bar_lo < p["bar"] < bar_hi]
        if not candidates_p:
            return None
        return max(candidates_p, key=lambda e: e["value"])

    results: list[PatternCandidate] = []
    n_troughs = len(troughs)

    for li in range(n_troughs - 2):
        ls = troughs[li]
        t1, E1 = ls["bar"], ls["value"]

        for hi in range(li + 1, n_troughs - 1):
            head = troughs[hi]
            t3, E3 = head["bar"], head["value"]
            if not (sep_min <= t3 - t1 <= sep_max):
                if t3 - t1 > sep_max:
                    break
                continue
            # Head trough must be lower than left shoulder trough
            if E3 >= E1 or E3 >= E1:  # strict: head is lowest
                if E3 >= E1:
                    continue

            for ri in range(hi + 1, n_troughs):
                rs = troughs[ri]
                t5, E5 = rs["bar"], rs["value"]
                if not (sep_min <= t5 - t3 <= sep_max):
                    if t5 - t3 > sep_max:
                        break
                    continue
                # Head must be lower than right shoulder
                if E3 >= E5:
                    continue

                # Neckline anchors: highest peak in each interval
                nl1 = _highest_peak(t1, t3)
                nl2 = _highest_peak(t3, t5)
                if nl1 is None or nl2 is None:
                    continue
                E2, t2 = nl1["value"], nl1["bar"]
                E4, t4 = nl2["value"], nl2["bar"]

                # Shoulder symmetry
                if abs(E1 - E5) / (0.5 * (E1 + E5)) > sym_tol:
                    continue
                # Neckline point symmetry
                if abs(E2 - E4) / (0.5 * (E2 + E4)) > nl_tol:
                    continue
                # Neckline slope
                slope = (E4 - E2) / (t4 - t2) if t4 != t2 else 0.0
                if abs(slope) > max_slope:
                    continue

                _E2, _slope, _t2 = E2, slope, t2
                def neckline(t: float, e2=_E2, s=_slope, t2_=_t2) -> float:
                    return e2 + s * (t - t2_)

                if vr.get("required", True):
                    V_E2 = _local_volume(volume, t2)
                    V_E4 = _local_volume(volume, t4)
                    if not (V_E2 > V_E4):
                        continue

                breakout_bar = None
                scan_start = t5 + 1
                scan_end = min(scan_start + confirm_bars_n, len(close))
                for b in range(scan_start, scan_end):
                    nl_val = neckline(b)
                    if close[b] > nl_val * (1 + beta):
                        mu_v = _rolling_mean_volume(volume, b, avg_bars)
                        if volume[b] >= mu_v * vol_mult:
                            breakout_bar = b
                            break

                results.append(PatternCandidate(
                    pattern_type="inverse_head_shoulders",
                    extrema_indices=[t1, t2, t3, t4, t5],
                    extrema_values=[E1, E2, E3, E4, E5],
                    neckline_fn=neckline,
                    breakout_bar=breakout_bar,
                    label=1 if breakout_bar is not None else 0,
                    meta={"slope": slope, "non_consecutive": True},
                ))

    return results


def _check_inverse_head_shoulders(
    extrema: list[dict],
    start: int,
    cfg_ihs: dict,
    cfg_confirm: dict,
    volume: np.ndarray,
    close: np.ndarray,
) -> PatternCandidate | None:
    """Inverse H&S: trough, peak, trough, peak, trough."""
    if start + 4 >= len(extrema):
        return None

    pts = extrema[start:start + 5]
    expected = ["min", "max", "min", "max", "min"]
    if [p["type"] for p in pts] != expected:
        return None

    E1, E2, E3, E4, E5 = [p["value"] for p in pts]
    t1, t2, t3, t4, t5 = [p["bar"] for p in pts]

    geom = cfg_ihs["geometry"]
    sep_min = geom.get("min_trough_separation_bars", 5)
    sep_max = geom.get("max_trough_separation_bars", 40)

    if not (sep_min <= t3 - t1 <= sep_max) or not (sep_min <= t5 - t3 <= sep_max):
        return None
    if E3 >= E1 or E3 >= E5:
        return None

    sym_tol = geom["shoulder_symmetry_tolerance_pct"]
    if abs(E1 - E5) / (0.5 * (E1 + E5)) > sym_tol:
        return None

    nl_tol = geom["neckline_point_symmetry_tolerance_pct"]
    if abs(E2 - E4) / (0.5 * (E2 + E4)) > nl_tol:
        return None

    slope = (E4 - E2) / (t4 - t2) if t4 != t2 else 0.0
    if abs(slope) > geom.get("neckline_max_slope", 0.15):
        return None

    def neckline(t: float) -> float:
        return E2 + slope * (t - t2)

    vr = cfg_ihs["volume_rules"]
    if vr.get("required", True):
        V_E2 = _local_volume(volume, t2)
        V_E4 = _local_volume(volume, t4)
        if not (V_E2 > V_E4):
            return None

    beta = cfg_confirm["breakout_confirmation_pct"]
    confirm_bars = cfg_confirm.get("confirm_break_within_bars", 8)
    avg_bars = cfg_confirm.get("breakout_volume_average_bars", 20)
    vol_mult = vr.get("breakout_volume_min_multiplier", 1.2)

    breakout_bar = None
    scan_start = t5 + 1
    scan_end = min(scan_start + confirm_bars, len(close))
    for b in range(scan_start, scan_end):
        nl_val = neckline(b)
        if close[b] > nl_val * (1 + beta):
            mu_v = _rolling_mean_volume(volume, b, avg_bars)
            if volume[b] >= mu_v * vol_mult:
                breakout_bar = b
                break

    return PatternCandidate(
        pattern_type="inverse_head_shoulders",
        extrema_indices=[t1, t2, t3, t4, t5],
        extrema_values=[E1, E2, E3, E4, E5],
        neckline_fn=neckline,
        breakout_bar=breakout_bar,
        label=1 if breakout_bar is not None else 0,
        meta={"slope": slope},
    )


def _check_double_top(
    extrema: list[dict],
    start: int,
    cfg_dt: dict,
    cfg_confirm: dict,
    volume: np.ndarray,
    close: np.ndarray,
) -> PatternCandidate | None:
    """Double Top: peak, trough, peak."""
    if start + 2 >= len(extrema):
        return None

    pts = extrema[start:start + 3]
    expected = ["max", "min", "max"]
    if [p["type"] for p in pts] != expected:
        return None

    E1, E2, E3 = [p["value"] for p in pts]
    t1, t2, t3 = [p["bar"] for p in pts]

    geom = cfg_dt["geometry"]
    sep_min = geom.get("min_separation_bars", 5)
    sep_max = geom.get("max_separation_bars", 50)

    if not (sep_min <= t3 - t1 <= sep_max):
        return None

    # Peak equivalence
    if abs(E1 - E3) / (0.5 * (E1 + E3)) > geom["peak_tolerance_pct"]:
        return None

    # Pullback depth
    avg_peak = 0.5 * (E1 + E3)
    if (avg_peak - E2) / avg_peak < geom["min_pullback_pct"]:
        return None

    # Prior uptrend — look back from E1
    prior_lookback = max(0, t1 - geom.get("prior_trend_lookback_bars", 80))
    P0 = float(close[prior_lookback])
    if P0 > 0 and (E1 - P0) / P0 < geom["prior_trend_pct"]:
        return None

    vr = cfg_dt["volume_rules"]
    if vr.get("required", True):
        V_E1 = _local_volume(volume, t1)
        V_E3 = _local_volume(volume, t3)
        if not (V_E1 > V_E3):
            return None

    beta = cfg_confirm["breakout_confirmation_pct"]
    confirm_bars = cfg_confirm.get("confirm_break_within_bars", 8)
    avg_bars = cfg_confirm.get("breakout_volume_average_bars", 20)
    vol_mult = vr.get("breakout_volume_min_multiplier", 1.2)

    breakout_bar = None
    scan_start = t3 + 1
    scan_end = min(scan_start + confirm_bars, len(close))
    neckline_val = E2
    for b in range(scan_start, scan_end):
        if close[b] < neckline_val * (1 - beta):
            mu_v = _rolling_mean_volume(volume, b, avg_bars)
            if volume[b] >= mu_v * vol_mult:
                breakout_bar = b
                break

    return PatternCandidate(
        pattern_type="double_top",
        extrema_indices=[t1, t2, t3],
        extrema_values=[E1, E2, E3],
        neckline_fn=lambda t: neckline_val,
        breakout_bar=breakout_bar,
        label=1 if breakout_bar is not None else 0,
    )


def _check_double_bottom(
    extrema: list[dict],
    start: int,
    cfg_db: dict,
    cfg_confirm: dict,
    volume: np.ndarray,
    close: np.ndarray,
) -> PatternCandidate | None:
    """Double Bottom: trough, peak, trough."""
    if start + 2 >= len(extrema):
        return None

    pts = extrema[start:start + 3]
    expected = ["min", "max", "min"]
    if [p["type"] for p in pts] != expected:
        return None

    E1, E2, E3 = [p["value"] for p in pts]
    t1, t2, t3 = [p["bar"] for p in pts]

    geom = cfg_db["geometry"]
    sep_min = geom.get("min_separation_bars", 5)
    sep_max = geom.get("max_separation_bars", 50)

    if not (sep_min <= t3 - t1 <= sep_max):
        return None

    if abs(E1 - E3) / (0.5 * (E1 + E3)) > geom["trough_tolerance_pct"]:
        return None

    avg_trough = 0.5 * (E1 + E3)
    if (E2 - avg_trough) / avg_trough < geom["min_bounce_pct"]:
        return None

    prior_lookback = max(0, t1 - geom.get("prior_trend_lookback_bars", 80))
    P0 = float(close[prior_lookback])
    if P0 > 0 and (P0 - E1) / P0 < geom["prior_trend_pct"]:
        return None

    vr = cfg_db["volume_rules"]
    if vr.get("required", True):
        V_E1 = _local_volume(volume, t1)
        V_E3 = _local_volume(volume, t3)
        if not (V_E1 > V_E3):
            return None

    beta = cfg_confirm["breakout_confirmation_pct"]
    confirm_bars = cfg_confirm.get("confirm_break_within_bars", 8)
    avg_bars = cfg_confirm.get("breakout_volume_average_bars", 20)
    vol_mult = vr.get("breakout_volume_min_multiplier", 1.5)

    breakout_bar = None
    scan_start = t3 + 1
    scan_end = min(scan_start + confirm_bars, len(close))
    neckline_val = E2
    for b in range(scan_start, scan_end):
        if close[b] > neckline_val * (1 + beta):
            mu_v = _rolling_mean_volume(volume, b, avg_bars)
            if volume[b] >= mu_v * vol_mult:
                breakout_bar = b
                break

    return PatternCandidate(
        pattern_type="double_bottom",
        extrema_indices=[t1, t2, t3],
        extrema_values=[E1, E2, E3],
        neckline_fn=lambda t: neckline_val,
        breakout_bar=breakout_bar,
        label=1 if breakout_bar is not None else 0,
    )


# ---------------------------------------------------------------------------
# Non-consecutive scan for Double Top / Double Bottom
# ---------------------------------------------------------------------------

def _scan_dt_nonconsecutive(
    extrema: list[dict],
    cfg_dt: dict,
    cfg_confirm: dict,
    volume: np.ndarray,
    close: np.ndarray,
) -> list[PatternCandidate]:
    """Scan for Double Top using non-consecutive peak pairs.

    For each pair (P1, P2) of peaks (not necessarily adjacent in the extrema
    list), the highest trough strictly between them serves as the neckline.
    Enabled by ``labeling.double_top.non_consecutive_scan: true``.
    """
    geom = cfg_dt["geometry"]
    sep_min = geom.get("min_separation_bars", 5)
    sep_max = geom.get("max_separation_bars", 250)
    peak_tol = geom["peak_tolerance_pct"]
    pullback = geom["min_pullback_pct"]
    prior_trend = geom["prior_trend_pct"]
    prior_lb = geom.get("prior_trend_lookback_bars", 80)
    vr = cfg_dt["volume_rules"]
    beta = cfg_confirm["breakout_confirmation_pct"]
    confirm_bars_n = cfg_confirm.get("confirm_break_within_bars", 8)
    avg_bars = cfg_confirm.get("breakout_volume_average_bars", 20)
    vol_mult = vr.get("breakout_volume_min_multiplier", 1.2)

    peaks = [e for e in extrema if e["type"] == "max"]
    troughs = [e for e in extrema if e["type"] == "min"]

    def _highest_trough(bar_lo: int, bar_hi: int) -> dict | None:
        cands = [t for t in troughs if bar_lo < t["bar"] < bar_hi]
        if not cands:
            return None
        return max(cands, key=lambda e: e["value"])

    results: list[PatternCandidate] = []
    n_peaks = len(peaks)

    for i in range(n_peaks - 1):
        p1 = peaks[i]
        t1, E1 = p1["bar"], p1["value"]

        # Prior uptrend check from P1
        prior_lookback = max(0, t1 - prior_lb)
        P0 = float(close[prior_lookback])
        if P0 > 0 and (E1 - P0) / P0 < prior_trend:
            continue

        for j in range(i + 1, n_peaks):
            p2 = peaks[j]
            t3, E3 = p2["bar"], p2["value"]

            if not (sep_min <= t3 - t1 <= sep_max):
                if t3 - t1 > sep_max:
                    break
                continue

            # Peak equivalence
            if abs(E1 - E3) / (0.5 * (E1 + E3)) > peak_tol:
                continue

            # Neckline: highest trough between the two peaks
            trough = _highest_trough(t1, t3)
            if trough is None:
                continue
            t2, E2 = trough["bar"], trough["value"]

            # Pullback depth
            avg_peak = 0.5 * (E1 + E3)
            if (avg_peak - E2) / avg_peak < pullback:
                continue

            # Volume: first peak volume > second peak volume
            if vr.get("required", True):
                if not (_local_volume(volume, t1) > _local_volume(volume, t3)):
                    continue

            # Breakout confirmation
            neckline_val = E2
            breakout_bar = None
            scan_start = t3 + 1
            scan_end = min(scan_start + confirm_bars_n, len(close))
            for b in range(scan_start, scan_end):
                if close[b] < neckline_val * (1 - beta):
                    mu_v = _rolling_mean_volume(volume, b, avg_bars)
                    if volume[b] >= mu_v * vol_mult:
                        breakout_bar = b
                        break

            results.append(PatternCandidate(
                pattern_type="double_top",
                extrema_indices=[t1, t2, t3],
                extrema_values=[E1, E2, E3],
                neckline_fn=lambda t, nv=neckline_val: nv,
                breakout_bar=breakout_bar,
                label=1 if breakout_bar is not None else 0,
                meta={"non_consecutive": True},
            ))

    return results


def _scan_db_nonconsecutive(
    extrema: list[dict],
    cfg_db: dict,
    cfg_confirm: dict,
    volume: np.ndarray,
    close: np.ndarray,
) -> list[PatternCandidate]:
    """Scan for Double Bottom using non-consecutive trough pairs.

    Mirror of _scan_dt_nonconsecutive for bullish reversal pattern.
    Enabled by ``labeling.double_bottom.non_consecutive_scan: true``.
    """
    geom = cfg_db["geometry"]
    sep_min = geom.get("min_separation_bars", 5)
    sep_max = geom.get("max_separation_bars", 250)
    trough_tol = geom["trough_tolerance_pct"]
    bounce = geom["min_bounce_pct"]
    prior_trend = geom["prior_trend_pct"]
    prior_lb = geom.get("prior_trend_lookback_bars", 80)
    vr = cfg_db["volume_rules"]
    beta = cfg_confirm["breakout_confirmation_pct"]
    confirm_bars_n = cfg_confirm.get("confirm_break_within_bars", 8)
    avg_bars = cfg_confirm.get("breakout_volume_average_bars", 20)
    vol_mult = vr.get("breakout_volume_min_multiplier", 1.5)

    troughs = [e for e in extrema if e["type"] == "min"]
    peaks = [e for e in extrema if e["type"] == "max"]

    def _highest_peak(bar_lo: int, bar_hi: int) -> dict | None:
        cands = [p for p in peaks if bar_lo < p["bar"] < bar_hi]
        if not cands:
            return None
        return max(cands, key=lambda e: e["value"])

    results: list[PatternCandidate] = []
    n_troughs = len(troughs)

    for i in range(n_troughs - 1):
        t1_obj = troughs[i]
        t1, E1 = t1_obj["bar"], t1_obj["value"]

        # Prior downtrend check from T1
        prior_lookback = max(0, t1 - prior_lb)
        P0 = float(close[prior_lookback])
        if P0 > 0 and (P0 - E1) / P0 < prior_trend:
            continue

        for j in range(i + 1, n_troughs):
            t3_obj = troughs[j]
            t3, E3 = t3_obj["bar"], t3_obj["value"]

            if not (sep_min <= t3 - t1 <= sep_max):
                if t3 - t1 > sep_max:
                    break
                continue

            # Trough equivalence
            if abs(E1 - E3) / (0.5 * (E1 + E3)) > trough_tol:
                continue

            # Neckline: highest peak between the two troughs
            peak = _highest_peak(t1, t3)
            if peak is None:
                continue
            t2, E2 = peak["bar"], peak["value"]

            # Bounce depth
            avg_trough = 0.5 * (E1 + E3)
            if (E2 - avg_trough) / avg_trough < bounce:
                continue

            # Volume: first trough volume > second trough volume
            if vr.get("required", True):
                if not (_local_volume(volume, t1) > _local_volume(volume, t3)):
                    continue

            # Breakout confirmation
            neckline_val = E2
            breakout_bar = None
            scan_start = t3 + 1
            scan_end = min(scan_start + confirm_bars_n, len(close))
            for b in range(scan_start, scan_end):
                if close[b] > neckline_val * (1 + beta):
                    mu_v = _rolling_mean_volume(volume, b, avg_bars)
                    if volume[b] >= mu_v * vol_mult:
                        breakout_bar = b
                        break

            results.append(PatternCandidate(
                pattern_type="double_bottom",
                extrema_indices=[t1, t2, t3],
                extrema_values=[E1, E2, E3],
                neckline_fn=lambda t, nv=neckline_val: nv,
                breakout_bar=breakout_bar,
                label=1 if breakout_bar is not None else 0,
                meta={"non_consecutive": True},
            ))

    return results


# ---------------------------------------------------------------------------
# Main labeling driver
# ---------------------------------------------------------------------------

_PATTERN_CHECKERS = {
    "head_shoulders": _check_head_shoulders,
    "inverse_head_shoulders": _check_inverse_head_shoulders,
    "double_top": _check_double_top,
    "double_bottom": _check_double_bottom,
}

_EXTREMA_COUNT = {
    "head_shoulders": 5,
    "inverse_head_shoulders": 5,
    "double_top": 3,
    "double_bottom": 3,
}


def scan_patterns(
    extrema: list[dict],
    close: np.ndarray,
    volume: np.ndarray,
    cfg: dict,
    pattern_type: str | None = None,
) -> list[PatternCandidate]:
    """Scan extrema sequence for pattern candidates.

    Parameters
    ----------
    extrema : list[dict]
        Output of :func:`smoothing.extract_extrema`.
    close : array
        Raw (un-smoothed) close prices.
    volume : array
        Raw volume.
    cfg : dict
        Full resolved config.
    pattern_type : str or None
        If given, only scan for this pattern; otherwise scan for all.

    Returns
    -------
    list[PatternCandidate]
        Both confirmed (label=1) and unconfirmed (label=0) candidates.
    """
    labeling = cfg["labeling"]
    confirm_cfg = labeling["confirmation"]
    patterns = [pattern_type] if pattern_type else list(_PATTERN_CHECKERS.keys())

    candidates: list[PatternCandidate] = []

    for ptype in patterns:
        pat_cfg = labeling[ptype]

        # Non-consecutive scan for H&S / IHS if enabled in config
        if ptype == "head_shoulders" and pat_cfg.get("non_consecutive_scan", False):
            candidates.extend(
                _scan_hs_nonconsecutive(extrema, pat_cfg, confirm_cfg, volume, close)
            )
            continue

        if ptype == "inverse_head_shoulders" and pat_cfg.get("non_consecutive_scan", False):
            candidates.extend(
                _scan_ihs_nonconsecutive(extrema, pat_cfg, confirm_cfg, volume, close)
            )
            continue

        if ptype == "double_top" and pat_cfg.get("non_consecutive_scan", False):
            candidates.extend(
                _scan_dt_nonconsecutive(extrema, pat_cfg, confirm_cfg, volume, close)
            )
            continue

        if ptype == "double_bottom" and pat_cfg.get("non_consecutive_scan", False):
            candidates.extend(
                _scan_db_nonconsecutive(extrema, pat_cfg, confirm_cfg, volume, close)
            )
            continue

        checker = _PATTERN_CHECKERS[ptype]
        window_size = _EXTREMA_COUNT[ptype]
        for i in range(len(extrema) - window_size + 1):
            c = checker(extrema, i, pat_cfg, confirm_cfg, volume, close)
            if c is not None:
                candidates.append(c)

    return candidates


def dedup_candidates(
    candidates: list[PatternCandidate],
    proximity_bars: int = 80,
) -> list[PatternCandidate]:
    """Remove near-duplicate candidates of the same pattern type.

    Two candidates are considered duplicates when they share the same
    ``pattern_type`` and their anchor bars (breakout bar, or last extremum
    if unconfirmed) fall within *proximity_bars* of each other.

    When two candidates clash, the confirmed one (label=1) is kept; if both
    have the same label, the earlier-arriving one is kept.
    """
    if not candidates:
        return candidates

    def _anchor(c: PatternCandidate) -> int:
        return c.breakout_bar if c.breakout_bar is not None else c.extrema_indices[-1]

    by_type: dict[str, list[PatternCandidate]] = {}
    for c in candidates:
        by_type.setdefault(c.pattern_type, []).append(c)

    result: list[PatternCandidate] = []
    for group in by_type.values():
        group_sorted = sorted(group, key=_anchor)
        kept = [group_sorted[0]]
        for c in group_sorted[1:]:
            last = kept[-1]
            if abs(_anchor(c) - _anchor(last)) < proximity_bars:
                if c.label > last.label:
                    kept[-1] = c
            else:
                kept.append(c)
        result.extend(kept)

    return result


def filter_nested_candidates(
    candidates: list[PatternCandidate],
) -> list[PatternCandidate]:
    """Remove candidates whose bar span is fully contained within another of the same type.

    When candidate A is nested inside B (same pattern type, A.start >= B.start
    and A.end <= B.end):
    - Remove A, keep B (outer span wins).
    - Exception: if A is confirmed (label=1) and B is not (label=0), remove B
      and keep A instead.

    Controlled by ``labeling.multi_bandwidth.filter_nested`` config flag.
    """
    if len(candidates) < 2:
        return candidates

    def _span(c: PatternCandidate) -> tuple[int, int]:
        end = c.breakout_bar if c.breakout_bar is not None else c.extrema_indices[-1]
        return c.extrema_indices[0], end

    by_type: dict[str, list[PatternCandidate]] = {}
    for c in candidates:
        by_type.setdefault(c.pattern_type, []).append(c)

    result: list[PatternCandidate] = []
    for group in by_type.values():
        n = len(group)
        remove = [False] * n
        for i in range(n):
            if remove[i]:
                continue
            si_s, si_e = _span(group[i])
            for j in range(i + 1, n):
                if remove[j]:
                    continue
                sj_s, sj_e = _span(group[j])
                # j nested in i
                if si_s <= sj_s and sj_e <= si_e:
                    if group[j].label > group[i].label:
                        remove[i] = True
                        break
                    else:
                        remove[j] = True
                # i nested in j
                elif sj_s <= si_s and si_e <= sj_e:
                    if group[i].label > group[j].label:
                        remove[j] = True
                    else:
                        remove[i] = True
                        break
        result.extend(c for c, rm in zip(group, remove) if not rm)

    return result


def multi_bandwidth_scan(
    close: np.ndarray,
    volume: np.ndarray,
    cfg: dict,
    pattern_type: str | None = None,
) -> list[PatternCandidate]:
    """Scan patterns at multiple bandwidths and deduplicate overlapping detections.

    Reads ``labeling.multi_bandwidth`` from config:
      - ``enabled`` (bool): if False, falls back to single-bandwidth scan
      - ``bandwidths`` (list[float]): bandwidths to scan at
      - ``dedup_proximity_bars`` (int): anchor-bar proximity threshold for dedup

    Returns combined + deduplicated candidate list.
    """
    from .smoothing import smooth_and_extract, multi_smooth_and_extract

    mb_cfg = cfg["labeling"].get("multi_bandwidth", {})
    if not mb_cfg.get("enabled", False):
        _, extrema, _ = smooth_and_extract(close, cfg)
        return scan_patterns(extrema, close, volume, cfg, pattern_type=pattern_type)

    bandwidths = mb_cfg.get("bandwidths", [3, 5, 8, 10, 15])
    proximity_bars = mb_cfg.get("dedup_proximity_bars", 80)

    all_candidates: list[PatternCandidate] = []
    for _m_hat, extrema, _h in multi_smooth_and_extract(close, bandwidths, cfg):
        candidates = scan_patterns(extrema, close, volume, cfg, pattern_type=pattern_type)
        all_candidates.extend(candidates)

    result = dedup_candidates(all_candidates, proximity_bars)
    if mb_cfg.get("filter_nested", False):
        result = filter_nested_candidates(result)
    return result


def diagnose_double_pattern(
    extrema: list[dict],
    close: np.ndarray,
    volume: np.ndarray,
    cfg: dict,
    pattern_type: str = "double_top",
) -> pd.DataFrame:
    """Return a DataFrame explaining why each 3-extrema window passed or failed.

    Useful for debugging why Double Top / Double Bottom produce 0 candidates.
    ``pattern_type`` must be ``double_top`` or ``double_bottom``.
    """
    assert pattern_type in ("double_top", "double_bottom"), pattern_type
    labeling = cfg["labeling"]
    confirm_cfg = labeling["confirmation"]
    pat_cfg = labeling[pattern_type]
    geom = pat_cfg["geometry"]
    vr = pat_cfg["volume_rules"]
    beta = confirm_cfg["breakout_confirmation_pct"]
    confirm_bars = confirm_cfg.get("confirm_break_within_bars", 8)
    avg_bars = confirm_cfg.get("breakout_volume_average_bars", 20)

    is_top = pattern_type == "double_top"
    expected = ["max", "min", "max"] if is_top else ["min", "max", "min"]
    sep_min = geom.get("min_separation_bars", 5)
    sep_max = geom.get("max_separation_bars", 50)

    rows = []
    for i in range(len(extrema) - 2):
        pts = extrema[i:i + 3]
        types = [p["type"] for p in pts]
        if types != expected:
            continue

        E1, E2, E3 = [p["value"] for p in pts]
        t1, t2, t3 = [p["bar"] for p in pts]

        row: dict = {"i": i, "t1": t1, "t2": t2, "t3": t3}

        sep_ok = sep_min <= t3 - t1 <= sep_max
        row["sep_ok"] = sep_ok
        row["t3-t1"] = t3 - t1

        tol_key = "peak_tolerance_pct" if is_top else "trough_tolerance_pct"
        tol = geom[tol_key]
        equiv_ok = abs(E1 - E3) / (0.5 * (E1 + E3)) <= tol
        row["equiv_ok"] = equiv_ok
        row["equiv_ratio"] = abs(E1 - E3) / (0.5 * (E1 + E3))

        pb_key = "min_pullback_pct" if is_top else "min_bounce_pct"
        pb = geom[pb_key]
        avg_ext = 0.5 * (E1 + E3)
        depth = (avg_ext - E2) / avg_ext if is_top else (E2 - avg_ext) / avg_ext
        pb_ok = depth >= pb
        row["pullback_ok"] = pb_ok
        row["depth_ratio"] = depth

        prior_lookback = max(0, t1 - geom.get("prior_trend_lookback_bars", 80))
        P0 = float(close[prior_lookback])
        trend = (E1 - P0) / P0 if is_top else (P0 - E1) / P0
        trend_ok = trend >= geom["prior_trend_pct"]
        row["trend_ok"] = trend_ok
        row["trend_ratio"] = trend

        V_E1 = _local_volume(volume, t1)
        V_E3 = _local_volume(volume, t3)
        vol_ok = V_E1 > V_E3 if vr.get("required", True) else True
        row["vol_ok"] = vol_ok

        # Breakout scan
        neckline_val = E2
        breakout_found = False
        scan_end = min(t3 + 1 + confirm_bars, len(close))
        for b in range(t3 + 1, scan_end):
            condition = close[b] < neckline_val * (1 - beta) if is_top else close[b] > neckline_val * (1 + beta)
            if condition:
                mu_v = _rolling_mean_volume(volume, b, avg_bars)
                vol_mult = vr.get("breakout_volume_min_multiplier", 1.2)
                if volume[b] >= mu_v * vol_mult:
                    breakout_found = True
                    break
        row["breakout_ok"] = breakout_found
        row["pass"] = all([sep_ok, equiv_ok, pb_ok, trend_ok, vol_ok, breakout_found])
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_label_array(
    n_bars: int,
    candidates: list[PatternCandidate],
    cfg: dict,
) -> np.ndarray:
    """Build per-bar binary label array from confirmed candidates.

    Positive label only at the breakout bar (labeling_policy §6).
    """
    labels = np.zeros(n_bars, dtype=np.int64)
    pos_label = cfg["labeling"]["classes"]["positive_label"]

    for c in candidates:
        if c.label == 1 and c.breakout_bar is not None:
            if 0 <= c.breakout_bar < n_bars:
                labels[c.breakout_bar] = pos_label

    return labels
