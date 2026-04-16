"""Nadaraya-Watson kernel regression and extrema extraction.

Implements PATTERNS_EXPLAINED.md §1 (smoothing + extrema detection)
using config keys from ``labeling.extrema_detection``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter, lfilter_zi


# ---------------------------------------------------------------------------
# Nadaraya-Watson estimator
# ---------------------------------------------------------------------------

def gaussian_kernel(z: np.ndarray) -> np.ndarray:
    """Standard Gaussian kernel K(z) = (1/sqrt(2π)) exp(-z²/2)."""
    return np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)


def nadaraya_watson(
    t: np.ndarray,
    y: np.ndarray,
    h: float,
    t_eval: np.ndarray | None = None,
    truncate_sigma: float = 4.0,
) -> np.ndarray:
    """Nadaraya-Watson kernel regression estimator.

    Uses a truncated kernel window (default 4σ) to achieve O(n × window)
    instead of O(n²), critical for large datasets (100K+ bars).

    Parameters
    ----------
    t : array, shape (n,)
        Observed time indices (integers).
    y : array, shape (n,)
        Observed values (e.g. close prices).
    h : float
        Bandwidth.
    t_eval : array or None
        Points at which to evaluate; defaults to *t*.
    truncate_sigma : float
        Number of standard deviations beyond which kernel weights
        are treated as zero. 4.0 keeps >99.99% of total weight.

    Returns
    -------
    m_hat : array
        Smoothed values at *t_eval*.
    """
    if t_eval is None:
        t_eval = t
    n = len(t)
    m = len(t_eval)
    m_hat = np.empty(m, dtype=np.float64)

    # Truncation window in index units
    half_window = int(np.ceil(truncate_sigma * h))

    # Fast path for equally-spaced integer indices (common case)
    if n == m and np.allclose(t, t_eval):
        for i in range(m):
            lo = max(0, i - half_window)
            hi = min(n, i + half_window + 1)
            z = (t[lo:hi] - t[i]) / h
            w = np.exp(-0.5 * z ** 2)  # skip constant — cancels in ratio
            w_sum = w.sum()
            if w_sum < 1e-15:
                m_hat[i] = y[i]
            else:
                m_hat[i] = (w * y[lo:hi]).sum() / w_sum
    else:
        # General case with search
        for i in range(m):
            centre = t_eval[i]
            lo = np.searchsorted(t, centre - half_window * (t[1] - t[0]) if len(t) > 1 else centre - half_window, side="left")
            hi = np.searchsorted(t, centre + half_window * (t[1] - t[0]) if len(t) > 1 else centre + half_window, side="right")
            lo = max(0, lo)
            hi = min(n, hi)
            if hi <= lo:
                m_hat[i] = y[np.argmin(np.abs(t - centre))]
                continue
            z = (t[lo:hi] - centre) / h
            w = np.exp(-0.5 * z ** 2)
            w_sum = w.sum()
            if w_sum < 1e-15:
                m_hat[i] = y[np.argmin(np.abs(t - centre))]
            else:
                m_hat[i] = (w * y[lo:hi]).sum() / w_sum
    return m_hat


# ---------------------------------------------------------------------------
# Causal (one-sided) NW smoothing — no lookahead, FIR via lfilter
# ---------------------------------------------------------------------------

def nadaraya_watson_causal(
    y: np.ndarray,
    h: float,
) -> np.ndarray:
    """Causal one-sided NW smoothing via FIR lfilter.

    Each output point sees only past bars (no future leakage).
    Replaces the bidirectional nadaraya_watson() for extrema detection.
    """
    radius = max(1, int(4.0 * h))
    k = np.arange(radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (k / h) ** 2)
    kernel = kernel / kernel.sum()
    zi = lfilter_zi(kernel, [1.0]) * float(y[0])
    smoothed, _ = lfilter(kernel, [1.0], y.astype(np.float64), zi=zi)
    return smoothed


# ---------------------------------------------------------------------------
# AICc bandwidth selection (Hurvich, Simonoff & Tsai 1998)
# ---------------------------------------------------------------------------

def _hat_matrix_trace(t: np.ndarray, h: float) -> float:
    """Compute trace of the hat matrix H for NW with bandwidth *h*.

    Uses truncated kernel for O(n × window) performance.
    """
    n = len(t)
    trace = 0.0
    half_window = int(np.ceil(4.0 * h))
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        z = (t[lo:hi] - t[i]) / h
        w = np.exp(-0.5 * z ** 2)
        w_sum = w.sum()
        if w_sum > 1e-15:
            # Index of t[i] within the slice
            local_i = i - lo
            trace += w[local_i] / w_sum
    return trace


def aicc_score(t: np.ndarray, y: np.ndarray, h: float) -> float:
    """Corrected Akaike Information Criterion for NW bandwidth *h*."""
    n = len(t)
    m_hat = nadaraya_watson(t, y, h)
    residuals = y - m_hat
    rss = (residuals ** 2).sum()
    sigma2 = rss / n

    trace_H = _hat_matrix_trace(t, h)
    denom = 1.0 - (trace_H + 2.0) / n
    if denom <= 0:
        return np.inf
    return n * np.log(sigma2 + 1e-15) + n * (1.0 + trace_H / n) / denom


def select_bandwidth_aicc(
    t: np.ndarray,
    y: np.ndarray,
    min_bw: float,
    max_bw: float,
) -> float:
    """Select bandwidth minimising AICc over [min_bw, max_bw]."""
    result = minimize_scalar(
        lambda h: aicc_score(t, y, h),
        bounds=(min_bw, max_bw),
        method="bounded",
        options={"xatol": 0.1, "maxiter": 50},
    )
    return float(result.x)


# ---------------------------------------------------------------------------
# Extrema detection (PATTERNS_EXPLAINED §1.3)
# ---------------------------------------------------------------------------

def extract_extrema(
    t: np.ndarray,
    m_hat: np.ndarray,
    cfg_extrema: dict,
) -> list[dict]:
    """Extract alternating local extrema from smoothed series.

    Parameters
    ----------
    t : array
        Time indices corresponding to *m_hat*.
    m_hat : array
        Smoothed close prices.
    cfg_extrema : dict
        ``labeling.extrema_detection.extrema_validation`` section.

    Returns
    -------
    list of dict
        Each dict has keys ``idx`` (index into *t*), ``bar`` (bar number),
        ``value`` (smoothed price), ``type`` ("max" | "min").
    """
    eps = cfg_extrema.get("zero_crossing_epsilon", 1e-6)
    min_2nd = cfg_extrema.get("min_second_derivative_abs", 1e-7)
    min_sep = cfg_extrema.get("min_extrema_separation_bars", 2)
    enforce_alt = cfg_extrema.get("enforce_alternation", True)

    n = len(m_hat)
    # Central difference derivatives
    d1 = np.zeros(n)
    d2 = np.zeros(n)
    d1[1:-1] = (m_hat[2:] - m_hat[:-2]) / 2.0
    d2[1:-1] = m_hat[2:] - 2.0 * m_hat[1:-1] + m_hat[:-2]

    raw_extrema: list[dict] = []
    for i in range(2, n - 2):
        # Detect sign change in first derivative (zero crossing)
        sign_change = (d1[i - 1] * d1[i] < 0) or (abs(d1[i]) <= eps)
        if not sign_change:
            continue
        if d2[i] < -min_2nd:
            raw_extrema.append(
                {"idx": i, "bar": int(t[i]), "value": float(m_hat[i]), "type": "max"}
            )
        elif d2[i] > min_2nd:
            raw_extrema.append(
                {"idx": i, "bar": int(t[i]), "value": float(m_hat[i]), "type": "min"}
            )

    # Enforce minimum separation
    filtered: list[dict] = []
    for ext in raw_extrema:
        if filtered and abs(ext["bar"] - filtered[-1]["bar"]) < min_sep:
            # Keep the more extreme one
            if ext["type"] == filtered[-1]["type"]:
                if ext["type"] == "max" and ext["value"] > filtered[-1]["value"]:
                    filtered[-1] = ext
                elif ext["type"] == "min" and ext["value"] < filtered[-1]["value"]:
                    filtered[-1] = ext
                continue
        filtered.append(ext)

    # Enforce strict alternation
    if enforce_alt and len(filtered) > 1:
        alternating: list[dict] = [filtered[0]]
        for ext in filtered[1:]:
            if ext["type"] != alternating[-1]["type"]:
                alternating.append(ext)
            else:
                # Keep the more extreme value
                if ext["type"] == "max" and ext["value"] > alternating[-1]["value"]:
                    alternating[-1] = ext
                elif ext["type"] == "min" and ext["value"] < alternating[-1]["value"]:
                    alternating[-1] = ext
        filtered = alternating

    return filtered


def smooth_and_extract(
    close: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, list[dict], float]:
    """Full pipeline: smooth close prices and extract extrema.

    Parameters
    ----------
    close : array
        Raw close prices.
    cfg : dict
        Full resolved config (reads ``labeling.extrema_detection``).

    Returns
    -------
    m_hat : array
        Smoothed close.
    extrema : list[dict]
        Alternating extrema with ``idx``, ``bar``, ``value``, ``type``.
    bandwidth : float
        Selected (or fixed) bandwidth.
    """
    ed_cfg = cfg["labeling"]["extrema_detection"]
    sm_cfg = ed_cfg["smoothing"]
    bw_cfg = sm_cfg["bandwidth"]
    ev_cfg = ed_cfg["extrema_validation"]

    t = np.arange(len(close), dtype=np.float64)
    y = close.astype(np.float64)

    # Bandwidth selection
    method = bw_cfg.get("selection_method", "aicc")
    min_bw = bw_cfg.get("min_bandwidth", 2.0)
    max_bw = bw_cfg.get("max_bandwidth", 30.0)

    if method == "fixed":
        h = float(bw_cfg.get("fixed_value", 5.0))
    elif method == "aicc":
        # For performance, subsample on long series but keep enough
        # resolution to avoid over-smoothing
        max_sample = min(int(bw_cfg.get("lookback_bars", 120)), len(y))
        # Use a representative subsample with at least 200 points
        effective_sample = max(200, max_sample)
        if len(y) > effective_sample:
            idx_sub = np.linspace(0, len(y) - 1, effective_sample, dtype=int)
            h = select_bandwidth_aicc(t[idx_sub], y[idx_sub], min_bw, max_bw)
        else:
            h = select_bandwidth_aicc(t, y, min_bw, max_bw)
    else:
        h = (min_bw + max_bw) / 2.0

    m_hat = nadaraya_watson_causal(y, h)
    extrema = extract_extrema(t, m_hat, ev_cfg)

    return m_hat, extrema, h


def multi_smooth_and_extract(
    close: np.ndarray,
    bandwidths: list[float],
    cfg: dict,
) -> list[tuple[np.ndarray, list[dict], float]]:
    """Run NW smoothing + extrema extraction at multiple fixed bandwidths.

    Parameters
    ----------
    close : array
        Raw close prices.
    bandwidths : list[float]
        Fixed bandwidth values to scan at.
    cfg : dict
        Full resolved config (reads ``labeling.extrema_detection.extrema_validation``).

    Returns
    -------
    list of (m_hat, extrema, bandwidth) tuples, one per bandwidth.
    """
    ed_cfg = cfg["labeling"]["extrema_detection"]
    ev_cfg = ed_cfg["extrema_validation"]
    t = np.arange(len(close), dtype=np.float64)
    y = close.astype(np.float64)

    results = []
    for h in bandwidths:
        m_hat = nadaraya_watson_causal(y, float(h))
        extrema = extract_extrema(t, m_hat, ev_cfg)
        results.append((m_hat, extrema, float(h)))
    return results
