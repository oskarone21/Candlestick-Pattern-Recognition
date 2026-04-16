"""Windowed dataset with hard-negative sampling — PATTERNS_EXPLAINED §6–§7.

Reads windowing and labeling_policy settings from config.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .labeling import PatternCandidate


# ---------------------------------------------------------------------------
# Augmentation helpers (training-only, positive windows only)
# ---------------------------------------------------------------------------

def _augment_jitter(window: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add proportional Gaussian noise to OHLCV.

    The same price-noise factor is applied to O, H, L, C together so that
    the within-bar order (H >= O,C >= L) is preserved.  Volume gets an
    independent noise draw.
    """
    out = window.copy()
    price_noise = 1.0 + rng.normal(0.0, sigma, size=(len(window), 1))
    out[:, :4] = out[:, :4] * price_noise        # O H L C
    vol_noise = 1.0 + rng.normal(0.0, sigma * 2, size=len(window))
    out[:, 4] = np.maximum(out[:, 4] * vol_noise, 0.0)
    return out


def _augment_magnitude_scale(window: np.ndarray, scale_range: float, rng: np.random.Generator) -> np.ndarray:
    """Uniformly scale all prices by a single factor and volume independently."""
    out = window.copy()
    price_factor = 1.0 + rng.uniform(-scale_range, scale_range)
    vol_factor   = 1.0 + rng.uniform(-scale_range, scale_range)
    out[:, :4] *= price_factor
    out[:, 4]  *= vol_factor
    return out


def _augment_time_warp(window: np.ndarray, warp_factor: float, rng: np.random.Generator) -> np.ndarray:
    """Resample the sequence with a smoothly warped time axis.

    A random warp path is generated, then each column is linearly interpolated
    onto the original length so the output shape is unchanged.
    """
    T = len(window)
    # Build a monotonically increasing warp path with slight random deviations
    raw = np.cumsum(np.abs(rng.normal(1.0, warp_factor, size=T)))
    raw = raw / raw[-1] * (T - 1)               # rescale to [0, T-1]
    src = np.arange(T, dtype=np.float64)
    out = np.empty_like(window)
    for col in range(window.shape[1]):
        out[:, col] = np.interp(src, raw, window[:, col])
    return out


def augment_positive_windows(
    windows: list[np.ndarray],
    cfg: dict,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Return augmented copies of positive windows.

    Reads ``augmentation`` section from config:
      - ``enabled`` (bool)
      - ``positive_multiplier`` (int): extra copies per original window
      - ``techniques.jitter.sigma``
      - ``techniques.magnitude_scale.scale_range``
      - ``techniques.time_warp.warp_factor``
    """
    aug_cfg = cfg.get("augmentation", {})
    if not aug_cfg.get("enabled", False) or not windows:
        return []

    multiplier = aug_cfg.get("positive_multiplier", 4)
    jitter_cfg = aug_cfg.get("techniques", {}).get("jitter", {})
    scale_cfg  = aug_cfg.get("techniques", {}).get("magnitude_scale", {})
    warp_cfg   = aug_cfg.get("techniques", {}).get("time_warp", {})

    augmented: list[np.ndarray] = []
    for _ in range(multiplier):
        for w in windows:
            aug = w.copy()
            if jitter_cfg.get("enabled", True):
                aug = _augment_jitter(aug, jitter_cfg.get("sigma", 0.002), rng)
            if scale_cfg.get("enabled", True):
                aug = _augment_magnitude_scale(aug, scale_cfg.get("scale_range", 0.03), rng)
            if warp_cfg.get("enabled", True):
                aug = _augment_time_warp(aug, warp_cfg.get("warp_factor", 0.1), rng)
            augmented.append(aug)

    return augmented


def build_windows(
    ohlcv: np.ndarray,
    labels: np.ndarray,
    candidates: list[PatternCandidate],
    cfg: dict,
    augment: bool = True,
    bar_lo: int = 0,
    bar_hi: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract fixed-length OHLCV windows anchored at breakout bars.

    Parameters
    ----------
    ohlcv : array, shape (n_bars, 5)
        Columns: open, high, low, close, volume.
    labels : array, shape (n_bars,)
        Per-bar binary labels from :func:`labeling.build_label_array`.
    candidates : list[PatternCandidate]
        Candidates for this split only (positive and negative).
    cfg : dict
        Full config — uses ``windowing`` and ``labeling.labeling_policy``.
    augment : bool
        If True, generate augmented positive copies (training split only).
    bar_lo : int
        Lower bar index (inclusive) for random negative sampling.
    bar_hi : int or None
        Upper bar index (exclusive) for random negative sampling.
        Defaults to len(ohlcv).

    Returns
    -------
    X : array, shape (n_samples, lookback, 5)
    y : array, shape (n_samples,)
    """
    lookback = cfg["windowing"]["lookback_bars"]
    stride = cfg["windowing"].get("stride", 1)
    lp = cfg["labeling"]["labeling_policy"]
    hn = lp.get("hard_negative_sampling", {})
    target_ratio_str = hn.get("target_positive_to_negative_ratio", "1:3")
    pos_part, neg_part = map(int, target_ratio_str.split(":"))

    n = len(ohlcv)
    if bar_hi is None:
        bar_hi = n
    bar_hi = min(bar_hi, n)

    pos_windows: list[tuple[np.ndarray, int]] = []
    neg_windows: list[tuple[np.ndarray, int]] = []

    # --- Positive windows: anchored at breakout bar ---
    for c in candidates:
        if c.label == 1 and c.breakout_bar is not None:
            end = c.breakout_bar + 1
            start = end - lookback
            if start >= 0 and end <= n:
                window = ohlcv[start:end].copy()
                pos_windows.append((window, 1))

    # --- Hard negatives ---
    # 1) Unconfirmed pattern candidates (partial formations, failed breakouts)
    if hn.get("enabled", True):
        for c in candidates:
            if c.label == 0:
                anchor = c.extrema_indices[-1]
                end = anchor + 1
                start = end - lookback
                if start >= 0 and end <= n:
                    window = ohlcv[start:end].copy()
                    neg_windows.append((window, 0))

    # 2) Random negatives — restricted to [bar_lo, bar_hi) so no split leaks
    positive_bars = set()
    for c in candidates:
        for idx in c.extrema_indices:
            for offset in range(-lookback, lookback + 1):
                positive_bars.add(idx + offset)
        if c.breakout_bar is not None:
            for offset in range(-lookback, lookback + 1):
                positive_bars.add(c.breakout_bar + offset)

    max_neg = len(pos_windows) * neg_part // pos_part if pos_windows else 100
    needed = max(0, max_neg - len(neg_windows))

    rng = np.random.default_rng(cfg["project"]["seed"])
    available = [
        b for b in range(max(lookback, bar_lo), bar_hi)
        if b not in positive_bars
    ]
    if available and needed > 0:
        chosen = rng.choice(available, size=min(needed, len(available)), replace=False)
        for anchor in chosen:
            end = anchor + 1
            start = end - lookback
            window = ohlcv[start:end].copy()
            neg_windows.append((window, 0))

    # --- Augmentation: training split only, never val/test ---
    aug_windows: list[tuple[np.ndarray, int]] = []
    aug_cfg = cfg.get("augmentation", {})
    if augment and aug_cfg.get("enabled", False) and pos_windows:
        raw_pos = [w for w, _ in pos_windows]
        extra = augment_positive_windows(raw_pos, cfg, rng)
        aug_windows = [(w, 1) for w in extra]

    # --- Combine ---
    all_windows = pos_windows + aug_windows + neg_windows
    if not all_windows:
        return np.empty((0, lookback, 5)), np.empty((0,), dtype=np.int64)

    rng.shuffle(all_windows)
    X = np.stack([w[0] for w in all_windows])
    y = np.array([w[1] for w in all_windows], dtype=np.int64)

    return X, y


class PatternDataset(Dataset):
    """PyTorch dataset wrapping OHLCV windows and binary labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray, normalize: bool = True):
        """
        Parameters
        ----------
        X : array, shape (n, lookback, 5)
        y : array, shape (n,)
        normalize : bool
            If True, per-window z-score normalisation on each feature.
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

        if normalize and len(X) > 0:
            # Per-window, per-feature z-score
            means = self.X.mean(axis=1, keepdims=True)
            stds = self.X.std(axis=1, keepdims=True) + 1e-8
            self.X = (self.X - means) / stds

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])
