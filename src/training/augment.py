"""
Training-set data augmentation for OHLCV pattern windows.

Only applied to the TRAINING set — never to validation or test sets.
Only positive samples (label=1) are augmented to address class imbalance.

Three augmentation techniques:
  1. Gaussian noise   : tiny random perturbations on all 5 features
  2. Volume jitter    : scale the volume column independently (more volatile)
  3. Price jitter     : coherent noise on OHLC columns only (keeps OHLC order)

All augmented values are clipped to [0, 1] since inputs are min-max normalised.
Augmented samples keep the same label (1) as their source window.

All parameters read from cfg['augmentation']. No values hardcoded.
Follow TEAM_STANDARDS.md: config-driven, single responsibility.
"""

import numpy as np


def augment_positives(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate augmented copies of positive training windows.

    Parameters
    ----------
    X_train : np.ndarray, shape (N, T, 5)
        Normalised OHLCV training windows.
    y_train : np.ndarray, shape (N,)
        Training labels (0 or 1).
    cfg : dict
        Full config. Reads from cfg['augmentation'].

    Returns
    -------
    X_aug : np.ndarray
        Original training windows PLUS augmented positive copies, shuffled.
    y_aug : np.ndarray
        Corresponding labels.
    """
    aug_cfg    = cfg.get("augmentation", {})
    n_copies   = aug_cfg.get("n_copies_per_positive", 4)
    noise_std  = aug_cfg.get("gaussian_noise_std", 0.012)
    vol_range  = aug_cfg.get("volume_jitter_range", 0.12)
    price_std  = aug_cfg.get("price_coherent_noise_std", 0.008)
    seed       = cfg["project"]["seed"]

    rng = np.random.default_rng(seed)

    pos_idx = np.where(y_train == 1)[0]
    n_pos   = len(pos_idx)

    if n_pos == 0 or n_copies == 0:
        return X_train, y_train

    X_new = []
    y_new = []

    for idx in pos_idx:
        window = X_train[idx]          # (T, 5) — already normalised to [0,1]

        for _ in range(n_copies):
            aug = window.copy()

            # --- 1. Coherent price noise (columns 0-3: O, H, L, C) ---
            # Add the same per-timestep noise to all four price columns so
            # the relative OHLC shape is preserved.
            price_noise = rng.normal(0.0, price_std, size=(aug.shape[0], 1))
            aug[:, :4] = aug[:, :4] + price_noise

            # --- 2. Independent Gaussian noise on all 5 features ---
            full_noise = rng.normal(0.0, noise_std, size=aug.shape)
            aug = aug + full_noise

            # --- 3. Volume jitter (column 4) ---
            vol_factor = 1.0 + rng.uniform(-vol_range, vol_range)
            aug[:, 4] = aug[:, 4] * vol_factor

            # Clip to valid normalised range
            aug = np.clip(aug, 0.0, 1.0)

            X_new.append(aug)
            y_new.append(1)

    X_new = np.array(X_new, dtype=np.float32)
    y_new = np.array(y_new, dtype=np.int64)

    # Stack original + augmented
    X_combined = np.concatenate([X_train, X_new], axis=0)
    y_combined = np.concatenate([y_train, y_new], axis=0)

    # Shuffle (within training set only)
    perm = rng.permutation(len(y_combined))
    X_combined = X_combined[perm]
    y_combined = y_combined[perm]

    n_orig_pos = int((y_train == 1).sum())
    n_total_pos = int((y_combined == 1).sum())
    print(
        f"[augment] Positive samples: {n_orig_pos} -> {n_total_pos} "
        f"({n_copies} copies each) | "
        f"Total training: {len(y_train)} -> {len(y_combined)}"
    )

    return X_combined, y_combined
