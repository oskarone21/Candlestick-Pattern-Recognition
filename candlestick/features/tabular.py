from __future__ import annotations

import numpy as np


def flatten_sequence_features(X: np.ndarray) -> np.ndarray:
    """Flatten (n, seq, features) -> (n, seq*features) for classic ML models."""
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {X.shape}")
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])
