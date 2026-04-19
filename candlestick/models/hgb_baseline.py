from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


from candlestick.domain import DEFAULT_SEED


def build_hgb_model(seed: int = DEFAULT_SEED, params: dict[str, Any] | None = None) -> HistGradientBoostingClassifier:
    params = params or {}
    return HistGradientBoostingClassifier(
        random_state=seed,
        learning_rate=float(params.get("learning_rate", 0.05)),
        max_depth=None if params.get("max_depth") is None else int(params.get("max_depth")),
        max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
        min_samples_leaf=int(params.get("min_samples_leaf", 20)),
        l2_regularization=float(params.get("l2_regularization", 0.0)),
        max_iter=int(params.get("max_iter", 300)),
    )


def train_hgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = DEFAULT_SEED,
    params: dict[str, Any] | None = None,
) -> HistGradientBoostingClassifier:
    model = build_hgb_model(seed=seed, params=params)
    model.fit(X_train, y_train)
    return model


def predict_proba(model: HistGradientBoostingClassifier, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def save_model(model: HistGradientBoostingClassifier, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> HistGradientBoostingClassifier:
    return joblib.load(path)
