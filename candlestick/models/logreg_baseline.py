from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logreg_model(seed: int = 42, params: dict[str, Any] | None = None) -> Pipeline:
    params = params or {}
    clf = LogisticRegression(
        random_state=seed,
        max_iter=int(params.get("max_iter", 1200)),
        C=float(params.get("C", 1.0)),
        class_weight=params.get("class_weight", "balanced"),
        solver=params.get("solver", "lbfgs"),
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def train_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    model = build_logreg_model(seed=seed, params=params)
    model.fit(X_train, y_train)
    return model


def predict_proba(model: Pipeline, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)
