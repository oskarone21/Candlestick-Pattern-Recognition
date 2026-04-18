from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from candlestick.models import hgb_baseline, logreg_baseline
from candlestick.models.lstm import train_lstm
from candlestick.models.tcn import train_tcn
from candlestick.models.transformer import train_transformer


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
    params: dict[str, Any] | None = None,
):
    name = model_name.lower()
    params = params or {}

    if name == "logreg":
        return logreg_baseline.train_logreg(X_train, y_train, seed=seed, params=params)
    if name == "hgb":
        return hgb_baseline.train_hgb(X_train, y_train, seed=seed, params=params)
    if name == "lstm":
        return train_lstm(X_train, y_train, X_val, y_val, cfg=cfg, seed=seed, params=params)
    if name == "tcn":
        return train_tcn(X_train, y_train, X_val, y_val, cfg=cfg, seed=seed, params=params)
    if name == "transformer":
        return train_transformer(X_train, y_train, X_val, y_val, cfg=cfg, seed=seed, params=params)

    raise ValueError(f"Unknown model: {model_name}")


def predict_model_proba(model_name: str, model, X: np.ndarray) -> np.ndarray:
    name = model_name.lower()
    if name == "logreg":
        return logreg_baseline.predict_proba(model, X)
    if name == "hgb":
        return hgb_baseline.predict_proba(model, X)
    if name in {"lstm", "tcn", "transformer"}:
        return model.predict_proba(X)
    raise ValueError(f"Unknown model: {model_name}")


def save_model(model_name: str, model, path: str | Path) -> None:
    name = model_name.lower()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if name == "logreg":
        logreg_baseline.save_model(model, str(out))
        return
    if name == "hgb":
        hgb_baseline.save_model(model, str(out))
        return
    if name in {"lstm", "tcn", "transformer"}:
        model.save(out)
        return
    raise ValueError(f"Unknown model: {model_name}")
