from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from candlestick.models import hgb_baseline, logreg_baseline
from candlestick.models.lstm import LSTMClassifier, train_lstm
from candlestick.models.tcn import TCNClassifier, train_tcn
from candlestick.models.torch_common import load_torch_binary
from candlestick.models.transformer import TransformerClassifier, train_transformer
from candlestick.domain import (
    MODEL_HGB,
    MODEL_LOGREG,
    MODEL_LSTM,
    MODEL_TCN,
    MODEL_TRANSFORMER,
)

TrainHandler = Callable[..., Any]
PredictHandler = Callable[[Any, np.ndarray], np.ndarray]
SaveHandler = Callable[[Any, Path], None]
LoadHandler = Callable[[Path], Any]


def _save_logreg(model, path: Path) -> None:
    logreg_baseline.save_model(model, str(path))


def _save_hgb(model, path: Path) -> None:
    hgb_baseline.save_model(model, str(path))


def _save_torch_model(model, path: Path) -> None:
    model.save(path)


def _load_logreg(path: Path):
    return logreg_baseline.load_model(str(path))


def _load_hgb(path: Path):
    return hgb_baseline.load_model(str(path))


def _load_lstm(path: Path):
    return load_torch_binary(path, LSTMClassifier, device="cpu")


def _load_tcn(path: Path):
    return load_torch_binary(path, TCNClassifier, device="cpu")


def _load_transformer(path: Path):
    return load_torch_binary(path, TransformerClassifier, device="cpu")


TRAINERS: dict[str, TrainHandler] = {
    MODEL_LOGREG: lambda X_train, y_train, X_val, y_val, cfg, seed, params: logreg_baseline.train_logreg(
        X_train,
        y_train,
        seed=seed,
        params=params,
    ),
    MODEL_HGB: lambda X_train, y_train, X_val, y_val, cfg, seed, params: hgb_baseline.train_hgb(
        X_train,
        y_train,
        seed=seed,
        params=params,
    ),
    MODEL_LSTM: train_lstm,
    MODEL_TCN: train_tcn,
    MODEL_TRANSFORMER: train_transformer,
}

PREDICTORS: dict[str, PredictHandler] = {
    MODEL_LOGREG: logreg_baseline.predict_proba,
    MODEL_HGB: hgb_baseline.predict_proba,
    MODEL_LSTM: lambda model, X: model.predict_proba(X),
    MODEL_TCN: lambda model, X: model.predict_proba(X),
    MODEL_TRANSFORMER: lambda model, X: model.predict_proba(X),
}

SAVERS: dict[str, SaveHandler] = {
    MODEL_LOGREG: _save_logreg,
    MODEL_HGB: _save_hgb,
    MODEL_LSTM: _save_torch_model,
    MODEL_TCN: _save_torch_model,
    MODEL_TRANSFORMER: _save_torch_model,
}

LOADERS: dict[str, LoadHandler] = {
    MODEL_LOGREG: _load_logreg,
    MODEL_HGB: _load_hgb,
    MODEL_LSTM: _load_lstm,
    MODEL_TCN: _load_tcn,
    MODEL_TRANSFORMER: _load_transformer,
}


def _normalize_model_name(model_name: str) -> str:
    name = model_name.lower()
    if name not in TRAINERS:
        raise ValueError(f"Unknown model: {model_name}")
    return name


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
    trainer = TRAINERS[_normalize_model_name(model_name)]
    return trainer(X_train, y_train, X_val, y_val, cfg=cfg, seed=seed, params=params or {})


def predict_model_proba(model_name: str, model, X: np.ndarray) -> np.ndarray:
    predictor = PREDICTORS[_normalize_model_name(model_name)]
    return predictor(model, X)


def save_model(model_name: str, model, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    saver = SAVERS[_normalize_model_name(model_name)]
    saver(model, out)


def load_model(model_name: str, path: str | Path):
    loader = LOADERS[_normalize_model_name(model_name)]
    return loader(Path(path))
