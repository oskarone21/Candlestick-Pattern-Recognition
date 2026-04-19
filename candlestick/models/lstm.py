from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from candlestick.domain import DEFAULT_DROPOUT, DEFAULT_HIDDEN_DIM, DEFAULT_SEED, MODEL_LSTM
from candlestick.models.torch_common import TorchBinaryModel, train_torch_binary


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_layers: int = 2,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        logits = self.head(last).squeeze(-1)
        return logits


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict[str, Any],
    seed: int = 42,
    params: dict[str, Any] | None = None,
) -> TorchBinaryModel:
    params = params or {}
    init_kwargs = {
        "input_dim": int(X_train.shape[-1]),
        "hidden_dim": int(params.get("hidden_dim", 64)),
        "num_layers": int(params.get("num_layers", 2)),
        "dropout": float(params.get("dropout", cfg.get("model", {}).get("dropout", 0.2))),
    }

    return train_torch_binary(
        model_ctor=LSTMClassifier,
        model_name="lstm",
        init_kwargs=init_kwargs,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cfg=cfg,
        seed=seed,
        override_params=params,
    )
