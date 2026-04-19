from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from candlestick.domain import DEFAULT_DROPOUT, DEFAULT_HIDDEN_DIM, DEFAULT_SEED, MODEL_TCN
from candlestick.models.torch_common import TorchBinaryModel, train_torch_binary


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def _trim(self, x: torch.Tensor, trim: int) -> torch.Tensor:
        return x[:, :, :-trim] if trim > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trim1 = self.conv1.padding[0]
        out = self.conv1(x)
        out = self._trim(out, trim1)
        out = self.relu(out)
        out = self.dropout(out)

        trim2 = self.conv2.padding[0]
        out = self.conv2(out)
        out = self._trim(out, trim2)
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int = 64,
        levels: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(levels):
            dilation = 2**i
            layers.append(TemporalBlock(in_ch, channels, kernel_size, dilation, dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        last = out[:, :, -1]
        last = self.dropout(last)
        logits = self.head(last).squeeze(-1)
        return logits


def train_tcn(
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
        "channels": int(params.get("channels", 64)),
        "levels": int(params.get("levels", 3)),
        "kernel_size": int(params.get("kernel_size", 3)),
        "dropout": float(params.get("dropout", cfg.get("model", {}).get("dropout", 0.2))),
    }

    return train_torch_binary(
        model_ctor=TCNClassifier,
        model_name="tcn",
        init_kwargs=init_kwargs,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cfg=cfg,
        seed=seed,
        override_params=params,
    )
