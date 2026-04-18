from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from candlestick.models.torch_common import TorchBinaryModel, train_torch_binary


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        dim_feedforward: int = 128,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model=d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = self.norm(x)
        pooled = x[:, -1, :]
        pooled = self.dropout(pooled)
        logits = self.head(pooled).squeeze(-1)
        return logits


def train_transformer(
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
        "d_model": int(params.get("d_model", 64)),
        "nhead": int(params.get("nhead", 4)),
        "num_layers": int(params.get("num_layers", 2)),
        "dropout": float(params.get("dropout", cfg.get("model", {}).get("dropout", 0.2))),
        "dim_feedforward": int(params.get("dim_feedforward", 128)),
    }

    return train_torch_binary(
        model_ctor=TransformerClassifier,
        model_name="transformer",
        init_kwargs=init_kwargs,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cfg=cfg,
        seed=seed,
        override_params=params,
    )
