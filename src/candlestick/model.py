"""Model architectures — TCN (primary), LSTM, Transformer.

Architecture selection is driven by ``model.architecture`` in config.
All hyper-parameters come from config; nothing is hardcoded.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Temporal Convolutional Network (TCN)
# ---------------------------------------------------------------------------

class _CausalConv1dBlock(nn.Module):
    """Single dilated causal convolution block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, seq_len)."""
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # causal trim
        out = F.relu(self.bn1(out))
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]  # causal trim
        out = F.relu(self.bn2(out))
        out = self.dropout(out)

        res = self.residual(x)
        return F.relu(out + res)


class TCN(nn.Module):
    """Temporal Convolutional Network for sequence classification.

    Parameters are read from ``model.*`` and ``windowing.*`` config keys.
    """

    def __init__(
        self,
        input_size: int = 5,
        num_classes: int = 2,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 128, 128]

        layers: list[nn.Module] = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                _CausalConv1dBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, features) → logits (batch, num_classes)."""
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        out = self.network(x)
        return self.classifier(self.dropout(out[:, :, -1]))


# ---------------------------------------------------------------------------
# LSTM alternative
# ---------------------------------------------------------------------------

class StackedLSTM(nn.Module):
    """Stacked LSTM for sequence classification."""

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


# ---------------------------------------------------------------------------
# Transformer alternative
# ---------------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    """Lightweight Transformer encoder for sequence classification."""

    def __init__(
        self,
        input_size: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        out = self.encoder(x)
        return self.classifier(out.mean(dim=1))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> nn.Module:
    """Instantiate the model architecture specified in config.

    Reads ``model.architecture``, ``model.num_classes``, ``model.dropout``.
    """
    mcfg = cfg["model"]
    arch = mcfg.get("architecture", "tcn")
    num_classes = mcfg.get("num_classes", 2)
    dropout = mcfg.get("dropout", 0.2)
    input_size = 5  # OHLCV

    if arch == "tcn":
        return TCN(
            input_size=input_size,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif arch == "lstm":
        return StackedLSTM(
            input_size=input_size,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif arch == "transformer":
        return TransformerClassifier(
            input_size=input_size,
            num_classes=num_classes,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
