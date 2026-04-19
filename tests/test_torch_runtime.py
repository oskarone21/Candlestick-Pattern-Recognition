from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from candlestick.models.torch_common import (
    SEQUENCE_NORMALIZATION_TRAIN_ZSCORE,
    SEQUENCE_NORMALIZATION_WINDOW_MINMAX,
    TorchBinaryModel,
    _apply_normalization,
    _fit_normalization_stats,
    _select_device,
    load_torch_binary,
    runtime_summary,
)


class TinySequenceModel(nn.Module):
    def __init__(self, input_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.mean(dim=1)).squeeze(-1)


def test_select_device_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)

    assert _select_device("auto").type == "mps"
    assert _select_device("cpu").type == "cpu"


def test_select_device_raises_for_unavailable_explicit_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        _select_device("cuda")


def test_runtime_summary_only_enables_mixed_precision_on_cuda(monkeypatch, base_cfg):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)

    summary = runtime_summary(base_cfg)

    assert summary["selected_device"] == "cpu"
    assert summary["mixed_precision_enabled"] is False


def test_sequence_normalization_uses_train_only_stats():
    X_train = np.array(
        [
            [[1.0, 10.0], [3.0, 14.0]],
            [[5.0, 18.0], [7.0, 22.0]],
        ],
        dtype=np.float32,
    )
    stats = _fit_normalization_stats(X_train)
    X_norm = _apply_normalization(X_train, stats)

    assert np.allclose(X_norm.mean(axis=(0, 1)), np.zeros(2), atol=1.0e-6)
    assert np.allclose(X_norm.std(axis=(0, 1)), np.ones(2), atol=1.0e-6)


def test_window_minmax_normalization_is_per_window_only():
    window_a = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]], dtype=np.float32)
    window_b = np.array([[100.0, 1000.0], [150.0, 1200.0], [200.0, 1500.0]], dtype=np.float32)

    normalized_pair = _apply_normalization(
        np.stack([window_a, window_b]),
        stats=None,
        mode=SEQUENCE_NORMALIZATION_WINDOW_MINMAX,
    )
    normalized_single = _apply_normalization(
        np.expand_dims(window_a, axis=0),
        stats=None,
        mode=SEQUENCE_NORMALIZATION_WINDOW_MINMAX,
    )[0]

    assert np.allclose(normalized_pair[0], normalized_single)
    assert np.allclose(normalized_pair[0].min(axis=0), np.zeros(2), atol=1.0e-6)
    assert np.allclose(normalized_pair[0].max(axis=0), np.ones(2), atol=1.0e-6)


def test_torch_checkpoint_round_trip_preserves_normalization_mode(tmp_path):
    model = TinySequenceModel(input_dim=2)
    wrapped = TorchBinaryModel(
        model=model,
        model_name="tiny-sequence",
        init_kwargs={"input_dim": 2},
        device=torch.device("cpu"),
        normalization_mode=SEQUENCE_NORMALIZATION_WINDOW_MINMAX,
        normalization_stats=None,
    )

    ckpt_path = tmp_path / "tiny-sequence.bin"
    wrapped.save(ckpt_path)
    loaded = load_torch_binary(ckpt_path, TinySequenceModel, device="cpu")

    assert loaded.normalization_mode == SEQUENCE_NORMALIZATION_WINDOW_MINMAX
    assert loaded.normalization_stats is None
    assert loaded.init_kwargs == {"input_dim": 2}


def test_torch_checkpoint_round_trip_preserves_train_zscore_stats(tmp_path):
    model = TinySequenceModel(input_dim=2)
    wrapped = TorchBinaryModel(
        model=model,
        model_name="tiny-sequence",
        init_kwargs={"input_dim": 2},
        device=torch.device("cpu"),
        normalization_mode=SEQUENCE_NORMALIZATION_TRAIN_ZSCORE,
        normalization_stats={
            "mean": np.array([[[1.0, 2.0]]], dtype=np.float32),
            "std": np.array([[[3.0, 4.0]]], dtype=np.float32),
        },
    )

    ckpt_path = tmp_path / "tiny-sequence-zscore.bin"
    wrapped.save(ckpt_path)
    loaded = load_torch_binary(ckpt_path, TinySequenceModel, device="cpu")

    assert loaded.normalization_mode == SEQUENCE_NORMALIZATION_TRAIN_ZSCORE
    assert np.allclose(loaded.normalization_stats["mean"], wrapped.normalization_stats["mean"])
    assert np.allclose(loaded.normalization_stats["std"], wrapped.normalization_stats["std"])
