from __future__ import annotations

import numpy as np
import pytest
import torch

from candlestick.models.torch_common import _apply_normalization, _fit_normalization_stats, _select_device, runtime_summary


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
