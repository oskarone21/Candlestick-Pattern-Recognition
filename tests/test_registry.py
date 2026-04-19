from __future__ import annotations

import pytest

from candlestick.domain import MODEL_HGB, MODEL_LOGREG, MODEL_LSTM, MODEL_TCN, MODEL_TRANSFORMER
from candlestick.models.registry import LOADERS, PREDICTORS, SAVERS, TRAINERS, load_model, predict_model_proba, save_model, train_model


def test_registry_maps_stay_in_sync():
    expected = {MODEL_LOGREG, MODEL_HGB, MODEL_LSTM, MODEL_TCN, MODEL_TRANSFORMER}
    assert set(TRAINERS) == expected
    assert set(PREDICTORS) == expected
    assert set(SAVERS) == expected
    assert set(LOADERS) == expected


@pytest.mark.parametrize("operation", [train_model, predict_model_proba, save_model, load_model])
def test_registry_rejects_unknown_model(operation, tmp_path):
    if operation is train_model:
        with pytest.raises(ValueError):
            operation("unknown", None, None, None, None, {}, 42)
        return

    if operation is predict_model_proba:
        with pytest.raises(ValueError):
            operation("unknown", object(), None)
        return

    if operation is load_model:
        with pytest.raises(ValueError):
            operation("unknown", tmp_path / "model.bin")
        return

    with pytest.raises(ValueError):
        operation("unknown", object(), tmp_path / "model.bin")
