from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from chart_patterns.domain import CalibrationMethod, EPSILON_COMPARE


def _clip_probabilities(y_prob: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y_prob, dtype=float), EPSILON_COMPARE, 1.0 - EPSILON_COMPARE)


def _logit(y_prob: np.ndarray) -> np.ndarray:
    clipped = _clip_probabilities(y_prob)
    return np.log(clipped / (1.0 - clipped))


def fit_probability_calibrator(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """
    Fit a lightweight Platt-style calibrator on validation probabilities.

    If the validation labels are degenerate, return an identity calibrator.
    """

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"method": CalibrationMethod.IDENTITY.value, "fitted": False}

    features = _logit(y_prob).reshape(-1, 1)
    model = LogisticRegression(max_iter=200)
    model.fit(features, y_true)

    return {
        "method": CalibrationMethod.PLATT.value,
        "fitted": True,
        "coef": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
    }


def apply_probability_calibrator(
    y_prob: np.ndarray,
    calibrator: dict[str, Any] | None,
) -> np.ndarray:
    if not calibrator or not calibrator.get("fitted", False):
        return np.asarray(y_prob, dtype=float)

    if calibrator.get("method") != CalibrationMethod.PLATT.value:
        return np.asarray(y_prob, dtype=float)

    logits = _logit(np.asarray(y_prob, dtype=float))
    score = calibrator["coef"] * logits + calibrator["intercept"]
    return 1.0 / (1.0 + np.exp(-score))


def calibration_diagnostics(
    y_true: np.ndarray,
    raw_prob: np.ndarray,
    calibrated_prob: np.ndarray,
    calibrator: dict[str, Any] | None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    raw_prob = np.asarray(raw_prob, dtype=float)
    calibrated_prob = np.asarray(calibrated_prob, dtype=float)

    if len(y_true) == 0:
        return {
            "calibrator": calibrator or {"method": CalibrationMethod.IDENTITY.value, "fitted": False},
            "raw_brier": None,
            "calibrated_brier": None,
        }

    return {
        "calibrator": calibrator or {"method": "identity", "fitted": False},
        "raw_brier": float(brier_score_loss(y_true, raw_prob)),
        "calibrated_brier": float(brier_score_loss(y_true, calibrated_prob)),
        "mean_raw_probability": float(raw_prob.mean()),
        "mean_calibrated_probability": float(calibrated_prob.mean()),
    }
