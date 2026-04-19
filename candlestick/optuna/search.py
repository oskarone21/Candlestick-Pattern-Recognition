from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from candlestick.domain import CALIBRATED_MODEL_NAMES, CFG_MODEL_SELECTION, CFG_PROJECT, MODEL_HGB, MODEL_LSTM, MODEL_LOGREG, MODEL_TCN, MODEL_TRANSFORMER
from candlestick.eval.calibration import apply_probability_calibrator, fit_probability_calibrator
from candlestick.eval.classification import choose_threshold, evaluate_threshold_metrics, metric_value
from candlestick.models.registry import predict_model_proba, train_model


class OptunaUnavailableError(RuntimeError):
    pass


def _suggest_params(trial, model_name: str) -> dict[str, Any]:
    name = model_name.lower()
    if name == "hgb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
            "l2_regularization": trial.suggest_float("l2_regularization", 1.0e-8, 1.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 120, 400),
        }

    if name == "lstm":
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 96, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.05, 0.4),
            "lr": trial.suggest_float("lr", 1.0e-4, 5.0e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }

    if name == "tcn":
        return {
            "channels": trial.suggest_categorical("channels", [32, 64, 96, 128]),
            "levels": trial.suggest_int("levels", 2, 4),
            "kernel_size": trial.suggest_categorical("kernel_size", [2, 3, 5]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.4),
            "lr": trial.suggest_float("lr", 1.0e-4, 5.0e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }

    if name == "transformer":
        d_model = trial.suggest_categorical("d_model", [32, 64, 96, 128])
        valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]
        return {
            "d_model": d_model,
            "nhead": trial.suggest_categorical("nhead", valid_heads),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dim_feedforward": trial.suggest_categorical("dim_feedforward", [64, 128, 192, 256]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.4),
            "lr": trial.suggest_float("lr", 1.0e-4, 5.0e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }

    return {}


def _maybe_calibrate(model_name: str, y_val: np.ndarray, val_prob: np.ndarray) -> np.ndarray:
    if model_name.lower() not in {"lstm", "tcn"}:
        return val_prob
    calibrator = fit_probability_calibrator(y_true=y_val, y_prob=val_prob)
    return apply_probability_calibrator(val_prob, calibrator)


def _validation_objective_score(
    model_name: str,
    y_val: np.ndarray,
    val_prob_raw: np.ndarray,
    cfg: dict[str, Any],
) -> float:
    selection_cfg = cfg.get("model_selection", {})
    precision_floor = float(selection_cfg.get("precision_floor", 0.0))
    recall_floor = float(selection_cfg.get("recall_floor", 0.0))
    primary_metric = str(selection_cfg.get("primary_selection_metric", "f1")).lower()

    val_prob = _maybe_calibrate(model_name, y_val, val_prob_raw)
    threshold, _ = choose_threshold(
        y_true=y_val,
        y_prob=val_prob,
        precision_floor=precision_floor,
        recall_floor=recall_floor,
        primary_metric=primary_metric,
    )
    metrics = evaluate_threshold_metrics(y_val, val_prob, threshold)
    return float(metric_value(metrics, primary_metric))


def tune_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
    pattern_name: str,
) -> dict[str, Any]:
    """Run Optuna tuning and return best hyperparameters."""
    if model_name.lower() == "logreg":
        return {}

    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError as exc:
        raise OptunaUnavailableError(
            "optuna is not installed. Install it with `pip install optuna`."
        ) from exc

    opt_cfg = cfg.get("optuna", {})
    if not opt_cfg.get("enabled", True):
        return {}

    n_trials = int(opt_cfg.get("n_trials", 100))
    timeout_hours = float(opt_cfg.get("timeout_hours", 10))
    timeout_sec = int(timeout_hours * 3600)
    epochs_per_trial = int(opt_cfg.get("epochs_per_trial", 8))

    storage_path = Path(opt_cfg.get("storage_path", "outputs/optuna/studies.db"))
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path}"

    run_name = str(cfg.get("project", {}).get("run_name", "default"))
    study_name = f"{run_name}_{pattern_name}_{model_name}"

    def objective(trial):
        params = _suggest_params(trial, model_name)
        if model_name.lower() in {"lstm", "tcn", "transformer"}:
            params["epochs"] = epochs_per_trial
            params["early_stopping_patience"] = min(3, epochs_per_trial)

        model = train_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            cfg=cfg,
            seed=seed,
            params=params,
        )

        val_prob_raw = predict_model_proba(model_name, model, X_val)
        return _validation_objective_score(model_name, y_val, val_prob_raw, cfg)

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)

    best = study.best_params if study.best_trial is not None else {}
    return dict(best)
