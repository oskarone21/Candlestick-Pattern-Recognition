from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from candlestick.config import ensure_dir, load_config
from candlestick.datasets.window_builder import build_pattern_dataset, save_pattern_dataset
from candlestick.eval.calibration import (
    apply_probability_calibrator,
    calibration_diagnostics,
    fit_probability_calibrator,
)
from candlestick.eval.classification import (
    attach_confidence_intervals,
    choose_threshold,
    evaluate_threshold_metrics,
    metric_value,
    save_metrics_report,
)
from candlestick.features.tabular import flatten_sequence_features
from candlestick.labeling.pattern_rules import detect_pattern_events
from candlestick.models.registry import predict_model_proba, save_model, train_model
from candlestick.models.torch_common import runtime_summary
from candlestick.optuna.search import OptunaUnavailableError, tune_model
from candlestick.split.time_split import assign_split_column, time_based_split


def _parse_ratio(ratio: str) -> tuple[int, int]:
    try:
        left, right = ratio.split(":")
        return max(int(left), 1), max(int(right), 1)
    except Exception:
        return 1, 3


def _downsample_train_negatives(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_ratio: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample negatives in train only; keep validation/test untouched."""
    if len(y_train) == 0:
        return X_train, y_train

    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X_train, y_train

    pos_n, neg_n = _parse_ratio(target_ratio)
    target_neg = max(int(len(pos_idx) * (neg_n / pos_n)), len(pos_idx))
    if len(neg_idx) <= target_neg:
        return X_train, y_train

    rng = np.random.default_rng(seed)
    keep_neg = rng.choice(neg_idx, size=target_neg, replace=False)
    keep_idx = np.sort(np.concatenate([pos_idx, keep_neg]))
    return X_train[keep_idx], y_train[keep_idx]


def _generate_smoke_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 1400
    ts = pd.date_range("2024-01-02 09:30", periods=n, freq="15min", tz="America/New_York")

    trend = np.linspace(0, 3.5, n)
    wave = np.sin(np.linspace(0, 50, n)) * 1.2
    noise = rng.normal(0, 0.2, n)
    close = 100 + trend + wave + noise
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.05, n)
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.4, n)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.4, n)
    vol = rng.integers(5000, 20000, size=n)

    return pd.DataFrame(
        {
            "symbol": "SPY",
            "ts_event": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )


def _load_prices(cfg: dict[str, Any], smoke: bool) -> pd.DataFrame:
    path = Path(cfg["paths"].get("processed_15m_path", "data/processed/spy_15m.csv"))
    if smoke:
        return _generate_smoke_data()

    if not path.exists():
        raise FileNotFoundError(
            f"Processed 15m dataset not found at {path}. Run scripts/prepare_15m_dataset.py first."
        )

    df = pd.read_csv(path)
    working_timezone = cfg.get("data_source", {}).get("timestamp", {}).get(
        "convert_to_timezone", "America/New_York"
    )
    df["ts_event"] = pd.to_datetime(df["ts_event"], errors="coerce", utc=True).dt.tz_convert(working_timezone)
    return df.dropna(subset=["ts_event"]).copy()


def _load_prebuilt_pattern_dataset(
    prebuilt_root: Path | None,
    pattern: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | None:
    if prebuilt_root is None:
        return None

    pattern_dir = prebuilt_root / pattern
    npz_path = pattern_dir / "dataset.npz"
    meta_path = pattern_dir / "metadata.csv"
    if not npz_path.exists() or not meta_path.exists():
        return None

    with np.load(npz_path) as data:
        X = data["X"]
        y = data["y"]
    meta = pd.read_csv(meta_path)
    return X, y, meta


def _prepare_split(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    eval_cfg = cfg["evaluation"]
    conf = cfg["labeling"]["confirmation"]
    lookback = int(cfg["windowing"].get("lookback_bars", 80))
    confirm_horizon = int(conf.get("confirm_break_within_bars", 8))
    default_embargo = lookback + confirm_horizon
    embargo = int(cfg["evaluation"].get("embargo_bars", default_embargo))

    if bool(eval_cfg.get("enforce_min_embargo", True)):
        embargo = max(embargo, default_embargo)

    split = time_based_split(
        meta,
        train_ratio=float(eval_cfg.get("train_ratio", 0.7)),
        val_ratio=float(eval_cfg.get("val_ratio", 0.15)),
        test_ratio=float(eval_cfg.get("test_ratio", 0.15)),
        embargo_bars=embargo,
    )
    meta_split = assign_split_column(meta, split)

    if len(split.val) == 0 or len(split.test) == 0:
        if bool(eval_cfg.get("allow_zero_embargo_fallback", False)):
            split = time_based_split(
                meta,
                train_ratio=float(eval_cfg.get("train_ratio", 0.7)),
                val_ratio=float(eval_cfg.get("val_ratio", 0.15)),
                test_ratio=float(eval_cfg.get("test_ratio", 0.15)),
                embargo_bars=0,
            )
            meta_split = assign_split_column(meta, split)
        else:
            raise ValueError(
                "Split produced empty validation or test fold after embargo. "
                "Adjust ratios/embargo instead of silently disabling purge. "
                "Set evaluation.allow_zero_embargo_fallback=true only if you accept leakage risk."
            )

    if len(split.train) == 0 or len(split.val) == 0 or len(split.test) == 0:
        raise ValueError("Split has an empty fold. Adjust dataset size, ratios, or embargo settings.")

    return {
        "X_train": X[split.train],
        "y_train": y[split.train],
        "X_val": X[split.val],
        "y_val": y[split.val],
        "X_test": X[split.test],
        "y_test": y[split.test],
        "idx_train": split.train,
        "idx_val": split.val,
        "idx_test": split.test,
    }, meta_split


def _model_input_for_name(name: str, X: np.ndarray) -> np.ndarray:
    if name in {"logreg", "hgb"}:
        return flatten_sequence_features(X)
    return X


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _safe_tune(
    model_name: str,
    train_data: dict[str, np.ndarray],
    cfg: dict[str, Any],
    seed: int,
    pattern: str,
) -> dict[str, Any]:
    opt_cfg = cfg.get("optuna", {})
    tuned_models = set(opt_cfg.get("tuned_models", ["lstm", "tcn"]))
    if not opt_cfg.get("enabled", True) or model_name not in tuned_models:
        return {}

    try:
        return tune_model(
            model_name=model_name,
            X_train=train_data["X_train"],
            y_train=train_data["y_train"],
            X_val=train_data["X_val"],
            y_val=train_data["y_val"],
            cfg=cfg,
            seed=seed,
            pattern_name=pattern,
        )
    except OptunaUnavailableError:
        return {}


def _maybe_calibrate_probabilities(
    model_name: str,
    y_val: np.ndarray,
    val_prob_raw: np.ndarray,
    test_prob_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    if model_name.lower() not in {"lstm", "tcn"}:
        calibrator = {"method": "identity", "fitted": False}
        return (
            val_prob_raw,
            test_prob_raw,
            calibrator,
            calibration_diagnostics(y_val, val_prob_raw, val_prob_raw, calibrator),
        )

    calibrator = fit_probability_calibrator(y_true=y_val, y_prob=val_prob_raw)
    val_prob = apply_probability_calibrator(val_prob_raw, calibrator)
    test_prob = apply_probability_calibrator(test_prob_raw, calibrator)
    return (
        val_prob,
        test_prob,
        calibrator,
        calibration_diagnostics(y_val, val_prob_raw, val_prob, calibrator),
    )


def _select_pattern_champion(
    model_results: list[dict[str, Any]],
    primary_metric: str,
    min_val_support: int,
) -> dict[str, Any] | None:
    best_supported: dict[str, Any] | None = None
    best_any: dict[str, Any] | None = None

    for result in model_results:
        score = float(metric_value(result["val_metrics"], primary_metric))
        support_ok = result["val_metrics"]["support"]["positive"] >= min_val_support

        if best_any is None or score > float(best_any["selection_score"]):
            best_any = {**result, "selection_score": score}

        if support_ok and (best_supported is None or score > float(best_supported["selection_score"])):
            best_supported = {**result, "selection_score": score}

    return best_supported or best_any


def run_experiment_suite(cfg: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    seed = int(cfg["project"].get("seed", 42))
    np.random.seed(seed)

    run_name = cfg["project"].get("run_name", "pattern_suite")
    metrics_root = ensure_dir(Path(cfg["paths"].get("metrics_dir", "outputs/metrics")) / run_name)
    ckpt_root = ensure_dir(Path(cfg["paths"].get("checkpoints_dir", "outputs/checkpoints")) / run_name)
    datasets_root = ensure_dir(Path(cfg["paths"].get("datasets_dir", "outputs/datasets")) / run_name)
    prebuilt_root = cfg.get("paths", {}).get("prebuilt_datasets_dir")
    prebuilt_root = Path(prebuilt_root) if prebuilt_root else None

    runtime = runtime_summary(cfg)
    _save_json(metrics_root / "runtime_summary.json", runtime)
    print(
        "Runtime summary:",
        f"device={runtime['selected_device']}",
        f"cuda_available={runtime['cuda_available']}",
        f"mps_available={runtime['mps_available']}",
        f"mixed_precision={runtime['mixed_precision_enabled']}",
        f"gpu={runtime['gpu_name'] or 'n/a'}",
    )

    prices = _load_prices(cfg, smoke=smoke)
    prices = prices[prices["symbol"].astype(str).str.upper() == "SPY"].sort_values("ts_event").reset_index(drop=True)

    patterns = cfg["labeling"].get(
        "allowed_patterns",
        ["head_shoulders", "inverse_head_shoulders", "double_top", "double_bottom"],
    )
    model_names = cfg.get("model_selection", {}).get("candidate_models", ["logreg", "lstm", "tcn"])
    model_selection_cfg = cfg.get("model_selection", {})
    primary_metric = str(model_selection_cfg.get("primary_selection_metric", "f1")).lower()
    precision_floor = float(model_selection_cfg.get("precision_floor", 0.0))
    recall_floor = float(model_selection_cfg.get("recall_floor", 0.0))
    grid_size = int(cfg.get("evaluation", {}).get("threshold_grid_size", 181))
    min_support = int(cfg.get("evaluation", {}).get("minimum_test_positive_support", 5))
    min_val_support = int(cfg.get("evaluation", {}).get("minimum_validation_positive_support", min_support))

    summary_rows: list[dict[str, Any]] = []
    champion_rows: list[dict[str, Any]] = []

    for pattern in patterns:
        prebuilt = _load_prebuilt_pattern_dataset(prebuilt_root, pattern)
        if prebuilt is not None:
            X, y, meta = prebuilt
        else:
            events = detect_pattern_events(prices, pattern, cfg)
            if events.empty:
                continue

            lookback = int(cfg["windowing"].get("lookback_bars", 80))
            X, y, meta = build_pattern_dataset(prices, events, lookback_bars=lookback)
            if len(y) == 0:
                continue

        save_pattern_dataset(datasets_root, X, y, meta, pattern)

        try:
            split_data, meta_split = _prepare_split(X, y, meta, cfg)
        except ValueError as exc:
            print(f"Skipping pattern `{pattern}` due to split constraints: {exc}")
            continue

        pattern_dir = ensure_dir(metrics_root / pattern)
        meta_split.to_csv(pattern_dir / "split_metadata.csv", index=False)

        hard_cfg = cfg["labeling"].get("labeling_policy", {}).get("hard_negative_sampling", {})
        X_train_fit = split_data["X_train"]
        y_train_fit = split_data["y_train"]
        if hard_cfg.get("enabled", True):
            X_train_fit, y_train_fit = _downsample_train_negatives(
                X_train_fit,
                y_train_fit,
                target_ratio=str(hard_cfg.get("target_positive_to_negative_ratio", "1:3")),
                seed=seed,
            )

        model_results: list[dict[str, Any]] = []

        for model_name in model_names:
            X_train_m = _model_input_for_name(model_name, X_train_fit)
            X_val_m = _model_input_for_name(model_name, split_data["X_val"])
            X_test_m = _model_input_for_name(model_name, split_data["X_test"])

            if len(X_train_m) == 0 or len(X_val_m) == 0 or len(X_test_m) == 0:
                continue
            if len(np.unique(y_train_fit)) < 2:
                continue

            best_params = _safe_tune(
                model_name,
                {
                    "X_train": X_train_m,
                    "y_train": y_train_fit,
                    "X_val": X_val_m,
                    "y_val": split_data["y_val"],
                },
                cfg,
                seed,
                pattern,
            )

            model = train_model(
                model_name=model_name,
                X_train=X_train_m,
                y_train=y_train_fit,
                X_val=X_val_m,
                y_val=split_data["y_val"],
                cfg=cfg,
                seed=seed,
                params=best_params,
            )

            val_prob_raw = predict_model_proba(model_name, model, X_val_m)
            test_prob_raw = predict_model_proba(model_name, model, X_test_m)
            val_prob, test_prob, calibrator, calibration = _maybe_calibrate_probabilities(
                model_name=model_name,
                y_val=split_data["y_val"],
                val_prob_raw=val_prob_raw,
                test_prob_raw=test_prob_raw,
            )

            threshold, threshold_diag = choose_threshold(
                y_true=split_data["y_val"],
                y_prob=val_prob,
                precision_floor=precision_floor,
                recall_floor=recall_floor,
                primary_metric=primary_metric,
                grid_size=grid_size,
            )
            val_metrics = evaluate_threshold_metrics(split_data["y_val"], val_prob, threshold)
            test_metrics = evaluate_threshold_metrics(split_data["y_test"], test_prob, threshold)
            test_metrics = attach_confidence_intervals(
                test_metrics,
                split_data["y_test"],
                test_prob,
                n_bootstrap=int(cfg.get("evaluation", {}).get("bootstrap_iterations", 300)),
                seed=seed,
            )
            test_metrics["pattern"] = pattern
            test_metrics["model"] = model_name
            test_metrics["best_params"] = best_params

            model_path = ckpt_root / pattern / f"{model_name}.bin"
            save_model(model_name, model, model_path)
            postprocess_path = ckpt_root / pattern / f"{model_name}.postprocess.json"
            _save_json(postprocess_path, {"calibrator": calibrator})

            report = dict(test_metrics)
            report["artifact_path"] = str(model_path)
            report["postprocess_path"] = str(postprocess_path)
            report["selection_metric"] = primary_metric
            report["selection_value"] = float(metric_value(val_metrics, primary_metric))
            report["selection_metrics"] = val_metrics
            report["threshold_diagnostics"] = threshold_diag
            report["calibration"] = calibration
            report["runtime_summary"] = runtime

            metrics_path = pattern_dir / f"{model_name}.json"
            save_metrics_report(report, metrics_path)
            _save_json(pattern_dir / f"{model_name}_threshold_diagnostics.json", threshold_diag)
            _save_json(pattern_dir / f"{model_name}_calibration.json", calibration)

            val_pred_df = meta.iloc[split_data["idx_val"]].copy().reset_index(drop=True)
            val_pred_df["proba_raw"] = val_prob_raw
            val_pred_df["proba"] = val_prob
            val_pred_df["threshold"] = threshold
            val_pred_df["model"] = model_name
            val_pred_df["pattern"] = pattern
            val_pred_df.to_csv(pattern_dir / f"{model_name}_val_predictions.csv", index=False)

            pred_df = meta.iloc[split_data["idx_test"]].copy().reset_index(drop=True)
            pred_df["proba_raw"] = test_prob_raw
            pred_df["proba"] = test_prob
            pred_df["threshold"] = threshold
            pred_df["model"] = model_name
            pred_df["pattern"] = pattern
            pred_df.to_csv(pattern_dir / f"{model_name}_test_predictions.csv", index=False)

            selection_value = float(metric_value(val_metrics, primary_metric))
            summary_rows.append(
                {
                    "pattern": pattern,
                    "model": model_name,
                    "selection_split": "val",
                    "selection_metric": primary_metric,
                    "selection_metric_value": selection_value,
                    "selection_precision": val_metrics["precision"],
                    "selection_recall": val_metrics["recall"],
                    "selection_f1": val_metrics["f1"],
                    "selection_f2": val_metrics["f2"],
                    "f1": test_metrics["f1"],
                    "f2": test_metrics["f2"],
                    "precision": test_metrics["precision"],
                    "recall": test_metrics["recall"],
                    "pr_auc": test_metrics["pr_auc"],
                    "threshold": threshold,
                    "support_positive": test_metrics["support"]["positive"],
                    "support_negative": test_metrics["support"]["negative"],
                }
            )

            model_results.append(
                {
                    "pattern": pattern,
                    "model": model_name,
                    "threshold": threshold,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "pred_df": pred_df,
                }
            )

        champion = _select_pattern_champion(model_results, primary_metric=primary_metric, min_val_support=min_val_support)
        if champion is not None:
            champion_path = pattern_dir / "champion_test_predictions.csv"
            champion["pred_df"].to_csv(champion_path, index=False)
            champion_rows.append(
                {
                    "pattern": pattern,
                    "champion_model": champion["model"],
                    "threshold": champion["threshold"],
                    "selection_split": "val",
                    "selection_metric": primary_metric,
                    "selection_metric_value": champion["selection_score"],
                    "selection_f1": champion["val_metrics"]["f1"],
                    "selection_f2": champion["val_metrics"]["f2"],
                    "predictions_path": str(champion_path),
                }
            )

    summary_columns = [
        "pattern",
        "model",
        "selection_split",
        "selection_metric",
        "selection_metric_value",
        "selection_precision",
        "selection_recall",
        "selection_f1",
        "selection_f2",
        "f1",
        "f2",
        "precision",
        "recall",
        "pr_auc",
        "threshold",
        "support_positive",
        "support_negative",
    ]
    champion_columns = [
        "pattern",
        "champion_model",
        "threshold",
        "selection_split",
        "selection_metric",
        "selection_metric_value",
        "selection_f1",
        "selection_f2",
        "predictions_path",
    ]

    summary_df = pd.DataFrame(summary_rows, columns=summary_columns)
    champions_df = pd.DataFrame(champion_rows, columns=champion_columns)
    summary_df.to_csv(metrics_root / "model_comparison_summary.csv", index=False)
    champions_df.to_csv(metrics_root / "champions.csv", index=False)

    macro: dict[str, Any] = {"primary_selection_metric": primary_metric}
    if not summary_df.empty and not champions_df.empty:
        champion_metrics = summary_df.merge(
            champions_df[["pattern", "champion_model"]],
            left_on=["pattern", "model"],
            right_on=["pattern", "champion_model"],
            how="inner",
        )
        macro.update(
            {
                "macro_f1": float(champion_metrics["f1"].mean()),
                "macro_f2": float(champion_metrics["f2"].mean()),
                "macro_precision": float(champion_metrics["precision"].mean()),
                "macro_recall": float(champion_metrics["recall"].mean()),
                "macro_selection_metric": float(champion_metrics["selection_metric_value"].mean()),
                "patterns_covered": int(champion_metrics["pattern"].nunique()),
            }
        )

    _save_json(metrics_root / "macro_summary.json", macro)
    print(f"Experiment suite finished. Outputs at: {metrics_root}")
    return {
        "run_name": run_name,
        "metrics_root": metrics_root,
        "runtime_summary": runtime,
        "summary_df": summary_df,
        "champions_df": champions_df,
        "macro": macro,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-model experiment suite for 4 candlestick patterns")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--set", dest="set_overrides", action="append", default=[])
    parser.add_argument("--smoke", action="store_true", help="Run on synthetic small data for CI/smoke testing")
    args = parser.parse_args()

    cfg = load_config(args.config, args.config_override, args.set_overrides)
    run_experiment_suite(cfg=cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
