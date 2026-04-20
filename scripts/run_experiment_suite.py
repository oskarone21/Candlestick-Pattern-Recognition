from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd

try:
    from scripts._bootstrap import ensure_repo_root
except ImportError:
    from _bootstrap import ensure_repo_root

ensure_repo_root()

from candlestick.config import ensure_dir, load_config
from candlestick.datasets.window_builder import build_pattern_dataset, save_pattern_dataset
from candlestick.domain import (
    CALIBRATED_MODEL_NAMES,
    COLUMN_CLOSE,
    COLUMN_HIGH,
    COLUMN_LOW,
    COLUMN_OPEN,
    COLUMN_SYMBOL,
    COLUMN_TS_EVENT,
    COLUMN_VOLUME,
    DEFAULT_INSTRUMENT,
    DEFAULT_WORKING_TIMEZONE,
    METRIC_F1,
    MODEL_HGB,
    MODEL_LOGREG,
    SELECTION_SPLIT_VAL,
)
from candlestick.eval.calibration import (
    apply_probability_calibrator,
    calibration_diagnostics,
    fit_probability_calibrator,
)
from candlestick.eval.classification import (
    attach_confidence_intervals,
    choose_threshold,
    evaluate_threshold_metrics,
    selection_metric_value,
    save_metrics_report,
)
from candlestick.eval.label_sanity import (
    label_sanity_issues,
    summarize_pattern_dataset,
    summarize_pattern_events,
    summarize_processed_prices,
    summarize_split_support,
)
from candlestick.features.tabular import flatten_sequence_features
from candlestick.labeling.pattern_rules import detect_pattern_events
from candlestick.models.registry import predict_model_proba, save_model, train_model
from candlestick.models.torch_common import runtime_summary
from candlestick.optuna.search import OptunaUnavailableError, tune_model
from candlestick.project_utils import (
    allowed_patterns_from_cfg,
    candidate_models_from_cfg,
    load_processed_prices,
    model_selection_from_cfg,
    run_name_from_cfg,
)
from candlestick.split.time_split import assign_split_column, time_based_split


SUMMARY_COLUMNS = [
    "pattern",
    "model",
    "selection_split",
    "selection_metric",
    "selection_metric_value",
    "selection_precision",
    "selection_precision_lower_bound",
    "selection_recall",
    "selection_recall_lower_bound",
    "selection_f1",
    "selection_f1_lower_bound",
    "selection_f2",
    "selection_f2_lower_bound",
    "f1",
    "f2",
    "precision",
    "recall",
    "pr_auc",
    "threshold",
    "support_positive",
    "support_negative",
]

CHAMPION_COLUMNS = [
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


class SplitData(TypedDict):
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray


class ModelSummaryRow(TypedDict):
    pattern: str
    model: str
    selection_split: str
    selection_metric: str
    selection_metric_value: float
    selection_precision: float
    selection_precision_lower_bound: float
    selection_recall: float
    selection_recall_lower_bound: float
    selection_f1: float
    selection_f1_lower_bound: float
    selection_f2: float
    selection_f2_lower_bound: float
    f1: float
    f2: float
    precision: float
    recall: float
    pr_auc: float
    threshold: float
    support_positive: int
    support_negative: int


class ChampionRow(TypedDict):
    pattern: str
    champion_model: str
    threshold: float
    selection_split: str
    selection_metric: str
    selection_metric_value: float
    selection_f1: float
    selection_f2: float
    predictions_path: str


class ModelResult(TypedDict):
    pattern: str
    model: str
    threshold: float
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    pred_df: pd.DataFrame
    selection_score: float


@dataclass(frozen=True)
class RunRoots:
    metrics_root: Path
    ckpt_root: Path
    datasets_root: Path


@dataclass(frozen=True)
class PatternArtifacts:
    source: str
    events: pd.DataFrame | None
    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame


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


def _augment_train_positives(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    aug_cfg = cfg.get("training", {}).get("augmentation", {})
    if not bool(aug_cfg.get("enabled", False)):
        return X_train, y_train

    n_copies = int(aug_cfg.get("n_copies_per_positive", 2))
    if n_copies <= 0 or len(y_train) == 0:
        return X_train, y_train

    pos_idx = np.flatnonzero(y_train == 1)
    if len(pos_idx) == 0:
        return X_train, y_train

    X_train = X_train.astype(np.float32, copy=False)
    noise_std = float(aug_cfg.get("gaussian_noise_std", 0.012))
    price_std = float(aug_cfg.get("price_coherent_noise_std", 0.008))
    vol_range = float(aug_cfg.get("volume_jitter_range", 0.12))
    rng = np.random.default_rng(seed)

    augmented_windows: list[np.ndarray] = []
    augmented_labels: list[int] = []
    for idx in pos_idx:
        window = X_train[idx]
        window_min = np.min(window, axis=0, keepdims=True)
        window_max = np.max(window, axis=0, keepdims=True)
        denom = np.where((window_max - window_min) < 1.0e-6, 1.0, window_max - window_min).astype(np.float32)
        normalized = ((window - window_min) / denom).astype(np.float32, copy=False)

        for _ in range(n_copies):
            augmented = normalized.copy()
            price_noise = rng.normal(0.0, price_std, size=(augmented.shape[0], 1)).astype(np.float32)
            augmented[:, :4] += price_noise
            augmented += rng.normal(0.0, noise_std, size=augmented.shape).astype(np.float32)
            augmented[:, 4] *= np.float32(1.0 + rng.uniform(-vol_range, vol_range))
            augmented = np.clip(augmented, 0.0, 1.0)
            augmented_windows.append((augmented * denom + window_min).astype(np.float32, copy=False))
            augmented_labels.append(1)

    if not augmented_windows:
        return X_train, y_train

    X_aug = np.concatenate([X_train, np.stack(augmented_windows)], axis=0)
    y_aug = np.concatenate([y_train, np.asarray(augmented_labels, dtype=np.int64)], axis=0)
    order = rng.permutation(len(y_aug))
    return X_aug[order], y_aug[order]


def _apply_train_augmentation(
    split_data: SplitData,
    cfg: dict[str, Any],
    seed: int,
) -> SplitData:
    X_train_aug, y_train_aug = _augment_train_positives(
        split_data["X_train"],
        split_data["y_train"],
        cfg=cfg,
        seed=seed,
    )
    return {
        **split_data,
        "X_train": X_train_aug,
        "y_train": y_train_aug,
    }


def _generate_smoke_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_rows = 1400
    timestamps = pd.date_range(
        "2024-01-02 09:30",
        periods=n_rows,
        freq="15min",
        tz=DEFAULT_WORKING_TIMEZONE,
    )

    trend = np.linspace(0, 3.5, n_rows)
    wave = np.sin(np.linspace(0, 50, n_rows)) * 1.2
    noise = rng.normal(0, 0.2, n_rows)
    close = 100 + trend + wave + noise
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.05, n_rows)
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.4, n_rows)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.4, n_rows)
    volume = rng.integers(5000, 20000, size=n_rows)

    return pd.DataFrame(
        {
            COLUMN_SYMBOL: DEFAULT_INSTRUMENT,
            COLUMN_TS_EVENT: timestamps,
            COLUMN_OPEN: open_,
            COLUMN_HIGH: high,
            COLUMN_LOW: low,
            COLUMN_CLOSE: close,
            COLUMN_VOLUME: volume,
        }
    )


def _load_prices(cfg: dict[str, Any], smoke: bool) -> pd.DataFrame:
    if smoke:
        return _generate_smoke_data()
    return load_processed_prices(cfg)


def _resolve_roots(cfg: dict[str, Any]) -> tuple[RunRoots, Path | None]:
    run_name = run_name_from_cfg(cfg)
    roots = RunRoots(
        metrics_root=ensure_dir(Path(cfg["paths"].get("metrics_dir", "outputs/metrics")) / run_name),
        ckpt_root=ensure_dir(Path(cfg["paths"].get("checkpoints_dir", "outputs/checkpoints")) / run_name),
        datasets_root=ensure_dir(Path(cfg["paths"].get("datasets_dir", "outputs/datasets")) / run_name),
    )
    prebuilt_root = cfg.get("paths", {}).get("prebuilt_datasets_dir")
    return roots, Path(prebuilt_root) if prebuilt_root else None


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


def _build_or_load_pattern_dataset(
    prices: pd.DataFrame,
    pattern: str,
    cfg: dict[str, Any],
    prebuilt_root: Path | None,
) -> PatternArtifacts:
    prebuilt = _load_prebuilt_pattern_dataset(prebuilt_root, pattern)
    if prebuilt is not None:
        X, y, meta = prebuilt
        return PatternArtifacts(
            source="prebuilt_dataset",
            events=None,
            X=X,
            y=y,
            meta=meta,
        )

    lookback = int(cfg["windowing"].get("lookback_bars", 80))
    events = detect_pattern_events(prices, pattern, cfg)
    X, y, meta = build_pattern_dataset(prices, events, lookback_bars=lookback)
    return PatternArtifacts(
        source="live_generated",
        events=events,
        X=X,
        y=y,
        meta=meta,
    )


def _prepare_split(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[SplitData, pd.DataFrame]:
    eval_cfg = cfg["evaluation"]
    conf = cfg["labeling"]["confirmation"]
    lookback = int(cfg["windowing"].get("lookback_bars", 80))
    confirm_horizon = int(conf.get("confirm_break_within_bars", 8))
    default_embargo = lookback + confirm_horizon
    embargo = int(eval_cfg.get("embargo_bars", default_embargo))

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
                "Adjust ratios or embargo instead of silently disabling purge. "
                "Set evaluation.allow_zero_embargo_fallback=true only if you accept leakage risk."
            )

    if len(split.train) == 0 or len(split.val) == 0 or len(split.test) == 0:
        raise ValueError("Split has an empty fold. Adjust dataset size, ratios, or embargo settings.")

    split_data: SplitData = {
        "X_train": X[split.train],
        "y_train": y[split.train],
        "X_val": X[split.val],
        "y_val": y[split.val],
        "X_test": X[split.test],
        "y_test": y[split.test],
        "idx_train": split.train,
        "idx_val": split.val,
        "idx_test": split.test,
    }
    return split_data, meta_split


def _model_input_for_name(name: str, X: np.ndarray) -> np.ndarray:
    if name in {MODEL_LOGREG, MODEL_HGB}:
        return flatten_sequence_features(X)
    return X


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_git_metadata() -> dict[str, Any]:
    repo_root = _repo_root()

    def _run_git(*args: str) -> str | None:
        try:
            return (
                subprocess.check_output(["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            return None

    status = _run_git("status", "--short")
    return {
        "commit_sha": _run_git("rev-parse", "HEAD"),
        "branch": _run_git("branch", "--show-current"),
        "is_dirty": bool(status) if status is not None else None,
    }


def _requirements_versions(requirements_path: Path) -> dict[str, str | None]:
    if not requirements_path.exists():
        return {}

    versions: dict[str, str | None] = {}
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        requirement = raw_line.split("#", 1)[0].strip()
        if not requirement:
            continue

        normalized = requirement
        for marker in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            if marker in normalized:
                normalized = normalized.split(marker, 1)[0]
                break
        package_name = normalized.split("[", 1)[0].strip()
        if not package_name:
            continue

        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            versions[package_name] = None

    return versions


def _csv_artifact_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return summary

    sha256 = hashlib.sha256()
    row_count = 0
    first_line = True
    size_bytes = 0
    with path.open("rb") as handle:
        for line in handle:
            sha256.update(line)
            size_bytes += len(line)
            if first_line:
                first_line = False
                continue
            row_count += 1

    summary.update(
        {
            "size_bytes": size_bytes,
            "row_count": row_count,
            "sha256": sha256.hexdigest(),
        }
    )
    return summary


def _build_repro_manifest(
    cfg: dict[str, Any],
    runtime: dict[str, Any],
    prices: pd.DataFrame,
    smoke: bool,
) -> dict[str, Any]:
    repo_root = _repo_root()
    runtime_cfg = cfg.get("_runtime", {})
    paths_cfg = cfg.get("paths", {})

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name_from_cfg(cfg),
        "smoke": bool(smoke),
        "git": _read_git_metadata(),
        "python_version": sys.version.split()[0],
        "config_stack": {
            "base_config": runtime_cfg.get("config_path"),
            "config_overrides": list(runtime_cfg.get("config_overrides", [])),
            "set_overrides": list(runtime_cfg.get("set_overrides", [])),
        },
        "runtime": runtime,
        "data_source": {
            "provider": cfg.get("data_source", {}).get("provider"),
            "kaggle_dataset": cfg.get("data_source", {}).get("kaggle_dataset"),
            "kaggle_file_path": cfg.get("data_source", {}).get("kaggle_file_path"),
            "instrument": cfg.get("data_source", {}).get("instrument"),
            "base_timeframe": cfg.get("resampling", {}).get("base_timeframe"),
            "target_timeframe": cfg.get("resampling", {}).get("target_timeframe"),
        },
        "dependency_versions": _requirements_versions(repo_root / "requirements.txt"),
        "datasets": {
            "raw_1m": _csv_artifact_summary(Path(paths_cfg.get("raw_1m_path", "data/raw/spy_1m.csv"))),
            "processed_15m": _csv_artifact_summary(Path(paths_cfg.get("processed_15m_path", "data/processed/spy_15m.csv"))),
            "processed_price_summary": summarize_processed_prices(prices),
        },
    }


def _unique_issue_list(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _fatal_message_for_pattern(pattern: str, issues: list[str], split_error: str | None) -> str:
    reasons: list[str] = []
    if "zero_positive_events" in issues:
        reasons.append("detector found zero positive events")
    if "zero_positive_dataset_labels" in issues:
        reasons.append("dataset construction produced zero positive labels")
    for split_name in ("train", "val", "test"):
        issue_name = f"zero_positive_support_{split_name}"
        if issue_name in issues:
            reasons.append(f"{split_name} split has zero positive support")
    if "split_error" in issues and split_error:
        reasons.append(f"split preparation failed: {split_error}")
    if not reasons:
        reasons.append("label sanity checks failed")
    return f"Label sanity check failed for `{pattern}`: " + "; ".join(reasons)


def _build_pattern_label_sanity(
    pattern: str,
    artifacts: PatternArtifacts,
    split_data: SplitData | None,
    meta_split: pd.DataFrame | None,
    split_error: str | None,
    smoke: bool,
) -> dict[str, Any]:
    event_summary = summarize_pattern_events(pattern, artifacts.events, source=artifacts.source)
    dataset_summary = summarize_pattern_dataset(pattern, artifacts.X, artifacts.y, artifacts.meta, source=artifacts.source)
    split_summary = summarize_split_support(split_data, meta_split)
    issues = _unique_issue_list(
        label_sanity_issues(
            event_summary,
            dataset_summary,
            split_summary,
            split_error=split_error,
        )
    )
    fatal = bool(issues) and not smoke
    return {
        "pattern": pattern,
        "source": artifacts.source,
        "event_summary": event_summary,
        "dataset_summary": dataset_summary,
        "split_summary": split_summary,
        "split_error": split_error,
        "warnings": issues,
        "fatal": fatal,
        "fatal_message": _fatal_message_for_pattern(pattern, issues, split_error) if fatal else None,
    }


def _safe_tune(
    model_name: str,
    train_data: dict[str, np.ndarray],
    cfg: dict[str, Any],
    seed: int,
    pattern: str,
) -> dict[str, Any]:
    opt_cfg = cfg.get("optuna", {})
    tuned_models = {str(model) for model in opt_cfg.get("tuned_models", ["lstm", "tcn"])}
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
    if model_name not in CALIBRATED_MODEL_NAMES:
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
    model_results: list[ModelResult],
    primary_metric: str,
    min_val_support: int,
    conservative_selection: bool,
    allow_low_support_fallback: bool,
) -> ModelResult | None:
    best_supported: ModelResult | None = None
    best_any: ModelResult | None = None

    for result in model_results:
        score = float(
            selection_metric_value(
                result["val_metrics"],
                primary_metric,
                conservative=conservative_selection,
            )
        )
        support_ok = result["val_metrics"]["support"]["positive"] >= min_val_support
        candidate: ModelResult = {**result, "selection_score": score}

        if best_any is None or score > best_any["selection_score"]:
            best_any = candidate
        if support_ok and (best_supported is None or score > best_supported["selection_score"]):
            best_supported = candidate

    if best_supported is not None:
        return best_supported
    if allow_low_support_fallback:
        return best_any
    return None


def _build_summary_row(
    pattern: str,
    model_name: str,
    primary_metric: str,
    conservative_selection: bool,
    threshold: float,
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> ModelSummaryRow:
    return {
        "pattern": pattern,
        "model": model_name,
        "selection_split": SELECTION_SPLIT_VAL,
        "selection_metric": primary_metric,
        "selection_metric_value": float(
            selection_metric_value(val_metrics, primary_metric, conservative=conservative_selection)
        ),
        "selection_precision": float(val_metrics["precision"]),
        "selection_precision_lower_bound": float(val_metrics["precision_lower_bound"]),
        "selection_recall": float(val_metrics["recall"]),
        "selection_recall_lower_bound": float(val_metrics["recall_lower_bound"]),
        "selection_f1": float(val_metrics["f1"]),
        "selection_f1_lower_bound": float(val_metrics["f1_lower_bound"]),
        "selection_f2": float(val_metrics["f2"]),
        "selection_f2_lower_bound": float(val_metrics["f2_lower_bound"]),
        "f1": float(test_metrics["f1"]),
        "f2": float(test_metrics["f2"]),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "pr_auc": float(test_metrics["pr_auc"]),
        "threshold": float(threshold),
        "support_positive": int(test_metrics["support"]["positive"]),
        "support_negative": int(test_metrics["support"]["negative"]),
    }


def _write_prediction_frame(
    meta: pd.DataFrame,
    split_idx: np.ndarray,
    pattern: str,
    model_name: str,
    threshold: float,
    probabilities: np.ndarray,
    raw_probabilities: np.ndarray,
) -> pd.DataFrame:
    frame = meta.iloc[split_idx].copy().reset_index(drop=True)
    frame["proba_raw"] = raw_probabilities
    frame["proba"] = probabilities
    frame["threshold"] = threshold
    frame["model"] = model_name
    frame["pattern"] = pattern
    return frame


def _evaluate_model(
    pattern: str,
    model_name: str,
    split_data: SplitData,
    meta: pd.DataFrame,
    cfg: dict[str, Any],
    seed: int,
    runtime: dict[str, Any],
    roots: RunRoots,
) -> tuple[ModelResult, ModelSummaryRow] | None:
    X_train_model = _model_input_for_name(model_name, split_data["X_train"])
    X_val_model = _model_input_for_name(model_name, split_data["X_val"])
    X_test_model = _model_input_for_name(model_name, split_data["X_test"])

    if len(X_train_model) == 0 or len(X_val_model) == 0 or len(X_test_model) == 0:
        return None
    if len(np.unique(split_data["y_train"])) < 2:
        return None

    best_params = _safe_tune(
        model_name,
        {
            "X_train": X_train_model,
            "y_train": split_data["y_train"],
            "X_val": X_val_model,
            "y_val": split_data["y_val"],
        },
        cfg,
        seed,
        pattern,
    )

    model = train_model(
        model_name=model_name,
        X_train=X_train_model,
        y_train=split_data["y_train"],
        X_val=X_val_model,
        y_val=split_data["y_val"],
        cfg=cfg,
        seed=seed,
        params=best_params,
    )

    val_prob_raw = predict_model_proba(model_name, model, X_val_model)
    test_prob_raw = predict_model_proba(model_name, model, X_test_model)
    val_prob, test_prob, calibrator, calibration = _maybe_calibrate_probabilities(
        model_name=model_name,
        y_val=split_data["y_val"],
        val_prob_raw=val_prob_raw,
        test_prob_raw=test_prob_raw,
    )

    model_selection_cfg = model_selection_from_cfg(cfg, pattern=pattern)
    primary_metric = str(model_selection_cfg.get("primary_selection_metric", METRIC_F1)).lower()
    conservative_selection = bool(model_selection_cfg.get("use_conservative_selection_scores", True))
    threshold, threshold_diag = choose_threshold(
        y_true=split_data["y_val"],
        y_prob=val_prob,
        precision_floor=float(model_selection_cfg.get("precision_floor", 0.0)),
        recall_floor=float(model_selection_cfg.get("recall_floor", 0.0)),
        minimum_predicted_positive_support=int(model_selection_cfg.get("minimum_predicted_positive_support", 0)),
        minimum_true_positive_support=int(model_selection_cfg.get("minimum_true_positive_support", 0)),
        primary_metric=primary_metric,
        grid_size=int(cfg.get("evaluation", {}).get("threshold_grid_size", 181)),
        conservative_selection=conservative_selection,
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

    model_path = roots.ckpt_root / pattern / f"{model_name}.bin"
    save_model(model_name, model, model_path)
    postprocess_path = roots.ckpt_root / pattern / f"{model_name}.postprocess.json"
    _save_json(postprocess_path, {"calibrator": calibrator})

    report = dict(test_metrics)
    report["artifact_path"] = str(model_path)
    report["postprocess_path"] = str(postprocess_path)
    report["selection_metric"] = primary_metric
    report["selection_value"] = float(
        selection_metric_value(val_metrics, primary_metric, conservative=conservative_selection)
    )
    report["selection_metrics"] = val_metrics
    report["selection_conservative"] = conservative_selection
    report["threshold_diagnostics"] = threshold_diag
    report["calibration"] = calibration
    report["runtime_summary"] = runtime

    pattern_dir = ensure_dir(roots.metrics_root / pattern)
    save_metrics_report(report, pattern_dir / f"{model_name}.json")
    _save_json(pattern_dir / f"{model_name}_threshold_diagnostics.json", threshold_diag)
    _save_json(pattern_dir / f"{model_name}_calibration.json", calibration)

    val_pred_df = _write_prediction_frame(
        meta,
        split_data["idx_val"],
        pattern,
        model_name,
        threshold,
        val_prob,
        val_prob_raw,
    )
    val_pred_df.to_csv(pattern_dir / f"{model_name}_val_predictions.csv", index=False)

    test_pred_df = _write_prediction_frame(
        meta,
        split_data["idx_test"],
        pattern,
        model_name,
        threshold,
        test_prob,
        test_prob_raw,
    )
    test_pred_df.to_csv(pattern_dir / f"{model_name}_test_predictions.csv", index=False)

    summary_row = _build_summary_row(
        pattern=pattern,
        model_name=model_name,
        primary_metric=primary_metric,
        conservative_selection=conservative_selection,
        threshold=threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    result: ModelResult = {
        "pattern": pattern,
        "model": model_name,
        "threshold": threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "pred_df": test_pred_df,
        "selection_score": float(
            selection_metric_value(val_metrics, primary_metric, conservative=conservative_selection)
        ),
    }
    return result, summary_row


def _run_pattern(
    prices: pd.DataFrame,
    pattern: str,
    cfg: dict[str, Any],
    seed: int,
    runtime: dict[str, Any],
    roots: RunRoots,
    prebuilt_root: Path | None,
) -> tuple[list[ModelSummaryRow], ChampionRow | None, dict[str, Any]]:
    artifacts = _build_or_load_pattern_dataset(prices, pattern, cfg, prebuilt_root)
    save_pattern_dataset(roots.datasets_root, artifacts.X, artifacts.y, artifacts.meta, pattern)

    pattern_dir = ensure_dir(roots.metrics_root / pattern)
    split_data: SplitData | None = None
    meta_split: pd.DataFrame | None = None
    split_error: str | None = None

    if len(artifacts.y) > 0 and not artifacts.meta.empty:
        try:
            split_data, meta_split = _prepare_split(artifacts.X, artifacts.y, artifacts.meta, cfg)
        except ValueError as exc:
            split_error = str(exc)
    else:
        split_error = "Dataset is empty after event-to-window construction."

    pattern_sanity = _build_pattern_label_sanity(
        pattern=pattern,
        artifacts=artifacts,
        split_data=split_data,
        meta_split=meta_split,
        split_error=split_error,
        smoke=bool(cfg.get("_runtime", {}).get("smoke", False)),
    )
    _save_json(pattern_dir / "label_sanity.json", pattern_sanity)

    if pattern_sanity["fatal"]:
        return [], None, pattern_sanity

    if split_error or split_data is None or meta_split is None:
        print(f"Skipping pattern `{pattern}` due to split constraints: {split_error}")
        return [], None, pattern_sanity

    meta_split.to_csv(pattern_dir / "split_metadata.csv", index=False)

    hard_cfg = cfg["labeling"].get("labeling_policy", {}).get("hard_negative_sampling", {})
    if hard_cfg.get("enabled", True):
        split_data["X_train"], split_data["y_train"] = _downsample_train_negatives(
            split_data["X_train"],
            split_data["y_train"],
            target_ratio=str(hard_cfg.get("target_positive_to_negative_ratio", "1:3")),
            seed=seed,
        )
    split_data = _apply_train_augmentation(split_data, cfg=cfg, seed=seed)

    summary_rows: list[ModelSummaryRow] = []
    model_results: list[ModelResult] = []
    for model_name in candidate_models_from_cfg(cfg):
        evaluated = _evaluate_model(pattern, model_name, split_data, artifacts.meta, cfg, seed, runtime, roots)
        if evaluated is None:
            continue
        model_result, summary_row = evaluated
        model_results.append(model_result)
        summary_rows.append(summary_row)

    selection_cfg = model_selection_from_cfg(cfg, pattern=pattern)
    primary_metric = str(selection_cfg.get("primary_selection_metric", METRIC_F1)).lower()
    conservative_selection = bool(selection_cfg.get("use_conservative_selection_scores", True))
    min_support = int(cfg.get("evaluation", {}).get("minimum_test_positive_support", 5))
    min_val_support = int(cfg.get("evaluation", {}).get("minimum_validation_positive_support", min_support))
    allow_low_support_fallback = bool(cfg.get("evaluation", {}).get("allow_low_support_champion_fallback", False))
    champion = _select_pattern_champion(
        model_results,
        primary_metric=primary_metric,
        min_val_support=min_val_support,
        conservative_selection=conservative_selection,
        allow_low_support_fallback=allow_low_support_fallback,
    )
    if champion is None:
        return summary_rows, None, pattern_sanity

    champion_path = pattern_dir / "champion_test_predictions.csv"
    champion["pred_df"].to_csv(champion_path, index=False)
    champion_row: ChampionRow = {
        "pattern": pattern,
        "champion_model": champion["model"],
        "threshold": float(champion["threshold"]),
        "selection_split": SELECTION_SPLIT_VAL,
        "selection_metric": primary_metric,
        "selection_metric_value": float(champion["selection_score"]),
        "selection_f1": float(champion["val_metrics"]["f1"]),
        "selection_f2": float(champion["val_metrics"]["f2"]),
        "predictions_path": str(champion_path),
    }
    return summary_rows, champion_row, pattern_sanity


def _write_run_outputs(
    roots: RunRoots,
    summary_rows: list[ModelSummaryRow],
    champion_rows: list[ChampionRow],
    primary_metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    champions_df = pd.DataFrame(champion_rows, columns=CHAMPION_COLUMNS)
    summary_df.to_csv(roots.metrics_root / "model_comparison_summary.csv", index=False)
    champions_df.to_csv(roots.metrics_root / "champions.csv", index=False)

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

    _save_json(roots.metrics_root / "macro_summary.json", macro)
    return summary_df, champions_df, macro


def run_experiment_suite(cfg: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    seed = int(cfg["project"].get("seed", 42))
    np.random.seed(seed)
    cfg = {
        **cfg,
        "_runtime": {
            **cfg.get("_runtime", {}),
            "smoke": bool(smoke),
        },
    }

    roots, prebuilt_root = _resolve_roots(cfg)
    runtime = runtime_summary(cfg)
    _save_json(roots.metrics_root / "runtime_summary.json", runtime)
    print(
        "Runtime summary:",
        f"device={runtime['selected_device']}",
        f"cuda_available={runtime['cuda_available']}",
        f"mps_available={runtime['mps_available']}",
        f"mixed_precision={runtime['mixed_precision_enabled']}",
        f"gpu={runtime['gpu_name'] or 'n/a'}",
    )

    prices = _load_prices(cfg, smoke=smoke)
    repro_manifest = _build_repro_manifest(cfg, runtime, prices, smoke=smoke)
    _save_json(roots.metrics_root / "repro_manifest.json", repro_manifest)
    primary_metric = str(model_selection_from_cfg(cfg).get("primary_selection_metric", METRIC_F1)).lower()
    run_label_sanity: dict[str, Any] = {
        "run_name": run_name_from_cfg(cfg),
        "smoke": bool(smoke),
        "prebuilt_datasets_dir": str(prebuilt_root) if prebuilt_root else None,
        "processed_price_summary": summarize_processed_prices(prices),
        "patterns": {},
    }
    _save_json(roots.metrics_root / "label_sanity.json", run_label_sanity)

    summary_rows: list[ModelSummaryRow] = []
    champion_rows: list[ChampionRow] = []
    for pattern in allowed_patterns_from_cfg(cfg):
        pattern_summary_rows, champion_row, pattern_sanity = _run_pattern(
            prices=prices,
            pattern=pattern,
            cfg=cfg,
            seed=seed,
            runtime=runtime,
            roots=roots,
            prebuilt_root=prebuilt_root,
        )
        run_label_sanity["patterns"][pattern] = pattern_sanity
        _save_json(roots.metrics_root / "label_sanity.json", run_label_sanity)
        if pattern_sanity["fatal"]:
            raise ValueError(pattern_sanity["fatal_message"])
        summary_rows.extend(pattern_summary_rows)
        if champion_row is not None:
            champion_rows.append(champion_row)

    summary_df, champions_df, macro = _write_run_outputs(
        roots=roots,
        summary_rows=summary_rows,
        champion_rows=champion_rows,
        primary_metric=primary_metric,
    )
    print(f"Experiment suite finished. Outputs at: {roots.metrics_root}")
    return {
        "run_name": run_name_from_cfg(cfg),
        "metrics_root": roots.metrics_root,
        "runtime_summary": runtime,
        "summary_df": summary_df,
        "champions_df": champions_df,
        "macro": macro,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the experiment suite for the configured patterns")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--set", dest="set_overrides", action="append", default=[])
    parser.add_argument("--smoke", action="store_true", help="Run on synthetic small data for CI or smoke testing")
    args = parser.parse_args()

    cfg = load_config(args.config, args.config_override, args.set_overrides)
    cfg = {
        **cfg,
        "_runtime": {
            **cfg.get("_runtime", {}),
            "config_path": str(Path(args.config)),
            "config_overrides": [str(Path(path)) for path in args.config_override],
            "set_overrides": list(args.set_overrides),
        },
    }
    run_experiment_suite(cfg=cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
