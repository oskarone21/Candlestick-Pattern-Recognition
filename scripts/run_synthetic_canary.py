from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts._bootstrap import ensure_repo_root
except ImportError:
    from _bootstrap import ensure_repo_root

ensure_repo_root()

from candlestick.benchmarks.synthetic_canary import SYNTHETIC_CANARY_TIERS, build_synthetic_canary_dataset
from candlestick.config import load_config
from candlestick.eval.calibration import apply_probability_calibrator
from candlestick.eval.classification import evaluate_threshold_metrics
from candlestick.features.tabular import flatten_sequence_features
from candlestick.models.registry import load_model, predict_model_proba


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_artifact_paths(metrics_root: Path, pattern: str, model_name: str) -> tuple[Path, Path]:
    report = _load_json(metrics_root / pattern / f"{model_name}.json")
    artifact_path = Path(report["artifact_path"])
    postprocess_path = Path(report["postprocess_path"]) if "postprocess_path" in report else artifact_path.with_suffix(".postprocess.json")
    return artifact_path, postprocess_path


def _load_negative_windows(datasets_root: Path, pattern: str) -> np.ndarray:
    npz_path = datasets_root / pattern / "dataset.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found for synthetic canary benchmark: {npz_path}")

    with np.load(npz_path) as data:
        X = data["X"]
        y = data["y"]
    negatives = X[y == 0]
    if len(negatives) == 0:
        raise ValueError(f"No negative windows available for pattern `{pattern}` in {npz_path}.")
    return negatives


def _predict_probabilities(
    model_name: str,
    artifact_path: Path,
    calibrator_path: Path,
    X: np.ndarray,
) -> np.ndarray:
    model = load_model(model_name, artifact_path)
    X_input = flatten_sequence_features(X) if model_name in {"logreg", "hgb"} else X
    raw_prob = predict_model_proba(model_name, model, X_input)
    calibrator = {"method": "identity", "fitted": False}
    if calibrator_path.exists():
        calibrator = _load_json(calibrator_path).get("calibrator", calibrator)
    return apply_probability_calibrator(raw_prob, calibrator)


def _summary_row(
    pattern: str,
    model_name: str,
    tier: str,
    threshold: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pattern": pattern,
        "champion_model": model_name,
        "tier": tier,
        "threshold": float(threshold),
        "samples": int(metrics["support"]["positive"] + metrics["support"]["negative"]),
        "positive_support": int(metrics["support"]["positive"]),
        "negative_support": int(metrics["support"]["negative"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "f2": float(metrics["f2"]),
        "pr_auc": float(metrics["pr_auc"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic canary benchmark for a completed 15-minute experiment run.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--set", dest="set_overrides", action="append", default=[])
    parser.add_argument("--metrics-root", required=True, help="Completed run metrics root containing champions.csv")
    parser.add_argument("--count-per-tier", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config, args.config_override, args.set_overrides)
    metrics_root = Path(args.metrics_root)
    champions = pd.read_csv(metrics_root / "champions.csv")
    run_name = metrics_root.name
    datasets_root = Path(cfg["paths"].get("datasets_dir", "outputs/datasets")) / run_name

    result: dict[str, Any] = {
        "run_name": run_name,
        "metrics_root": str(metrics_root),
        "datasets_root": str(datasets_root),
        "count_per_tier": int(args.count_per_tier),
        "patterns": {},
    }
    rows: list[dict[str, Any]] = []

    for record in champions.to_dict(orient="records"):
        pattern = str(record["pattern"])
        model_name = str(record["champion_model"])
        threshold = float(record["threshold"])
        negatives = _load_negative_windows(datasets_root, pattern)
        X_canary, y_canary, meta = build_synthetic_canary_dataset(
            negatives,
            pattern,
            count_per_tier=int(args.count_per_tier),
            seed=int(args.seed),
        )
        artifact_path, postprocess_path = _resolve_artifact_paths(metrics_root, pattern, model_name)
        probabilities = _predict_probabilities(model_name, artifact_path, postprocess_path, X_canary)

        pattern_result = {
            "champion_model": model_name,
            "threshold": threshold,
            "tiers": {},
            "overall": evaluate_threshold_metrics(y_canary, probabilities, threshold),
        }
        rows.append(_summary_row(pattern, model_name, "overall", threshold, pattern_result["overall"]))

        for tier in SYNTHETIC_CANARY_TIERS:
            mask = meta["tier"].astype(str) == tier
            tier_metrics = evaluate_threshold_metrics(y_canary[mask.to_numpy()], probabilities[mask.to_numpy()], threshold)
            pattern_result["tiers"][tier] = tier_metrics
            rows.append(_summary_row(pattern, model_name, tier, threshold, tier_metrics))

        result["patterns"][pattern] = pattern_result

    output_json = metrics_root / "synthetic_canary.json"
    output_csv = metrics_root / "synthetic_canary_summary.csv"
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print(f"Synthetic canary report saved to: {output_json}")
    print(f"Synthetic canary summary saved to: {output_csv}")


if __name__ == "__main__":
    main()
