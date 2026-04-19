from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from scripts._bootstrap import ensure_repo_root
except ImportError:
    from _bootstrap import ensure_repo_root

ensure_repo_root()

from candlestick.config import ensure_dir, load_config
from candlestick.datasets.window_builder import build_pattern_dataset
from candlestick.eval.label_sanity import (
    summarize_pattern_dataset,
    summarize_pattern_events,
    summarize_processed_prices,
    summarize_split_support,
)
from candlestick.labeling.pattern_rules import detect_pattern_events
from candlestick.project_utils import allowed_patterns_from_cfg, load_processed_prices, run_name_from_cfg
from scripts.run_experiment_suite import _prepare_split


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _current_pattern_summary(prices: pd.DataFrame, pattern: str, cfg: dict[str, Any]) -> dict[str, Any]:
    events = detect_pattern_events(prices, pattern, cfg)
    lookback = int(cfg["windowing"].get("lookback_bars", 80))
    X, y, meta = build_pattern_dataset(prices, events, lookback_bars=lookback)

    split_data = None
    meta_split = None
    split_error = None
    if len(y) > 0 and not meta.empty:
        try:
            split_data, meta_split = _prepare_split(X, y, meta, cfg)
        except ValueError as exc:
            split_error = str(exc)
    else:
        split_error = "Dataset is empty after event-to-window construction."

    return {
        "event_summary": summarize_pattern_events(pattern, events, source="live_generated"),
        "dataset_summary": summarize_pattern_dataset(pattern, X, y, meta, source="live_generated"),
        "split_summary": summarize_split_support(split_data, meta_split),
        "split_error": split_error,
    }


def _reference_split_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "folds": {}}

    frame = pd.read_csv(path)
    if frame.empty or "split" not in frame.columns or "label" not in frame.columns:
        return {"available": False, "folds": {}}

    folds: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        subset = frame[frame["split"].astype(str) == split_name]
        folds[split_name] = {
            "samples": int(len(subset)),
            "positive": int((subset["label"] == 1).sum()),
            "negative": int((subset["label"] == 0).sum()),
        }

    return {"available": True, "folds": folds}


def _reference_pattern_summary(pattern: str, reference_datasets_dir: Path, reference_metrics_dir: Path) -> dict[str, Any]:
    pattern_dir = reference_datasets_dir / pattern
    manifest_path = pattern_dir / "manifest.json"
    manifest = None
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

    split_summary = _reference_split_summary(reference_metrics_dir / pattern / "split_metadata.csv")
    return {
        "manifest": manifest,
        "split_summary": split_summary,
    }


def _comparison_rows(
    patterns: list[str],
    current: dict[str, Any],
    reference: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        current_pattern = current["patterns"][pattern]
        reference_pattern = reference["patterns"].get(pattern, {})
        reference_manifest = reference_pattern.get("manifest") or {}
        reference_split = reference_pattern.get("split_summary", {}).get("folds", {})
        current_split = current_pattern["split_summary"].get("folds", {})
        rows.append(
            {
                "pattern": pattern,
                "current_event_count": current_pattern["event_summary"]["event_count"],
                "current_positive_events": current_pattern["event_summary"]["positive_events"],
                "current_dataset_samples": current_pattern["dataset_summary"]["samples"],
                "current_dataset_positives": current_pattern["dataset_summary"]["positive_labels"],
                "current_train_positive": current_split.get("train", {}).get("positive"),
                "current_val_positive": current_split.get("val", {}).get("positive"),
                "current_test_positive": current_split.get("test", {}).get("positive"),
                "reference_dataset_samples": reference_manifest.get("samples"),
                "reference_dataset_positives": reference_manifest.get("positives"),
                "reference_train_positive": reference_split.get("train", {}).get("positive"),
                "reference_val_positive": reference_split.get("val", {}).get("positive"),
                "reference_test_positive": reference_split.get("test", {}).get("positive"),
                "current_split_error": current_pattern.get("split_error"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose live 15-minute label regression against a preserved reference.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--set", dest="set_overrides", action="append", default=[])
    parser.add_argument(
        "--reference-datasets-dir",
        default="outputs/datasets/stageA_compare_all_no_optuna",
        help="Preserved reference datasets root to compare against.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.config_override, args.set_overrides)
    prices = load_processed_prices(cfg)
    patterns = allowed_patterns_from_cfg(cfg)

    current = {
        "processed_price_summary": summarize_processed_prices(prices),
        "patterns": {
            pattern: _current_pattern_summary(prices, pattern, cfg)
            for pattern in patterns
        },
    }

    reference_datasets_dir = Path(args.reference_datasets_dir)
    reference_metrics_dir = Path(cfg["paths"].get("metrics_dir", "outputs/metrics")) / reference_datasets_dir.name
    reference = {
        "datasets_dir": str(reference_datasets_dir),
        "metrics_dir": str(reference_metrics_dir),
        "historical_price_comparison_available": False,
        "note": (
            "No preserved historical stage-A processed 15-minute CSV exists in the repository, "
            "so historical comparison is dataset-level and split-level only."
        ),
        "patterns": {
            pattern: _reference_pattern_summary(pattern, reference_datasets_dir, reference_metrics_dir)
            for pattern in patterns
        },
    }

    rows = _comparison_rows(patterns, current, reference)
    outputs_root = Path(cfg["paths"].get("outputs_root", "outputs"))
    diagnostics_root = ensure_dir(outputs_root / "diagnostics")
    run_name = run_name_from_cfg(cfg)
    json_path = diagnostics_root / f"label_regression_{run_name}.json"
    csv_path = diagnostics_root / f"label_regression_{run_name}.csv"

    report = {
        "run_name": run_name,
        "current": current,
        "reference": reference,
        "comparison_rows": rows,
    }
    _save_json(json_path, report)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"Label regression report saved to: {json_path}")
    print(f"Label regression summary saved to: {csv_path}")


if __name__ == "__main__":
    main()
