"""
run_trading_product.py — two-stage trading product pipeline.

Stage 1: train the existing sequence detector on pattern labels.
Stage 2: train a trade-selection model on forward PnL labels.
Replay : run a short paper-trading session on the latest bars.
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.loader import load_raw_ohlcv
from src.data.resampler import resample_ohlcv
from src.labeling.double_bottom import label_double_bottom
from src.labeling.double_top import label_double_top
from src.labeling.extrema import find_extrema
from src.labeling.head_shoulders import label_head_shoulders
from src.labeling.inverse_head_shoulders import (
    label_inverse_head_shoulders,
)
from src.labeling.smoother import smooth_ohlcv
from src.models.tcn import build_model
from src.training.dataset import make_dataloaders
from src.training.trainer import train
from src.trading.inference import (
    build_thresholded_report,
    choose_detection_threshold,
    compute_binary_metrics,
    predict_sequence_probabilities,
)
from src.trading.paper_trader import run_paper_replay
from src.trading.split import purged_time_split_indices
from src.trading.stage2 import (
    build_stage2_model,
    build_trade_classification_report,
    choose_trade_threshold,
    compute_trade_signal_metrics,
    extract_stage2_coefficients,
    predict_trade_probabilities,
)
from src.trading.trade_labeling import build_trade_dataset
from src.trading.visualization import render_product_visualisations
from src.trading.window_dataset import build_full_anchor_stage1_dataset


PATTERN_REGISTRY = {
    "head_shoulders": label_head_shoulders,
    "inverse_head_shoulders": label_inverse_head_shoulders,
    "double_top": label_double_top,
    "double_bottom": label_double_bottom,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge config overrides."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _save_json(path: str, payload: dict | list) -> None:
    """Persist a Python object to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the two-stage trading product.")
    parser.add_argument(
        "--config-override",
        default=None,
        help="Optional YAML override merged on top of configs/config.yaml",
    )
    args = parser.parse_args()

    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.config_override:
        with open(args.config_override, encoding="utf-8") as f:
            override = yaml.safe_load(f)
        cfg = _deep_merge(cfg, override)
        print(f"Config override loaded: {args.config_override}")

    active_pattern = cfg["labeling"]["active_pattern"]
    if active_pattern not in PATTERN_REGISTRY:
        raise ValueError(
            f"Unsupported active pattern '{active_pattern}'. "
            f"Supported: {sorted(PATTERN_REGISTRY)}"
        )
    label_fn = PATTERN_REGISTRY[active_pattern]

    product_root = os.path.join("outputs", "product", active_pattern)
    stage1_dir = os.path.join(product_root, "stage1")
    stage2_dir = os.path.join(product_root, "stage2")
    replay_dir = os.path.join(product_root, "paper_replay")
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    os.makedirs(replay_dir, exist_ok=True)

    cfg["paths"]["checkpoints_dir"] = os.path.join(stage1_dir, "checkpoints")
    cfg["paths"]["metrics_dir"] = stage1_dir
    os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg["project"]["device"] != "cpu"
        else "cpu"
    )

    print(f"\n{'=' * 70}")
    print(f"Two-Stage Trading Product — pattern: {active_pattern} | device: {device}")
    print(f"{'=' * 70}\n")

    print(">>> Step 1: Data and labels")
    df_1min = load_raw_ohlcv(cfg)
    df_15min = resample_ohlcv(df_1min, cfg)
    df_smooth, bandwidth = smooth_ohlcv(df_15min, cfg)
    extrema = find_extrema(df_smooth, cfg)
    rule_labeled = sorted(label_fn(df_smooth, extrema, cfg), key=lambda w: w.anchor_bar)
    rule_positive_count = sum(int(w.label) == 1 for w in rule_labeled)
    rule_negative_count = len(rule_labeled) - rule_positive_count
    stage1_data = build_full_anchor_stage1_dataset(df_smooth, rule_labeled, cfg)
    X_stage1 = stage1_data["X"]
    y_stage1 = stage1_data["y"]
    anchors = stage1_data["anchors"]
    stage1_metadata = stage1_data["metadata"]

    print(
        f"Rule-labeled anchors: {len(rule_labeled)} | "
        f"positive: {rule_positive_count} | negative: {rule_negative_count}"
    )
    print(
        f"Stage 1 dataset: {len(y_stage1)} windows | "
        f"positive: {int(y_stage1.sum())} | negative: {int((y_stage1 == 0).sum())}"
    )

    split_idx = purged_time_split_indices(anchors, cfg)
    tr_idx = split_idx["train"]
    val_idx = split_idx["val"]
    te_idx = split_idx["test"]

    train_loader, val_loader, test_loader = make_dataloaders(
        X_stage1[tr_idx], y_stage1[tr_idx],
        X_stage1[val_idx], y_stage1[val_idx],
        X_stage1[te_idx], y_stage1[te_idx],
        cfg,
    )

    print("\n>>> Step 2: Stage 1 training")
    stage1_model = build_model(cfg, device)
    history = train(stage1_model, train_loader, val_loader, cfg, device)
    stage1_model.load_state_dict(torch.load(history["checkpoint"], map_location=device))

    stage1_probs_all = predict_sequence_probabilities(stage1_model, X_stage1, device)
    stage1_threshold, stage1_search_rows, stage1_val_summary = choose_detection_threshold(
        y_true=y_stage1[val_idx],
        probs=stage1_probs_all[val_idx],
        cfg=cfg,
    )
    stage1_test_summary = compute_binary_metrics(
        y_true=y_stage1[te_idx],
        probs=stage1_probs_all[te_idx],
        threshold=stage1_threshold,
    )
    stage1_report = build_thresholded_report(
        y_true=y_stage1[te_idx],
        probs=stage1_probs_all[te_idx],
        threshold=stage1_threshold,
        positive_name=active_pattern,
    )

    pd.DataFrame(stage1_search_rows).to_csv(
        os.path.join(stage1_dir, "threshold_search.csv"), index=False
    )
    _save_json(
        os.path.join(stage1_dir, "threshold_metrics.json"),
        {
            "bandwidth": float(bandwidth),
            "rule_labeled_anchor_count": {
                "total": int(len(rule_labeled)),
                "positive": int(rule_positive_count),
                "negative": int(rule_negative_count),
            },
            "dataset_summary": stage1_data["summary"],
            "val_best": stage1_val_summary,
            "test": stage1_test_summary,
            "threshold": float(stage1_threshold),
            "classification_report": stage1_report,
        },
    )
    stage1_metadata.assign(stage1_prob=stage1_probs_all).to_csv(
        os.path.join(stage1_dir, "anchor_labels.csv"),
        index=False,
    )
    print(
        "[stage1] Tuned threshold — "
        f"{stage1_threshold:.3f} | val recall: {stage1_val_summary['recall']:.3f} | "
        f"test precision: {stage1_test_summary['precision']:.3f} | "
        f"test recall: {stage1_test_summary['recall']:.3f}"
    )

    print("\n>>> Step 3: Stage 2 trade dataset")
    trade_data = build_trade_dataset(
        anchor_metadata=stage1_metadata,
        df_ohlcv=df_15min,
        stage1_probs=stage1_probs_all,
        cfg=cfg,
        pattern_name=active_pattern,
    )
    X_stage2 = trade_data["X"]
    y_stage2 = trade_data["y"]
    net_returns = trade_data["net_returns"]
    trade_anchors = trade_data["anchors"]
    stage2_metadata = trade_data["metadata"]

    stage2_split_idx = purged_time_split_indices(trade_anchors, cfg)
    tr2_idx = stage2_split_idx["train"]
    val2_idx = stage2_split_idx["val"]
    te2_idx = stage2_split_idx["test"]

    print("\n>>> Step 4: Stage 2 training")
    stage2_model = build_stage2_model(cfg)
    stage2_model.fit(X_stage2[tr2_idx], y_stage2[tr2_idx])
    stage2_probs_all = predict_trade_probabilities(stage2_model, X_stage2)
    joblib.dump(stage2_model, os.path.join(stage2_dir, "stage2_model.joblib"))
    _save_json(os.path.join(stage2_dir, "feature_names.json"), trade_data["feature_names"])

    stage1_gate_val = stage2_metadata["stage1_prob"].to_numpy(dtype=float)[val2_idx] >= stage1_threshold
    stage1_gate_test = stage2_metadata["stage1_prob"].to_numpy(dtype=float)[te2_idx] >= stage1_threshold

    stage2_threshold, stage2_search_rows, stage2_val_summary = choose_trade_threshold(
        y_true=y_stage2[val2_idx],
        probs=stage2_probs_all[val2_idx],
        net_returns=net_returns[val2_idx],
        gate_mask=stage1_gate_val,
        cfg=cfg,
    )
    stage2_test_summary = compute_trade_signal_metrics(
        y_true=y_stage2[te2_idx],
        probs=stage2_probs_all[te2_idx],
        net_returns=net_returns[te2_idx],
        threshold=stage2_threshold,
        gate_mask=stage1_gate_test,
    )
    stage2_report = build_trade_classification_report(
        y_true=y_stage2[te2_idx],
        probs=stage2_probs_all[te2_idx],
        threshold=stage2_threshold,
        gate_mask=stage1_gate_test,
    )
    stage2_coefficients = extract_stage2_coefficients(stage2_model, trade_data["feature_names"])

    pd.DataFrame(stage2_search_rows).to_csv(
        os.path.join(stage2_dir, "threshold_search.csv"), index=False
    )
    stage2_metadata.iloc[te2_idx].assign(
        stage2_prob=stage2_probs_all[te2_idx],
        trade_signal=stage1_gate_test & (stage2_probs_all[te2_idx] >= stage2_threshold),
    ).to_csv(os.path.join(stage2_dir, "test_trade_candidates.csv"), index=False)
    _save_json(
        os.path.join(stage2_dir, "trade_metrics.json"),
        {
            "val_best": stage2_val_summary,
            "test": stage2_test_summary,
            "threshold": float(stage2_threshold),
            "classification_report": stage2_report,
            "top_coefficients": stage2_coefficients[:20],
        },
    )
    print(
        "[stage2] Tuned threshold — "
        f"{stage2_threshold:.3f} | test trades: {stage2_test_summary['executed_trades']} | "
        f"test total_net_return_pct: {stage2_test_summary['total_net_return_pct']:.3f}"
    )

    print("\n>>> Step 5: Profitability comparison")
    stage1_only_summary = compute_trade_signal_metrics(
        y_true=y_stage2[te2_idx],
        probs=np.ones(len(te2_idx), dtype=float),
        net_returns=net_returns[te2_idx],
        threshold=0.5,
        gate_mask=stage1_gate_test,
    )
    stage1_only_summary["strategy"] = "stage1_only"
    filtered_summary = dict(stage2_test_summary)
    filtered_summary["strategy"] = "stage1_plus_stage2"
    comparison_df = pd.DataFrame([stage1_only_summary, filtered_summary])
    comparison_df.to_csv(os.path.join(stage2_dir, "profitability_comparison.csv"), index=False)
    print(comparison_df.to_string(index=False))

    print("\n>>> Step 6: Paper replay")
    replay_summary = run_paper_replay(
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        df_ohlcv=df_15min,
        feature_frame=trade_data["feature_frame"],
        cfg=cfg,
        device=device,
        stage1_threshold=stage1_threshold,
        stage2_threshold=stage2_threshold,
        save_dir=replay_dir,
    )

    print("\n>>> Step 7: Visualisations")
    render_product_visualisations(
        df_ohlcv=df_15min,
        cfg=cfg,
        product_root=product_root,
        stage1_threshold=stage1_threshold,
    )

    _save_json(
        os.path.join(product_root, "summary.json"),
        {
            "pattern": active_pattern,
            "rule_labeled_anchor_count": {
                "total": int(len(rule_labeled)),
                "positive": int(rule_positive_count),
                "negative": int(rule_negative_count),
            },
            "stage1_dataset": stage1_data["summary"],
            "stage1_threshold": float(stage1_threshold),
            "stage2_threshold": float(stage2_threshold),
            "stage1_test": stage1_test_summary,
            "stage2_test": stage2_test_summary,
            "stage1_only_test_profitability": stage1_only_summary,
            "paper_replay": replay_summary,
        },
    )

    print(f"\nOutputs saved under: {product_root}\n")


if __name__ == "__main__":
    main()
