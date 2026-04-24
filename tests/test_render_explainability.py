from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from chart_patterns.domain import COLUMN_LABEL, COLUMN_SPLIT, COLUMN_WINDOW_END_TS, MODEL_LOGREG, MODEL_TCN
from chart_patterns.datasets.window_builder import save_pattern_dataset
from chart_patterns.explainability import (
    _explain_model,
    _plot_temporal_heatmap,
    build_feature_names,
    prepare_pattern_data,
    resolve_pattern_artifacts,
    resolve_run_paths,
    run_explainability_pipeline,
    summarize_shap_values,
)
from chart_patterns.models.logreg_baseline import train_logreg
from chart_patterns.models.registry import save_model
from scripts import render_explainability


def _mock_cfg(tmp_path: Path, run_name: str = "explainability_test") -> dict:
    return {
        "project": {"run_name": run_name},
        "paths": {
            "metrics_dir": str(tmp_path / "metrics"),
            "checkpoints_dir": str(tmp_path / "checkpoints"),
            "datasets_dir": str(tmp_path / "datasets"),
            "explainability_dir": str(tmp_path / "explainability"),
        },
    }


def _write_mock_run(tmp_path: Path) -> tuple[dict, str, str]:
    cfg = _mock_cfg(tmp_path)
    run_paths = resolve_run_paths(cfg)
    pattern = "double_bottom"
    model_name = MODEL_LOGREG

    rng = np.random.default_rng(7)
    X = rng.normal(size=(18, 6, 5)).astype(np.float32)
    raw_signal = X[:, -1, 3] + (0.7 * X[:, -2, 0]) - (0.3 * X[:, -3, 4])
    y = (raw_signal > np.median(raw_signal)).astype(np.int64)

    split_labels = (["train"] * 10) + (["val"] * 4) + (["test"] * 4)
    meta = pd.DataFrame(
        {
            COLUMN_LABEL: y,
            COLUMN_WINDOW_END_TS: pd.date_range(
                "2024-01-02 09:30",
                periods=len(y),
                freq="15min",
                tz="America/New_York",
            ).astype(str),
            "anchor_idx": np.arange(len(y)),
        }
    )

    save_pattern_dataset(run_paths.datasets_root, X, y, meta, pattern)

    metrics_pattern_dir = run_paths.metrics_root / pattern
    metrics_pattern_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({COLUMN_SPLIT: split_labels}).to_csv(metrics_pattern_dir / "split_metadata.csv", index=False)
    pd.DataFrame(
        [
            {
                "pattern": pattern,
                "champion_model": model_name,
                "threshold": 0.5,
            }
        ]
    ).to_csv(run_paths.metrics_root / "champions.csv", index=False)

    X_train = X[:10].reshape(10, -1)
    model = train_logreg(X_train, y[:10], seed=11)
    save_model(model_name, model, run_paths.checkpoints_root / pattern / f"{model_name}.bin")
    return cfg, pattern, model_name


def test_build_feature_names_and_summary_rankings():
    assert callable(render_explainability.main)

    feature_names = build_feature_names(3, ["open", "close"])

    assert feature_names[:3] == ["t_minus_002_open", "t_minus_002_close", "t_minus_001_open"]
    assert feature_names[-1] == "t_minus_000_close"

    shap_values = np.array(
        [
            [[0.1, -0.2], [0.0, 0.5], [0.3, 0.1]],
            [[0.2, -0.1], [0.4, 0.1], [0.1, -0.2]],
        ],
        dtype=np.float64,
    )
    summary = summarize_shap_values(
        shap_values=shap_values,
        feature_columns=["open", "close"],
        feature_names=feature_names,
        top_features=4,
    )

    assert summary["feature_groups"].iloc[0]["feature_group"] == "close"
    assert summary["top_flat_features"].iloc[0]["feature"] == "t_minus_001_close"
    assert set(summary["time_profile"].columns) == {"time_offset", "bars_ago", "mean_abs_shap"}


def test_prepare_pattern_data_uses_requested_splits(tmp_path):
    cfg, pattern, model_name = _write_mock_run(tmp_path)
    run_paths = resolve_run_paths(cfg)
    artifacts = resolve_pattern_artifacts(run_paths, pattern, model_name)

    prepared = prepare_pattern_data(
        artifacts,
        explain_split="test",
        background_split="train",
        explain_size=3,
        background_size=5,
        seed=21,
    )

    assert prepared.X_explain.shape == (3, 6, 5)
    assert prepared.X_background.shape == (5, 6, 5)
    assert prepared.sample_meta[COLUMN_WINDOW_END_TS].nunique() == 3


def test_explain_model_supports_sequence_models(monkeypatch):
    class DummySequenceCore(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=(1, 2))

    class FakeGradientExplainer:
        def __init__(self, model, background):
            self.model = model
            self.background = background

        def shap_values(self, samples):
            return np.ones((*samples.shape, 1), dtype=np.float32) * 0.05

    monkeypatch.setattr(
        "chart_patterns.explainability.shap",
        SimpleNamespace(GradientExplainer=FakeGradientExplainer),
    )

    model_bundle = SimpleNamespace(
        model=DummySequenceCore(),
        normalization_mode="window_minmax",
        normalization_stats={},
    )
    X_background = np.arange(40, dtype=np.float32).reshape(2, 4, 5)
    X_explain = np.arange(60, dtype=np.float32).reshape(3, 4, 5)

    values, base_values, explained_output = _explain_model(
        model_name=MODEL_TCN,
        model=model_bundle,
        X_background=X_background,
        X_explain=X_explain,
    )

    assert values.shape == (3, 4, 5)
    assert base_values.shape == (3,)
    assert explained_output.shape == (3,)
    assert np.allclose(values, 0.05)


def test_run_explainability_pipeline_emits_artifacts(tmp_path):
    cfg, pattern, _ = _write_mock_run(tmp_path)

    summary = run_explainability_pipeline(
        cfg,
        explain_size=3,
        background_size=6,
        top_features=6,
        max_waterfalls=2,
        seed=9,
    )

    assert summary["patterns_total"] == 1
    assert summary["patterns_succeeded"] == 1
    assert summary["patterns_failed"] == 0

    output_root = Path(summary["output_root"])
    pattern_dir = output_root / pattern
    summary_path = output_root / "explainability_summary.json"
    pattern_summary_path = pattern_dir / "summary.json"
    shap_npz_path = pattern_dir / "shap_values.npz"
    heatmap_path = pattern_dir / "temporal_heatmap.png"
    feature_csv_path = pattern_dir / "feature_group_importance.csv"
    waterfall_paths = sorted(pattern_dir.glob("sample_*_waterfall.png"))

    assert summary_path.exists()
    assert pattern_summary_path.exists()
    assert shap_npz_path.exists()
    assert heatmap_path.exists()
    assert feature_csv_path.exists()
    assert len(waterfall_paths) == 2

    pattern_summary = json.loads(pattern_summary_path.read_text(encoding="utf-8"))
    assert pattern_summary["pattern"] == pattern
    assert pattern_summary["model_name"] == MODEL_LOGREG
    assert len(pattern_summary["top_features"]) == 6

    with np.load(shap_npz_path) as loaded:
        assert loaded["values"].shape == (3, 6, 5)
        assert loaded["predicted_proba"].shape == (3,)

    feature_importance = pd.read_csv(feature_csv_path)
    assert set(feature_importance.columns) == {"feature_group", "mean_abs_shap"}
    assert feature_importance["mean_abs_shap"].iloc[0] > 0.0


def test_temporal_heatmap_uses_light_readable_palette(tmp_path):
    from PIL import Image

    feature_columns = ["open", "high", "low", "close", "volume"]
    frame = pd.DataFrame(
        [
            {
                "bars_ago": bars_ago,
                "feature_group": feature,
                "mean_abs_shap": float((80 - bars_ago) * (idx + 1)) / 400.0,
            }
            for bars_ago in range(80)
            for idx, feature in enumerate(feature_columns)
        ]
    )
    output_path = tmp_path / "temporal_heatmap.png"

    _plot_temporal_heatmap(frame, feature_columns, output_path)

    assert output_path.exists()
    pixels = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.uint8)
    near_black_ratio = np.all(pixels < 30, axis=2).mean()
    assert near_black_ratio < 0.10
