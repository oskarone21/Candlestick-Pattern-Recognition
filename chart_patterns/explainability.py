from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "chart_patterns_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn

try:
    import shap
except ImportError as exc:  # pragma: no cover - exercised only when SHAP is absent.
    shap = None
    _SHAP_IMPORT_ERROR = exc
else:
    _SHAP_IMPORT_ERROR = None

from chart_patterns.config import ensure_dir
from chart_patterns.datasets.window_builder import FEATURE_COLUMNS
from chart_patterns.domain import (
    CLASSICAL_MODEL_NAMES,
    COLUMN_LABEL,
    COLUMN_SPLIT,
    COLUMN_WINDOW_END_TS,
    MODEL_HGB,
    MODEL_LOGREG,
    SEQUENCE_MODEL_NAMES,
)
from chart_patterns.features.tabular import flatten_sequence_features
from chart_patterns.models.registry import load_model, predict_model_proba
from chart_patterns.project_utils import run_name_from_cfg

CLASSICAL_MODEL_NAMES = set(CLASSICAL_MODEL_NAMES)


@dataclass(frozen=True)
class ExplainabilityRunPaths:
    run_name: str
    metrics_root: Path
    checkpoints_root: Path
    datasets_root: Path
    output_root: Path


@dataclass(frozen=True)
class PatternArtifacts:
    pattern: str
    model_name: str
    model_path: Path
    dataset_path: Path
    metadata_path: Path
    split_metadata_path: Path
    output_dir: Path


@dataclass(frozen=True)
class PreparedPatternData:
    pattern: str
    model_name: str
    X_background: np.ndarray
    X_explain: np.ndarray
    y_explain: np.ndarray
    sample_meta: pd.DataFrame
    feature_names: list[str]
    feature_columns: list[str]


class TorchProbabilityWrapper(nn.Module):
    def __init__(self, torch_model) -> None:
        super().__init__()
        self.core = torch_model.model
        self.core.eval()
        self.mode = str(torch_model.normalization_mode).lower()
        stats = torch_model.normalization_stats or {}
        mean = stats.get("mean")
        std = stats.get("std")
        self.register_buffer(
            "mean",
            None if mean is None else torch.tensor(mean, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "std",
            None if std is None else torch.tensor(std, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        if self.mode == "train_zscore" and self.mean is not None and self.std is not None:
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        elif self.mode == "window_minmax":
            window_min = torch.amin(x, dim=1, keepdim=True)
            window_max = torch.amax(x, dim=1, keepdim=True)
            denom = torch.where((window_max - window_min) < 1.0e-6, torch.ones_like(window_max), window_max - window_min)
            x = (x - window_min) / denom
        logits = self.core(x)
        return torch.sigmoid(logits).unsqueeze(-1)


def resolve_run_paths(cfg: dict[str, Any], run_name: str | None = None) -> ExplainabilityRunPaths:
    resolved_run_name = run_name or run_name_from_cfg(cfg)
    paths_cfg = cfg.get("paths", {})
    return ExplainabilityRunPaths(
        run_name=resolved_run_name,
        metrics_root=Path(paths_cfg.get("metrics_dir", "outputs/metrics")) / resolved_run_name,
        checkpoints_root=Path(paths_cfg.get("checkpoints_dir", "outputs/checkpoints")) / resolved_run_name,
        datasets_root=Path(paths_cfg.get("datasets_dir", "outputs/datasets")) / resolved_run_name,
        output_root=ensure_dir(Path(paths_cfg.get("explainability_dir", "outputs/explainability")) / resolved_run_name),
    )


def build_feature_names(
    sequence_length: int,
    feature_columns: list[str] | None = None,
) -> list[str]:
    columns = feature_columns or list(FEATURE_COLUMNS)
    names: list[str] = []
    for offset in range(sequence_length):
        bars_ago = sequence_length - offset - 1
        for column in columns:
            names.append(f"t_minus_{bars_ago:03d}_{column}")
    return names


def normalize_pattern_list(patterns: list[str] | None) -> set[str] | None:
    if not patterns:
        return None
    normalized: set[str] = set()
    for item in patterns:
        for part in str(item).split(","):
            candidate = part.strip()
            if candidate:
                normalized.add(candidate)
    return normalized or None


def load_champion_rows(metrics_root: Path, patterns: set[str] | None = None) -> pd.DataFrame:
    champions_path = metrics_root / "champions.csv"
    if not champions_path.exists():
        raise FileNotFoundError(
            f"Champion file not found: {champions_path}. Run scripts/run_experiment_suite.py first."
        )

    champions = pd.read_csv(champions_path)
    if champions.empty:
        raise ValueError(f"Champion file is empty: {champions_path}")

    if "champion_model" not in champions.columns and "model" in champions.columns:
        champions = champions.rename(columns={"model": "champion_model"})

    required_columns = {"pattern", "champion_model"}
    missing = required_columns.difference(champions.columns)
    if missing:
        raise ValueError(f"Champion file missing required columns: {sorted(missing)}")

    if patterns:
        champions = champions[champions["pattern"].astype(str).isin(patterns)].copy()
        if champions.empty:
            raise ValueError(f"No champion rows matched requested patterns: {sorted(patterns)}")

    return champions.reset_index(drop=True)


def resolve_pattern_artifacts(run_paths: ExplainabilityRunPaths, pattern: str, model_name: str) -> PatternArtifacts:
    artifacts = PatternArtifacts(
        pattern=pattern,
        model_name=model_name,
        model_path=run_paths.checkpoints_root / pattern / f"{model_name}.bin",
        dataset_path=run_paths.datasets_root / pattern / "dataset.npz",
        metadata_path=run_paths.datasets_root / pattern / "metadata.csv",
        split_metadata_path=run_paths.metrics_root / pattern / "split_metadata.csv",
        output_dir=ensure_dir(run_paths.output_root / pattern),
    )
    missing_paths = [
        str(path)
        for path in (
            artifacts.model_path,
            artifacts.dataset_path,
            artifacts.metadata_path,
            artifacts.split_metadata_path,
        )
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"Missing explainability artifacts for pattern `{pattern}` model `{model_name}`: {missing_paths}"
        )
    return artifacts


def _sample_indices(indices: np.ndarray, max_count: int, seed: int, labels: np.ndarray | None = None) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if max_count <= 0 or len(indices) <= max_count:
        return indices

    rng = np.random.default_rng(seed)
    if labels is None or len(labels) != len(indices):
        return np.sort(rng.choice(indices, size=max_count, replace=False))

    positives = indices[np.asarray(labels, dtype=np.int64) == 1]
    negatives = indices[np.asarray(labels, dtype=np.int64) == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return np.sort(rng.choice(indices, size=max_count, replace=False))

    pos_target = min(len(positives), max_count // 2)
    neg_target = min(len(negatives), max_count - pos_target)
    remainder = max_count - pos_target - neg_target
    if remainder > 0:
        extra_pos = min(len(positives) - pos_target, remainder)
        pos_target += extra_pos
        remainder -= extra_pos
    if remainder > 0:
        neg_target += min(len(negatives) - neg_target, remainder)

    chosen_pos = rng.choice(positives, size=pos_target, replace=False) if pos_target else np.array([], dtype=np.int64)
    chosen_neg = rng.choice(negatives, size=neg_target, replace=False) if neg_target else np.array([], dtype=np.int64)
    chosen = np.concatenate([chosen_pos, chosen_neg])
    return np.sort(chosen.astype(np.int64))


def _load_dataset_with_metadata(artifacts: PatternArtifacts) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    with np.load(artifacts.dataset_path) as loaded:
        X = np.asarray(loaded["X"], dtype=np.float32)
        y = np.asarray(loaded["y"], dtype=np.int64)

    meta = pd.read_csv(artifacts.metadata_path)
    split_meta = pd.read_csv(artifacts.split_metadata_path)
    if len(meta) != len(X) or len(split_meta) != len(X) or len(y) != len(X):
        raise ValueError(
            f"Dataset and metadata row counts disagree for pattern `{artifacts.pattern}`: "
            f"X={len(X)}, y={len(y)}, metadata={len(meta)}, split_metadata={len(split_meta)}"
        )

    merged = meta.copy()
    if COLUMN_SPLIT in split_meta.columns:
        merged[COLUMN_SPLIT] = split_meta[COLUMN_SPLIT].astype(str).values
    elif COLUMN_SPLIT not in merged.columns:
        merged[COLUMN_SPLIT] = "all"

    if COLUMN_LABEL not in merged.columns:
        merged[COLUMN_LABEL] = y

    return X, y, merged


def prepare_pattern_data(
    artifacts: PatternArtifacts,
    *,
    explain_split: str,
    background_split: str,
    explain_size: int,
    background_size: int,
    seed: int,
    feature_columns: list[str] | None = None,
) -> PreparedPatternData:
    X, y, meta = _load_dataset_with_metadata(artifacts)
    if X.ndim != 3:
        raise ValueError(f"Expected 3D dataset for pattern `{artifacts.pattern}`, got shape {X.shape}")
    if len(X) == 0:
        raise ValueError(f"Dataset is empty for pattern `{artifacts.pattern}`")

    split_series = meta[COLUMN_SPLIT].astype(str)
    explain_idx = np.flatnonzero(split_series == explain_split)
    if len(explain_idx) == 0:
        explain_idx = np.arange(len(X), dtype=np.int64)

    background_idx = np.flatnonzero(split_series == background_split)
    if len(background_idx) == 0:
        background_idx = np.flatnonzero(split_series != explain_split)
    if len(background_idx) == 0:
        background_idx = np.arange(len(X), dtype=np.int64)

    sampled_explain_idx = _sample_indices(
        explain_idx,
        max_count=explain_size,
        seed=seed,
        labels=y[explain_idx],
    )
    sampled_background_idx = _sample_indices(
        background_idx,
        max_count=background_size,
        seed=seed + 1,
        labels=y[background_idx],
    )

    if len(sampled_explain_idx) == 0 or len(sampled_background_idx) == 0:
        raise ValueError(f"Could not sample background/explain rows for pattern `{artifacts.pattern}`")

    columns = feature_columns or list(FEATURE_COLUMNS)
    return PreparedPatternData(
        pattern=artifacts.pattern,
        model_name=artifacts.model_name,
        X_background=X[sampled_background_idx].astype(np.float32, copy=False),
        X_explain=X[sampled_explain_idx].astype(np.float32, copy=False),
        y_explain=y[sampled_explain_idx].astype(np.int64, copy=False),
        sample_meta=meta.iloc[sampled_explain_idx].reset_index(drop=True),
        feature_names=build_feature_names(X.shape[1], columns),
        feature_columns=columns,
    )


def _model_input_for_name(model_name: str, X: np.ndarray) -> np.ndarray:
    if model_name in CLASSICAL_MODEL_NAMES:
        return flatten_sequence_features(X)
    return X


def _normalize_explanation_values(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim >= 1 and array.shape[-1] == 2:
        array = array[..., 1]
    if array.ndim >= 1 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


def _normalize_base_values(base_values: Any, n_samples: int) -> np.ndarray:
    array = np.asarray(base_values, dtype=np.float64)
    if array.ndim >= 1 and array.shape[-1] == 2:
        array = array[..., 1]
    if array.ndim >= 1 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 0:
        return np.full(n_samples, float(array), dtype=np.float64)
    if array.ndim > 1:
        array = np.reshape(array, (array.shape[0], -1))[:, 0]
    if len(array) == 1 and n_samples != 1:
        return np.full(n_samples, float(array[0]), dtype=np.float64)
    return array.astype(np.float64, copy=False)


def _reshape_flat_values(values: np.ndarray, sequence_length: int, feature_count: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values before reshaping, got shape {values.shape}")
    return values.reshape(values.shape[0], sequence_length, feature_count)

def explain_pattern(
    artifacts: PatternArtifacts,
    *,
    explain_split: str,
    background_split: str,
    explain_size: int,
    background_size: int,
    top_features: int,
    max_waterfalls: int,
    seed: int,
) -> dict[str, Any]:
    prepared = prepare_pattern_data(
        artifacts,
        explain_split=explain_split,
        background_split=background_split,
        explain_size=explain_size,
        background_size=background_size,
        seed=seed,
    )
    model = load_model(artifacts.model_name, artifacts.model_path)

    values_3d, base_values, explained_output = _explain_model(
        model_name=artifacts.model_name,
        model=model,
        X_background=prepared.X_background,
        X_explain=prepared.X_explain,
    )
    predicted_proba = predict_model_proba(
        artifacts.model_name,
        model,
        _model_input_for_name(artifacts.model_name, prepared.X_explain),
    )

    summary = summarize_shap_values(
        shap_values=values_3d,
        feature_columns=prepared.feature_columns,
        feature_names=prepared.feature_names,
        top_features=top_features,
    )
    artifact_files = write_pattern_outputs(
        artifacts=artifacts,
        prepared=prepared,
        shap_values=values_3d,
        base_values=base_values,
        explained_output=explained_output,
        predicted_proba=predicted_proba,
        top_features=top_features,
        max_waterfalls=max_waterfalls,
        summary=summary,
    )

    return {
        "pattern": artifacts.pattern,
        "model_name": artifacts.model_name,
        "status": "ok",
        "background_split": background_split,
        "explain_split": explain_split,
        "background_samples": int(len(prepared.X_background)),
        "explained_samples": int(len(prepared.X_explain)),
        "top_features": summary["top_flat_features"].head(min(top_features, 5))["feature"].tolist(),
        "artifacts": artifact_files,
    }


def _explain_model(
    *,
    model_name: str,
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if shap is None:
        raise ImportError(
            "SHAP is required for explainability generation. Install the `shap` package before running this script."
        ) from _SHAP_IMPORT_ERROR

    sequence_length = int(X_explain.shape[1])
    feature_count = int(X_explain.shape[2])

    if model_name == MODEL_LOGREG:
        background_2d = flatten_sequence_features(X_background)
        explain_2d = flatten_sequence_features(X_explain)
        transformed_background = model[:-1].transform(background_2d)
        transformed_explain = model[:-1].transform(explain_2d)
        explainer = shap.LinearExplainer(model.named_steps["clf"], transformed_background)
        explanation = explainer(transformed_explain)
        values = _reshape_flat_values(_normalize_explanation_values(explanation.values), sequence_length, feature_count)
        base_values = _normalize_base_values(explanation.base_values, len(X_explain))
        explained_output = base_values + values.reshape(len(X_explain), -1).sum(axis=1)
        return values, base_values, explained_output

    if model_name == MODEL_HGB:
        background_2d = flatten_sequence_features(X_background)
        explain_2d = flatten_sequence_features(X_explain)
        explainer = shap.TreeExplainer(model, data=background_2d, model_output="raw")
        explanation = explainer(explain_2d)
        values = _reshape_flat_values(_normalize_explanation_values(explanation.values), sequence_length, feature_count)
        base_values = _normalize_base_values(explanation.base_values, len(X_explain))
        explained_output = base_values + values.reshape(len(X_explain), -1).sum(axis=1)
        return values, base_values, explained_output

    if model_name in SEQUENCE_MODEL_NAMES:
        wrapper = TorchProbabilityWrapper(model)
        background_tensor = torch.tensor(X_background, dtype=torch.float32)
        explain_tensor = torch.tensor(X_explain, dtype=torch.float32)
        explainer = shap.GradientExplainer(wrapper, background_tensor)
        raw_values = explainer.shap_values(explain_tensor)
        values = _normalize_explanation_values(raw_values)
        if values.ndim != 3:
            values = np.squeeze(values, axis=-1)
        base_value = float(wrapper(background_tensor).detach().cpu().numpy().mean())
        base_values = np.full(len(X_explain), base_value, dtype=np.float64)
        explained_output = base_values + values.reshape(len(X_explain), -1).sum(axis=1)
        return values.astype(np.float64, copy=False), base_values, explained_output

    background_input = _model_input_for_name(model_name, X_background)
    explain_input = _model_input_for_name(model_name, X_explain)
    predict_fn = lambda arr: predict_model_proba(model_name, model, np.asarray(arr))
    explainer = shap.Explainer(predict_fn, background_input, seed=0)
    explanation = explainer(explain_input)
    values = _normalize_explanation_values(explanation.values)
    if values.ndim == 2:
        values = _reshape_flat_values(values, sequence_length, feature_count)
    base_values = _normalize_base_values(explanation.base_values, len(X_explain))
    explained_output = base_values + values.reshape(len(X_explain), -1).sum(axis=1)
    return values.astype(np.float64, copy=False), base_values, explained_output


def summarize_shap_values(
    *,
    shap_values: np.ndarray,
    feature_columns: list[str],
    feature_names: list[str],
    top_features: int,
) -> dict[str, pd.DataFrame]:
    if shap_values.ndim != 3:
        raise ValueError(f"Expected 3D SHAP values, got shape {shap_values.shape}")

    mean_abs = np.abs(shap_values).mean(axis=0)
    sequence_length, feature_count = mean_abs.shape
    flat_mean_abs = mean_abs.reshape(-1)

    flat_rows = pd.DataFrame(
        {
            "feature": feature_names,
            "time_offset": np.repeat(np.arange(sequence_length), feature_count),
            "bars_ago": np.repeat(np.arange(sequence_length - 1, -1, -1), feature_count),
            "feature_group": feature_columns * sequence_length,
            "mean_abs_shap": flat_mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)

    group_rows = (
        flat_rows.groupby("feature_group", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    )
    time_rows = (
        flat_rows.groupby(["time_offset", "bars_ago"], as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values(["time_offset"], ascending=[True], ignore_index=True)
    )
    heatmap_rows = flat_rows.sort_values(["time_offset", "feature_group"], ignore_index=True)

    return {
        "top_flat_features": flat_rows.head(top_features).reset_index(drop=True),
        "flat_features": flat_rows,
        "feature_groups": group_rows,
        "time_profile": time_rows,
        "heatmap": heatmap_rows,
    }


def _save_dataframe(path: Path, frame: pd.DataFrame) -> str:
    frame.to_csv(path, index=False)
    return str(path)


def _plot_feature_group_importance(frame: pd.DataFrame, output_path: Path) -> str:
    top = frame.sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(top["feature_group"], top["mean_abs_shap"], color="#2c7fb8")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("Feature")
    ax.set_title("Global feature importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _plot_temporal_heatmap(frame: pd.DataFrame, feature_columns: list[str], output_path: Path) -> str:
    pivot = frame.pivot(index="bars_ago", columns="feature_group", values="mean_abs_shap")
    pivot = pivot.reindex(columns=feature_columns)
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=30, ha="right")
    y_step = max(len(pivot.index) // 10, 1)
    tick_positions = np.arange(0, len(pivot.index), y_step)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([int(pivot.index[pos]) for pos in tick_positions])
    ax.set_xlabel("Feature")
    ax.set_ylabel("Bars ago")
    ax.set_title("Mean |SHAP| by bar and feature")
    ax.grid(color="#d9d9d9", linewidth=0.5, alpha=0.7)
    fig.colorbar(image, ax=ax, label="Mean |SHAP|")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _plot_top_window_features(frame: pd.DataFrame, output_path: Path) -> str:
    top = frame.sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#f03b20")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("Window feature")
    ax.set_title("Top window-level contributors")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _representative_sample_indices(predicted_proba: np.ndarray, max_waterfalls: int) -> list[int]:
    if max_waterfalls <= 0 or len(predicted_proba) == 0:
        return []
    ranking = np.argsort(predicted_proba)
    candidates = [
        int(ranking[0]),
        int(ranking[len(ranking) // 2]),
        int(ranking[-1]),
    ]
    deduped: list[int] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    if len(deduped) >= max_waterfalls:
        return deduped[:max_waterfalls]

    for candidate in ranking[::-1]:
        candidate_int = int(candidate)
        if candidate_int not in deduped:
            deduped.append(candidate_int)
        if len(deduped) >= max_waterfalls:
            break
    return deduped[:max_waterfalls]


def _plot_sample_waterfall(
    *,
    sample_index: int,
    shap_values: np.ndarray,
    feature_names: list[str],
    sample_row: pd.Series,
    predicted_proba: float,
    explained_output: float,
    output_path: Path,
    top_k: int = 10,
) -> str:
    flattened = shap_values[sample_index].reshape(-1)
    ranking = np.argsort(np.abs(flattened))[-top_k:]
    selected = flattened[ranking]
    selected_names = [feature_names[idx] for idx in ranking]

    order = np.argsort(selected)
    selected = selected[order]
    selected_names = [selected_names[idx] for idx in order]
    colors = ["#1a9850" if value >= 0 else "#d73027" for value in selected]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(selected_names, selected, color=colors)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ts_label = sample_row.get(COLUMN_WINDOW_END_TS, f"sample_{sample_index}")
    ax.set_xlabel("SHAP contribution")
    ax.set_ylabel("Feature")
    ax.set_title(
        f"Sample explanation: p={predicted_proba:.3f}, explained={explained_output:.3f}\nwindow_end={ts_label}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def write_pattern_outputs(
    *,
    artifacts: PatternArtifacts,
    prepared: PreparedPatternData,
    shap_values: np.ndarray,
    base_values: np.ndarray,
    explained_output: np.ndarray,
    predicted_proba: np.ndarray,
    top_features: int,
    max_waterfalls: int,
    summary: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    output_dir = artifacts.output_dir
    flat_csv = _save_dataframe(output_dir / "flat_feature_importance.csv", summary["flat_features"])
    grouped_csv = _save_dataframe(output_dir / "feature_group_importance.csv", summary["feature_groups"])
    time_csv = _save_dataframe(output_dir / "time_profile.csv", summary["time_profile"])
    top_csv = _save_dataframe(output_dir / "top_window_features.csv", summary["top_flat_features"])

    sample_scores = prepared.sample_meta.copy()
    sample_scores["predicted_proba"] = np.asarray(predicted_proba, dtype=np.float64)
    sample_scores["explained_output"] = np.asarray(explained_output, dtype=np.float64)
    sample_scores["base_value"] = np.asarray(base_values, dtype=np.float64)
    sample_scores_csv = _save_dataframe(output_dir / "sample_scores.csv", sample_scores)

    np.savez_compressed(
        output_dir / "shap_values.npz",
        values=np.asarray(shap_values, dtype=np.float32),
        base_values=np.asarray(base_values, dtype=np.float32),
        explained_output=np.asarray(explained_output, dtype=np.float32),
        predicted_proba=np.asarray(predicted_proba, dtype=np.float32),
    )

    figures = {
        "feature_group_importance": _plot_feature_group_importance(
            summary["feature_groups"], output_dir / "feature_group_importance.png"
        ),
        "temporal_heatmap": _plot_temporal_heatmap(
            summary["heatmap"], prepared.feature_columns, output_dir / "temporal_heatmap.png"
        ),
        "top_window_features": _plot_top_window_features(
            summary["top_flat_features"], output_dir / "top_window_features.png"
        ),
    }

    waterfall_paths: list[str] = []
    for sample_index in _representative_sample_indices(predicted_proba, max_waterfalls):
        waterfall_paths.append(
            _plot_sample_waterfall(
                sample_index=sample_index,
                shap_values=shap_values,
                feature_names=prepared.feature_names,
                sample_row=prepared.sample_meta.iloc[sample_index],
                predicted_proba=float(predicted_proba[sample_index]),
                explained_output=float(explained_output[sample_index]),
                output_path=output_dir / f"sample_{sample_index:02d}_waterfall.png",
                top_k=min(top_features, 10),
            )
        )

    pattern_summary = {
        "pattern": artifacts.pattern,
        "model_name": artifacts.model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "background_samples": int(len(prepared.X_background)),
        "explained_samples": int(len(prepared.X_explain)),
        "files": {
            "flat_feature_importance_csv": flat_csv,
            "feature_group_importance_csv": grouped_csv,
            "time_profile_csv": time_csv,
            "top_window_features_csv": top_csv,
            "sample_scores_csv": sample_scores_csv,
            "figures": figures,
            "waterfalls": waterfall_paths,
            "shap_values_npz": str(output_dir / "shap_values.npz"),
        },
        "top_features": summary["top_flat_features"].to_dict(orient="records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(pattern_summary, handle, indent=2)
    return pattern_summary["files"]


def run_explainability_pipeline(
    cfg: dict[str, Any],
    *,
    run_name: str | None = None,
    patterns: list[str] | None = None,
    explain_split: str = "test",
    background_split: str = "train",
    explain_size: int = 32,
    background_size: int = 64,
    top_features: int = 15,
    max_waterfalls: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    run_paths = resolve_run_paths(cfg, run_name=run_name)
    champions = load_champion_rows(run_paths.metrics_root, patterns=normalize_pattern_list(patterns))

    results: list[dict[str, Any]] = []
    for row in champions.to_dict(orient="records"):
        pattern = str(row["pattern"])
        model_name = str(row["champion_model"])
        try:
            artifacts = resolve_pattern_artifacts(run_paths, pattern, model_name)
            result = explain_pattern(
                artifacts,
                explain_split=explain_split,
                background_split=background_split,
                explain_size=explain_size,
                background_size=background_size,
                top_features=top_features,
                max_waterfalls=max_waterfalls,
                seed=seed,
            )
        except Exception as exc:
            result = {
                "pattern": pattern,
                "model_name": model_name,
                "status": "failed",
                "error": str(exc),
            }
        results.append(result)

    summary = {
        "run_name": run_paths.run_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(run_paths.output_root),
        "patterns_total": int(len(results)),
        "patterns_succeeded": int(sum(result["status"] == "ok" for result in results)),
        "patterns_failed": int(sum(result["status"] != "ok" for result in results)),
        "results": results,
    }
    with (run_paths.output_root / "explainability_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
