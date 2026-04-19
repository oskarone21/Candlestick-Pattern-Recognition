from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from pandas.errors import EmptyDataError

try:
    from scripts._bootstrap import REPO_ROOT, ensure_repo_root
except ImportError:
    from _bootstrap import REPO_ROOT, ensure_repo_root

ensure_repo_root()

from candlestick.config import ensure_dir, load_config
from candlestick.project_utils import load_processed_prices
from candlestick.trading.backtest import run_backtest_for_predictions

DEFAULT_PRESENTATION_MIN_VAL_SUPPORT = 5
DEFAULT_PRESENTATION_MIN_TEST_SUPPORT = 5
DEFAULT_PRESENTATION_MIN_VISIBLE_PATTERNS = 3
DEFAULT_PRESENTATION_MIN_CHAMPION_F1 = 0.25


class PairBacktestRow(TypedDict):
    pattern: str
    model: str
    selection_f1: float
    selection_precision: float
    selection_recall: float
    f1: float
    precision: float
    recall: float
    pr_auc: float
    threshold: float
    support_positive: int
    support_negative: int
    train_positive_support: int
    val_positive_support: int
    test_positive_support: int
    presentation_eligible: bool
    trades: int
    total_pnl: float
    win_rate: float
    sharpe: float
    profit_factor: float
    max_drawdown: float
    expectancy: float


class ChampionPayloadRow(TypedDict):
    pattern: str
    model: str
    selection_split: str
    selection_f1: float
    threshold: float
    test_f1: float
    test_precision: float
    test_recall: float
    test_pr_auc: float
    train_positive_support: int
    val_positive_support: int
    test_positive_support: int
    presentation_eligible: bool
    trades: int
    total_pnl: float
    win_rate: float
    sharpe: float
    profit_factor: float
    max_drawdown: float
    expectancy: float


class PatternSupportRow(TypedDict):
    train_positive_support: int
    val_positive_support: int
    test_positive_support: int
    presentation_eligible: bool


class PresentationConfig(TypedDict):
    minimum_validation_positive_support: int
    minimum_test_positive_support: int
    minimum_visible_patterns: int
    minimum_champion_f1: float


def _resolve_run_name(metrics_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit

    candidates: list[tuple[float, str]] = []
    for child in metrics_dir.iterdir():
        summary = child / "model_comparison_summary.csv"
        if child.is_dir() and summary.exists():
            candidates.append((summary.stat().st_mtime, child.name))

    if not candidates:
        raise FileNotFoundError(f"No metric runs found under {metrics_dir}")

    candidates.sort(reverse=True)
    return candidates[0][1]


def _iso_or_none(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.isoformat()


def _float_or_none(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        output = float(value)
        return output if math.isfinite(output) else None
    except Exception:
        return None


def _int_or_zero(value: Any) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    return int(value)


def _bool_or_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return bool(value)


def _row_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    parsed = _float_or_none(value)
    return float(default if parsed is None else parsed)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _presentation_config(cfg: dict[str, Any] | None = None) -> PresentationConfig:
    presentation_cfg = cfg.get("dashboard", {}).get("presentation", {}) if cfg else {}
    return {
        "minimum_validation_positive_support": int(
            presentation_cfg.get("minimum_validation_positive_support", DEFAULT_PRESENTATION_MIN_VAL_SUPPORT)
        ),
        "minimum_test_positive_support": int(
            presentation_cfg.get("minimum_test_positive_support", DEFAULT_PRESENTATION_MIN_TEST_SUPPORT)
        ),
        "minimum_visible_patterns": int(
            presentation_cfg.get("minimum_visible_patterns", DEFAULT_PRESENTATION_MIN_VISIBLE_PATTERNS)
        ),
        "minimum_champion_f1": float(
            presentation_cfg.get("minimum_champion_f1", DEFAULT_PRESENTATION_MIN_CHAMPION_F1)
        ),
    }


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _sanitize_for_json(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _relative_path(path: Path, root: Path) -> str:
    return str(Path(os.path.relpath(path, root)))


def _load_metric_frames(metrics_root: Path, backtest_root: Path, gallery_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    summary_path = metrics_root / "model_comparison_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Metrics summary not found: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    champions_df = pd.read_csv(metrics_root / "champions.csv") if (metrics_root / "champions.csv").exists() else pd.DataFrame()
    champion_backtest = (
        pd.read_csv(backtest_root / "backtest_summary.csv")
        if (backtest_root / "backtest_summary.csv").exists()
        else pd.DataFrame()
    )
    macro = _read_json(metrics_root / "macro_summary.json")
    gallery_summary = _read_json(gallery_root / "gallery_summary.json")
    return summary_df, champions_df, champion_backtest, macro, gallery_summary


def _pattern_support_from_split(
    split_df: pd.DataFrame,
    presentation_cfg: PresentationConfig | None = None,
) -> PatternSupportRow:
    presentation_cfg = presentation_cfg or _presentation_config()
    positives = (
        split_df.groupby("split")["label"].sum().to_dict()
        if not split_df.empty and {"split", "label"}.issubset(split_df.columns)
        else {}
    )
    train_positive_support = int(positives.get("train", 0))
    val_positive_support = int(positives.get("val", 0))
    test_positive_support = int(positives.get("test", 0))
    return {
        "train_positive_support": train_positive_support,
        "val_positive_support": val_positive_support,
        "test_positive_support": test_positive_support,
        "presentation_eligible": (
            val_positive_support >= presentation_cfg["minimum_validation_positive_support"]
            and test_positive_support >= presentation_cfg["minimum_test_positive_support"]
        ),
    }


def _load_pattern_support(
    metrics_root: Path,
    summary_df: pd.DataFrame,
    presentation_cfg: PresentationConfig,
) -> dict[str, PatternSupportRow]:
    support_map: dict[str, PatternSupportRow] = {}
    for pattern in sorted(summary_df["pattern"].astype(str).unique().tolist()) if not summary_df.empty else []:
        split_path = metrics_root / pattern / "split_metadata.csv"
        if not split_path.exists():
            support_map[pattern] = {
                "train_positive_support": 0,
                "val_positive_support": 0,
                "test_positive_support": 0,
                "presentation_eligible": False,
            }
            continue
        support_map[pattern] = _pattern_support_from_split(pd.read_csv(split_path), presentation_cfg)
    return support_map


def _annotate_with_support(summary_df: pd.DataFrame, support_map: dict[str, PatternSupportRow]) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()

    annotated = summary_df.copy()
    annotated["train_positive_support"] = annotated["pattern"].map(
        lambda pattern: support_map.get(str(pattern), {}).get("train_positive_support", 0)
    )
    annotated["val_positive_support"] = annotated["pattern"].map(
        lambda pattern: support_map.get(str(pattern), {}).get("val_positive_support", 0)
    )
    annotated["test_positive_support"] = annotated["pattern"].map(
        lambda pattern: support_map.get(str(pattern), {}).get("test_positive_support", 0)
    )
    annotated["presentation_eligible"] = annotated["pattern"].map(
        lambda pattern: support_map.get(str(pattern), {}).get("presentation_eligible", False)
    )
    return annotated


def _gallery_samples(gallery_root: Path, output_dir: Path) -> dict[str, Any]:
    if not gallery_root.exists():
        return {}

    gallery: dict[str, Any] = {}
    for pattern_dir in sorted(path for path in gallery_root.iterdir() if path.is_dir()):
        buckets: dict[str, Any] = {}
        for bucket_dir in sorted(path for path in pattern_dir.iterdir() if path.is_dir()):
            index_path = bucket_dir / "index.csv"
            if not index_path.exists():
                continue

            try:
                frame = pd.read_csv(index_path)
            except EmptyDataError:
                continue
            buckets[bucket_dir.name] = [
                {
                    "image_path": _relative_path(Path(row["file"]), REPO_ROOT),
                    "image_src": _relative_path(Path(row["file"]), output_dir),
                    "window_end_ts": _iso_or_none(row.get("window_end_ts")),
                    "proba": _float_or_none(row.get("proba")),
                    "label": _int_or_zero(row.get("label")),
                    "y_pred": _int_or_zero(row.get("y_pred")),
                }
                for _, row in frame.head(3).iterrows()
            ]
        gallery[pattern_dir.name] = buckets
    return gallery


def _build_pair_row(row: pd.Series, summary: dict[str, Any]) -> PairBacktestRow:
    return {
        "pattern": str(row["pattern"]),
        "model": str(row["model"]),
        "selection_f1": _row_float(row, "selection_f1"),
        "selection_precision": _row_float(row, "selection_precision"),
        "selection_recall": _row_float(row, "selection_recall"),
        "f1": _row_float(row, "f1"),
        "precision": _row_float(row, "precision"),
        "recall": _row_float(row, "recall"),
        "pr_auc": _row_float(row, "pr_auc"),
        "threshold": _row_float(row, "threshold", 0.5),
        "support_positive": _int_or_zero(row["support_positive"]),
        "support_negative": _int_or_zero(row["support_negative"]),
        "train_positive_support": _int_or_zero(row.get("train_positive_support")),
        "val_positive_support": _int_or_zero(row.get("val_positive_support")),
        "test_positive_support": _int_or_zero(row.get("test_positive_support")),
        "presentation_eligible": _bool_or_false(row.get("presentation_eligible")),
        "trades": _int_or_zero(summary["trades"]),
        "total_pnl": float(summary["total_pnl"]),
        "win_rate": float(summary["win_rate"]),
        "sharpe": float(summary["sharpe"]),
        "profit_factor": float(summary["profit_factor"]),
        "max_drawdown": float(summary["max_drawdown"]),
        "expectancy": float(summary["expectancy"]),
    }


def _pair_curve_points(curve_frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "trade_number": int(idx + 1),
            "exit_ts": trade["exit_ts"].isoformat(),
            "entry_ts": _iso_or_none(trade.get("entry_ts")),
            "net_pnl": float(trade["net_pnl"]),
            "gross_pnl": float(trade["gross_pnl"]),
            "cumulative_pnl": float(trade["cumulative_pnl"]),
            "return": float(trade["return"]),
            "direction": str(trade["direction"]),
            "exit_reason": str(trade["exit_reason"]),
            "proba": float(trade["proba"]),
            "label": int(trade["label"]),
        }
        for idx, trade in curve_frame.iterrows()
    ]


def _aggregate_curve_points(frame: pd.DataFrame, label_key: str) -> list[dict[str, Any]]:
    return [
        {
            "trade_number": int(idx + 1),
            "exit_ts": trade["exit_ts"].isoformat(),
            "net_pnl": float(trade["net_pnl"]),
            "cumulative_pnl": float(trade["cumulative_pnl"]),
            label_key: str(trade[label_key]),
            "exit_reason": str(trade["exit_reason"]),
        }
        for idx, trade in frame.iterrows()
    ]


def _curve_frame(trades_df: pd.DataFrame) -> pd.DataFrame:
    curve_frame = trades_df.copy()
    curve_frame["exit_ts"] = pd.to_datetime(curve_frame["exit_ts"], errors="coerce")
    curve_frame = curve_frame.dropna(subset=["exit_ts"]).sort_values("exit_ts").reset_index(drop=True)
    curve_frame["cumulative_pnl"] = curve_frame["net_pnl"].cumsum()
    return curve_frame


def _build_backtest_payload(
    cfg: dict[str, Any],
    prices: pd.DataFrame,
    metrics_root: Path,
    summary_df: pd.DataFrame,
) -> dict[str, Any]:
    pair_rows: list[PairBacktestRow] = []
    pair_curves: list[dict[str, Any]] = []
    model_trade_frames: dict[str, list[pd.DataFrame]] = {}
    pattern_trade_frames: dict[str, list[pd.DataFrame]] = {}

    for _, row in summary_df.iterrows():
        pattern = str(row["pattern"])
        model = str(row["model"])
        pred_path = metrics_root / pattern / f"{model}_test_predictions.csv"
        if not pred_path.exists():
            continue

        pred_df = pd.read_csv(pred_path)
        trades_df, summary = run_backtest_for_predictions(
            price_df=prices,
            pred_df=pred_df,
            cfg=cfg,
            threshold=float(row["threshold"]),
        )

        pair_rows.append(_build_pair_row(row, summary))

        if trades_df.empty:
            pair_curves.append(
                {
                    "scope": "pair",
                    "series_id": f"{model}::{pattern}",
                    "model": model,
                    "pattern": pattern,
                    "trades": 0,
                    "total_pnl": 0.0,
                    "points": [],
                }
            )
            continue

        curve_frame = _curve_frame(trades_df)
        pair_curves.append(
            {
                "scope": "pair",
                "series_id": f"{model}::{pattern}",
                "model": model,
                "pattern": pattern,
                "trades": int(len(curve_frame)),
                "total_pnl": float(curve_frame["net_pnl"].sum()),
                "points": _pair_curve_points(curve_frame),
            }
        )
        model_trade_frames.setdefault(model, []).append(curve_frame.assign(pattern=pattern))
        pattern_trade_frames.setdefault(pattern, []).append(curve_frame.assign(model=model))

    aggregate_curves: list[dict[str, Any]] = []
    for model, frames in sorted(model_trade_frames.items()):
        frame = pd.concat(frames, ignore_index=True).sort_values("exit_ts").reset_index(drop=True)
        frame["cumulative_pnl"] = frame["net_pnl"].cumsum()
        aggregate_curves.append(
            {
                "scope": "model",
                "series_id": model,
                "label": model,
                "model": model,
                "points": _aggregate_curve_points(frame, "pattern"),
            }
        )

    for pattern, frames in sorted(pattern_trade_frames.items()):
        frame = pd.concat(frames, ignore_index=True).sort_values("exit_ts").reset_index(drop=True)
        frame["cumulative_pnl"] = frame["net_pnl"].cumsum()
        aggregate_curves.append(
            {
                "scope": "pattern",
                "series_id": pattern,
                "label": pattern,
                "pattern": pattern,
                "points": _aggregate_curve_points(frame, "model"),
            }
        )

    return {
        "pair_rows": sorted(pair_rows, key=lambda item: (item["pattern"], item["model"])),
        "pair_curves": pair_curves,
        "aggregate_curves": aggregate_curves,
    }


def _build_metrics_details(metrics_root: Path, summary_df: pd.DataFrame) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        pattern = str(row["pattern"])
        model = str(row["model"])
        detail = _read_json(metrics_root / pattern / f"{model}.json")
        details.append(
            {
                "pattern": pattern,
                "model": model,
                "artifact_path": detail.get("artifact_path"),
                "threshold": float(detail.get("threshold", row["threshold"])),
                "accuracy": _float_or_none(detail.get("accuracy")),
                "f1_ci_low": _float_or_none(detail.get("confidence_intervals", {}).get("f1", {}).get("ci_low")),
                "f1_ci_high": _float_or_none(detail.get("confidence_intervals", {}).get("f1", {}).get("ci_high")),
                "precision_ci_low": _float_or_none(
                    detail.get("confidence_intervals", {}).get("precision", {}).get("ci_low")
                ),
                "precision_ci_high": _float_or_none(
                    detail.get("confidence_intervals", {}).get("precision", {}).get("ci_high")
                ),
                "recall_ci_low": _float_or_none(
                    detail.get("confidence_intervals", {}).get("recall", {}).get("ci_low")
                ),
                "recall_ci_high": _float_or_none(
                    detail.get("confidence_intervals", {}).get("recall", {}).get("ci_high")
                ),
                "confusion_matrix": detail.get("confusion_matrix", {}),
                "train_positive_support": _int_or_zero(row.get("train_positive_support")),
                "val_positive_support": _int_or_zero(row.get("val_positive_support")),
                "test_positive_support": _int_or_zero(row.get("test_positive_support")),
                "presentation_eligible": _bool_or_false(row.get("presentation_eligible")),
            }
        )
    return details


def _build_champion_rows(
    champions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    pair_backtest_df: pd.DataFrame,
) -> list[ChampionPayloadRow]:
    if champions_df.empty:
        return []

    pair_lookup = {
        (str(row["pattern"]), str(row["model"])): row
        for row in pair_backtest_df.to_dict(orient="records")
    }
    champion_rows: list[ChampionPayloadRow] = []

    for _, row in champions_df.iterrows():
        pattern = str(row["pattern"])
        model = str(row["champion_model"])
        metric_row = summary_df[(summary_df["pattern"] == pattern) & (summary_df["model"] == model)].iloc[0]
        backtest_row = pair_lookup.get((pattern, model), {})
        champion_rows.append(
            {
                "pattern": pattern,
                "model": model,
                "selection_split": str(row["selection_split"]),
                "selection_f1": _row_float(row, "selection_f1"),
                "threshold": _row_float(row, "threshold", 0.5),
                "test_f1": _row_float(metric_row, "f1"),
                "test_precision": _row_float(metric_row, "precision"),
                "test_recall": _row_float(metric_row, "recall"),
                "test_pr_auc": _row_float(metric_row, "pr_auc"),
                "train_positive_support": _int_or_zero(metric_row.get("train_positive_support")),
                "val_positive_support": _int_or_zero(metric_row.get("val_positive_support")),
                "test_positive_support": _int_or_zero(metric_row.get("test_positive_support")),
                "presentation_eligible": _bool_or_false(metric_row.get("presentation_eligible")),
                "trades": _int_or_zero(backtest_row.get("trades") or backtest_row.get("n_trades")),
                "total_pnl": _float_or_none(backtest_row.get("total_pnl")) or 0.0,
                "win_rate": _float_or_none(backtest_row.get("win_rate")) or 0.0,
                "sharpe": _float_or_none(backtest_row.get("sharpe")) or 0.0,
                "profit_factor": _float_or_none(backtest_row.get("profit_factor")) or 0.0,
                "max_drawdown": _float_or_none(backtest_row.get("max_drawdown")) or 0.0,
                "expectancy": _float_or_none(backtest_row.get("expectancy")) or 0.0,
            }
        )

    return champion_rows


def _payload_meta(
    run_name: str,
    output_dir: Path,
    metrics_root: Path,
    backtest_root: Path,
    gallery_root: Path,
    summary_df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "output_dir": _relative_path(output_dir, REPO_ROOT),
        "source_paths": {
            "metrics": _relative_path(metrics_root, REPO_ROOT),
            "backtest": _relative_path(backtest_root, REPO_ROOT),
            "gallery": _relative_path(gallery_root, REPO_ROOT),
        },
        "models": sorted(summary_df["model"].unique().tolist()) if not summary_df.empty else [],
        "patterns": sorted(summary_df["pattern"].unique().tolist()) if not summary_df.empty else [],
    }


def _payload_hero(
    summary_df: pd.DataFrame,
    pair_backtest_df: pd.DataFrame,
    champion_rows: list[ChampionPayloadRow],
    macro: dict[str, Any],
) -> dict[str, Any]:
    visible_summary = summary_df[summary_df["presentation_eligible"]].copy() if not summary_df.empty else summary_df
    visible_pair_backtest = (
        pair_backtest_df[pair_backtest_df["presentation_eligible"]].copy()
        if not pair_backtest_df.empty
        else pair_backtest_df
    )
    visible_champions = [row for row in champion_rows if row["presentation_eligible"]]

    top_f1 = (
        visible_summary.sort_values(["f1", "pr_auc"], ascending=False).iloc[0].to_dict()
        if not visible_summary.empty
        else {}
    )
    profitable_pairs = (
        visible_pair_backtest[(visible_pair_backtest["trades"] > 0) & (visible_pair_backtest["total_pnl"] > 0)].copy()
        if not visible_pair_backtest.empty
        else visible_pair_backtest
    )
    top_pnl = (
        profitable_pairs.sort_values(["total_pnl", "profit_factor"], ascending=False).iloc[0].to_dict()
        if not profitable_pairs.empty
        else {}
    )
    best_model_avg_f1 = (
        visible_summary.groupby("model")["f1"].mean().sort_values(ascending=False).reset_index().iloc[0].to_dict()
        if not visible_summary.empty
        else {}
    )
    champion_positive = sum(1 for row in visible_champions if row["total_pnl"] > 0)

    return {
        "patterns_covered": int(visible_summary["pattern"].nunique()) if not visible_summary.empty else 0,
        "model_pair_count": int(len(visible_summary)),
        "champion_positive_patterns": int(champion_positive),
        "champion_total_patterns": int(len(visible_champions)),
        "best_test_f1_pair": top_f1,
        "best_backtest_pair": top_pnl,
        "best_model_by_mean_f1": best_model_avg_f1,
        "macro": macro,
    }


def _build_presentation_payload(
    summary_df: pd.DataFrame,
    champion_rows: list[ChampionPayloadRow],
    presentation_cfg: PresentationConfig,
) -> dict[str, Any]:
    support_rows = (
        summary_df[
            [
                "pattern",
                "train_positive_support",
                "val_positive_support",
                "test_positive_support",
                "presentation_eligible",
            ]
        ]
        .drop_duplicates(subset=["pattern"])
        .sort_values("pattern")
        if not summary_df.empty
        else pd.DataFrame()
    )
    visible_patterns = (
        support_rows.loc[support_rows["presentation_eligible"], "pattern"].astype(str).tolist()
        if not support_rows.empty
        else []
    )
    hidden_patterns = (
        support_rows.loc[~support_rows["presentation_eligible"], "pattern"].astype(str).tolist()
        if not support_rows.empty
        else []
    )
    visible_champions = [row for row in champion_rows if row["presentation_eligible"]]
    mean_visible_champion_f1 = (
        float(pd.Series([row["test_f1"] for row in visible_champions], dtype=float).mean())
        if visible_champions
        else 0.0
    )
    mean_visible_champion_pr_auc = (
        float(pd.Series([row["test_pr_auc"] for row in visible_champions], dtype=float).mean())
        if visible_champions
        else 0.0
    )
    has_strong_champion = any(
        row["test_f1"] >= presentation_cfg["minimum_champion_f1"] for row in visible_champions
    )
    presentation_eligible = (
        len(visible_patterns) >= presentation_cfg["minimum_visible_patterns"] and has_strong_champion
    )
    quality_score = (
        len(visible_patterns) * 100.0
        + mean_visible_champion_f1 * 10.0
        + mean_visible_champion_pr_auc
    )

    if presentation_eligible:
        presentation_reason = (
            f"Showing {len(visible_patterns)} supported patterns from the latest finished run; "
            f"{len(hidden_patterns)} hidden for lower support."
        )
    elif visible_patterns:
        presentation_reason = (
            f"Showing {len(visible_patterns)} supported patterns from the latest finished run; "
            f"{len(hidden_patterns)} hidden for lower support."
        )
    else:
        presentation_reason = "No patterns cleared the configured support floor in this finished run."

    return {
        "presentation_eligible": presentation_eligible,
        "visible_patterns": visible_patterns,
        "hidden_patterns": hidden_patterns,
        "quality_score": float(quality_score),
        "presentation_reason": presentation_reason,
        "mean_visible_champion_f1": mean_visible_champion_f1,
        "mean_visible_champion_pr_auc": mean_visible_champion_pr_auc,
    }


def _build_dashboard_payload(
    run_name: str,
    cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    metrics_root = Path(cfg["paths"].get("metrics_dir", "outputs/metrics")) / run_name
    backtest_root = Path(cfg["paths"].get("backtest_dir", "outputs/backtest")) / run_name
    gallery_root = Path(cfg["paths"].get("gallery_dir", "outputs/gallery")) / run_name

    summary_df, champions_df, champion_backtest, macro, gallery_summary = _load_metric_frames(
        metrics_root,
        backtest_root,
        gallery_root,
    )
    presentation_cfg = _presentation_config(cfg)
    support_map = _load_pattern_support(metrics_root, summary_df, presentation_cfg)
    summary_df = _annotate_with_support(summary_df, support_map)
    prices = load_processed_prices(cfg)
    backtest_payload = _build_backtest_payload(cfg, prices, metrics_root, summary_df)
    pair_backtest_df = pd.DataFrame(backtest_payload["pair_rows"])
    champion_rows = _build_champion_rows(champions_df, summary_df, pair_backtest_df)
    presentation = _build_presentation_payload(summary_df, champion_rows, presentation_cfg)

    return {
        "meta": _payload_meta(run_name, output_dir, metrics_root, backtest_root, gallery_root, summary_df),
        "hero": _payload_hero(summary_df, pair_backtest_df, champion_rows, macro),
        "presentation": presentation,
        "classification": {
            "summary_rows": summary_df.to_dict(orient="records"),
            "detail_rows": _build_metrics_details(metrics_root, summary_df),
            "champions": champion_rows,
        },
        "backtest": {
            "champion_rows": champion_rows,
            "champion_summary_rows": champion_backtest.to_dict(orient="records") if not champion_backtest.empty else [],
            **backtest_payload,
        },
        "gallery": {
            "summary": gallery_summary,
            "samples": _gallery_samples(gallery_root, output_dir),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static dashboard for the latest experiment results")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default="outputs/dashboard")
    parser.add_argument("--assets-dir", default="dashboard")
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only write dashboard data artifacts without copying the static dashboard assets.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics_dir = Path(cfg["paths"].get("metrics_dir", "outputs/metrics"))
    run_name = _resolve_run_name(metrics_dir, args.run_name)
    output_dir = ensure_dir(Path(args.output_root) / run_name)

    if not args.data_only:
        assets_dir = Path(args.assets_dir)
        if not assets_dir.exists():
            raise FileNotFoundError(
                f"Dashboard assets not found at {assets_dir}. Create the static dashboard files first."
            )
        shutil.copytree(assets_dir, output_dir, dirs_exist_ok=True)

    cfg = load_config(args.config, set_overrides=[f"project.run_name={run_name}"])
    payload = _sanitize_for_json(_build_dashboard_payload(run_name, cfg, output_dir))

    with (output_dir / "data.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with (output_dir / "data.js").open("w", encoding="utf-8") as handle:
        handle.write("window.__DASHBOARD_DATA__ = ")
        json.dump(payload, handle)
        handle.write(";\n")

    mode = "data snapshot" if args.data_only else "dashboard"
    print(f"{mode.title()} built at: {output_dir}")


if __name__ == "__main__":
    main()
