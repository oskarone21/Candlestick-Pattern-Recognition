from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from candlestick.config import ensure_dir, load_config
from candlestick.trading.backtest import run_backtest_for_predictions


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


def _load_prices(cfg: dict[str, Any]) -> pd.DataFrame:
    prices_path = Path(cfg["paths"].get("processed_15m_path", "data/processed/spy_15m.csv"))
    if not prices_path.exists():
        raise FileNotFoundError(f"Processed prices not found: {prices_path}")

    prices = pd.read_csv(prices_path)
    working_timezone = cfg.get("data_source", {}).get("timestamp", {}).get(
        "convert_to_timezone", "America/New_York"
    )
    prices["ts_event"] = pd.to_datetime(prices["ts_event"], errors="coerce", utc=True).dt.tz_convert(
        working_timezone
    )
    prices = prices.dropna(subset=["ts_event"]).copy()
    prices = prices[prices["symbol"].astype(str).str.upper() == "SPY"].sort_values("ts_event").reset_index(drop=True)
    return prices


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
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _int_or_zero(value: Any) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    return int(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(v) for v in value]
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


def _gallery_samples(gallery_root: Path, output_dir: Path) -> dict[str, Any]:
    if not gallery_root.exists():
        return {}

    out: dict[str, Any] = {}
    for pattern_dir in sorted(p for p in gallery_root.iterdir() if p.is_dir()):
        pattern_payload: dict[str, Any] = {}
        for bucket_dir in sorted(p for p in pattern_dir.iterdir() if p.is_dir()):
            index_path = bucket_dir / "index.csv"
            if not index_path.exists():
                continue
            frame = pd.read_csv(index_path)
            samples = []
            for _, row in frame.head(3).iterrows():
                file_path = Path(row["file"])
                samples.append(
                    {
                        "image_path": str(Path(shutil.os.path.relpath(file_path, REPO_ROOT))),
                        "image_src": str(Path(shutil.os.path.relpath(file_path, output_dir))),
                        "window_end_ts": _iso_or_none(row.get("window_end_ts")),
                        "proba": _float_or_none(row.get("proba")),
                        "label": _int_or_zero(row.get("label")),
                        "y_pred": _int_or_zero(row.get("y_pred")),
                    }
                )
            pattern_payload[bucket_dir.name] = samples
        out[pattern_dir.name] = pattern_payload
    return out


def _build_backtest_payload(
    cfg: dict[str, Any],
    prices: pd.DataFrame,
    metrics_root: Path,
    summary_df: pd.DataFrame,
) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []
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

        pair_row = {
            "pattern": pattern,
            "model": model,
            "selection_f1": float(row["selection_f1"]),
            "selection_precision": float(row["selection_precision"]),
            "selection_recall": float(row["selection_recall"]),
            "f1": float(row["f1"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "pr_auc": float(row["pr_auc"]),
            "threshold": float(row["threshold"]),
            "support_positive": _int_or_zero(row["support_positive"]),
            "support_negative": _int_or_zero(row["support_negative"]),
            "trades": _int_or_zero(summary["trades"]),
            "total_pnl": float(summary["total_pnl"]),
            "win_rate": float(summary["win_rate"]),
            "sharpe": float(summary["sharpe"]),
            "profit_factor": float(summary["profit_factor"]),
            "max_drawdown": float(summary["max_drawdown"]),
            "expectancy": float(summary["expectancy"]),
        }
        pair_rows.append(pair_row)

        if not trades_df.empty:
            curve_frame = trades_df.copy()
            curve_frame["exit_ts"] = pd.to_datetime(curve_frame["exit_ts"], errors="coerce")
            curve_frame = curve_frame.dropna(subset=["exit_ts"]).sort_values("exit_ts").reset_index(drop=True)
            curve_frame["cumulative_pnl"] = curve_frame["net_pnl"].cumsum()
            points = []
            for idx, trade in curve_frame.iterrows():
                points.append(
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
                )

            pair_curves.append(
                {
                    "scope": "pair",
                    "series_id": f"{model}::{pattern}",
                    "model": model,
                    "pattern": pattern,
                    "trades": int(len(curve_frame)),
                    "total_pnl": float(curve_frame["net_pnl"].sum()),
                    "points": points,
                }
            )
            model_trade_frames.setdefault(model, []).append(curve_frame.assign(pattern=pattern))
            pattern_trade_frames.setdefault(pattern, []).append(curve_frame.assign(model=model))
        else:
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
                "points": [
                    {
                        "trade_number": int(idx + 1),
                        "exit_ts": trade["exit_ts"].isoformat(),
                        "net_pnl": float(trade["net_pnl"]),
                        "cumulative_pnl": float(trade["cumulative_pnl"]),
                        "pattern": str(trade["pattern"]),
                        "exit_reason": str(trade["exit_reason"]),
                    }
                    for idx, trade in frame.iterrows()
                ],
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
                "points": [
                    {
                        "trade_number": int(idx + 1),
                        "exit_ts": trade["exit_ts"].isoformat(),
                        "net_pnl": float(trade["net_pnl"]),
                        "cumulative_pnl": float(trade["cumulative_pnl"]),
                        "model": str(trade["model"]),
                        "exit_reason": str(trade["exit_reason"]),
                    }
                    for idx, trade in frame.iterrows()
                ],
            }
        )

    return {
        "pair_rows": sorted(pair_rows, key=lambda x: (x["pattern"], x["model"])),
        "pair_curves": pair_curves,
        "aggregate_curves": aggregate_curves,
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

    summary_path = metrics_root / "model_comparison_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Metrics summary not found: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    champions_df = pd.read_csv(metrics_root / "champions.csv") if (metrics_root / "champions.csv").exists() else pd.DataFrame()
    macro = _read_json(metrics_root / "macro_summary.json")
    champion_backtest = pd.read_csv(backtest_root / "backtest_summary.csv") if (backtest_root / "backtest_summary.csv").exists() else pd.DataFrame()
    gallery_summary = _read_json(gallery_root / "gallery_summary.json")
    prices = _load_prices(cfg)

    metrics_details: list[dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        pattern = str(row["pattern"])
        model = str(row["model"])
        detail = _read_json(metrics_root / pattern / f"{model}.json")
        metrics_details.append(
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
                "recall_ci_low": _float_or_none(detail.get("confidence_intervals", {}).get("recall", {}).get("ci_low")),
                "recall_ci_high": _float_or_none(
                    detail.get("confidence_intervals", {}).get("recall", {}).get("ci_high")
                ),
                "confusion_matrix": detail.get("confusion_matrix", {}),
            }
        )

    backtest_payload = _build_backtest_payload(cfg, prices, metrics_root, summary_df)
    pair_backtest_df = pd.DataFrame(backtest_payload["pair_rows"])

    champion_rows: list[dict[str, Any]] = []
    if not champions_df.empty:
        champion_lookup = champion_backtest.set_index("pattern").to_dict(orient="index") if not champion_backtest.empty else {}
        for _, row in champions_df.iterrows():
            pattern = str(row["pattern"])
            model = str(row["champion_model"])
            metric_row = summary_df[(summary_df["pattern"] == pattern) & (summary_df["model"] == model)].iloc[0]
            bt = champion_lookup.get(pattern, {})
            champion_rows.append(
                {
                    "pattern": pattern,
                    "model": model,
                    "selection_split": str(row["selection_split"]),
                    "selection_f1": float(row["selection_f1"]),
                    "threshold": float(row["threshold"]),
                    "test_f1": float(metric_row["f1"]),
                    "test_precision": float(metric_row["precision"]),
                    "test_recall": float(metric_row["recall"]),
                    "test_pr_auc": float(metric_row["pr_auc"]),
                    "trades": _int_or_zero(bt.get("trades") or bt.get("n_trades")),
                    "total_pnl": _float_or_none(bt.get("total_pnl")) or 0.0,
                    "win_rate": _float_or_none(bt.get("win_rate")) or 0.0,
                    "sharpe": _float_or_none(bt.get("sharpe")) or 0.0,
                    "profit_factor": _float_or_none(bt.get("profit_factor")) or 0.0,
                    "max_drawdown": _float_or_none(bt.get("max_drawdown")) or 0.0,
                    "expectancy": _float_or_none(bt.get("expectancy")) or 0.0,
                }
            )

    top_f1 = summary_df.sort_values(["f1", "pr_auc"], ascending=False).iloc[0].to_dict() if not summary_df.empty else {}
    top_pnl = pair_backtest_df.sort_values(["total_pnl", "profit_factor"], ascending=False).iloc[0].to_dict() if not pair_backtest_df.empty else {}
    champion_positive = sum(1 for row in champion_rows if row["total_pnl"] > 0)
    best_model_avg_f1 = (
        summary_df.groupby("model")["f1"].mean().sort_values(ascending=False).reset_index().iloc[0].to_dict()
        if not summary_df.empty
        else {}
    )

    payload = {
        "meta": {
            "run_name": run_name,
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "output_dir": str(Path(shutil.os.path.relpath(output_dir, REPO_ROOT))),
            "source_paths": {
                "metrics": str(Path(shutil.os.path.relpath(metrics_root, REPO_ROOT))),
                "backtest": str(Path(shutil.os.path.relpath(backtest_root, REPO_ROOT))),
                "gallery": str(Path(shutil.os.path.relpath(gallery_root, REPO_ROOT))),
            },
            "models": sorted(summary_df["model"].unique().tolist()) if not summary_df.empty else [],
            "patterns": sorted(summary_df["pattern"].unique().tolist()) if not summary_df.empty else [],
        },
        "hero": {
            "patterns_covered": int(summary_df["pattern"].nunique()) if not summary_df.empty else 0,
            "model_pair_count": int(len(summary_df)),
            "champion_positive_patterns": int(champion_positive),
            "champion_total_patterns": int(len(champion_rows)),
            "best_test_f1_pair": top_f1,
            "best_backtest_pair": top_pnl,
            "best_model_by_mean_f1": best_model_avg_f1,
            "macro": macro,
        },
        "classification": {
            "summary_rows": summary_df.to_dict(orient="records"),
            "detail_rows": metrics_details,
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
    return payload


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
    payload = _build_dashboard_payload(run_name, cfg, output_dir)
    safe_payload = _sanitize_for_json(payload)
    with (output_dir / "data.json").open("w", encoding="utf-8") as f:
        json.dump(safe_payload, f, indent=2)
    with (output_dir / "data.js").open("w", encoding="utf-8") as f:
        f.write("window.__DASHBOARD_DATA__ = ")
        json.dump(safe_payload, f)
        f.write(";\n")

    mode = "data snapshot" if args.data_only else "dashboard"
    print(f"{mode.title()} built at: {output_dir}")


if __name__ == "__main__":
    main()
