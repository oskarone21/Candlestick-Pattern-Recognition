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

from chart_patterns.config import ensure_dir, load_config
from chart_patterns.domain import MODEL_LOGREG
from chart_patterns.project_utils import allowed_patterns_from_cfg, run_name_from_cfg
from chart_patterns.run_naming import (
    FINAL_RUN_MANIFEST_FILENAMES,
    canonical_profile_names,
    canonicalize_profile_name,
    final_run_dir_name,
    profile_override_path,
    profile_scan_dir_name,
    supported_profile_names,
)
from scripts.build_results_dashboard import build_dashboard_snapshot
from scripts.run_backtest import run_backtest
from scripts.run_experiment_suite import run_experiment_suite
from scripts.render_pattern_gallery import render_gallery

RULE_PROFILES = {
    profile_name: profile_override_path(profile_name)
    for profile_name in canonical_profile_names()
}


def _with_runtime_context(
    cfg: dict[str, Any],
    *,
    base_config: str,
    config_overrides: list[str],
    set_overrides: list[str],
    smoke: bool,
) -> dict[str, Any]:
    return {
        **cfg,
        "_runtime": {
            **cfg.get("_runtime", {}),
            "config_path": str(Path(base_config)),
            "config_overrides": [str(Path(path)) for path in config_overrides],
            "set_overrides": list(set_overrides),
            "smoke": bool(smoke),
        },
    }


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _profile_config(
    base_config: str,
    extra_overrides: list[str],
    set_overrides: list[str],
    profile_name: str,
    smoke: bool,
) -> dict[str, Any]:
    canonical_profile = canonicalize_profile_name(profile_name)
    if canonical_profile is None:
        raise ValueError("profile_name is required")
    config_overrides = [*extra_overrides, str(RULE_PROFILES[canonical_profile])]
    cfg = load_config(
        base_config,
        config_overrides,
        set_overrides,
    )
    return _with_runtime_context(
        cfg,
        base_config=base_config,
        config_overrides=config_overrides,
        set_overrides=set_overrides,
        smoke=smoke,
    )


def _screen_profile(
    base_config: str,
    extra_overrides: list[str],
    set_overrides: list[str],
    profile_name: str,
    smoke: bool,
) -> dict[str, Any]:
    canonical_profile = canonicalize_profile_name(profile_name)
    if canonical_profile is None:
        raise ValueError("profile_name is required")
    cfg = _profile_config(base_config, extra_overrides, set_overrides, canonical_profile, smoke)
    base_run_name = run_name_from_cfg(cfg)
    cfg["project"]["run_name"] = f"{base_run_name}_{canonical_profile}_screen"
    cfg["model_selection"]["candidate_models"] = [MODEL_LOGREG]
    cfg["optuna"]["enabled"] = False

    try:
        experiment = run_experiment_suite(cfg=cfg, smoke=smoke)
        backtest = run_backtest(cfg)
    except ValueError as exc:
        return {
            "profile": canonical_profile,
            "run_name": cfg["project"]["run_name"],
            "all_patterns_usable": False,
            "usable_patterns": 0,
            "patterns_covered": 0,
            "macro_selection_f2": 0.0,
            "macro_test_f2": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "mean_expectancy": 0.0,
            "total_pnl": 0.0,
            "total_trades": 0,
            "failed": True,
            "failure_reason": str(exc),
        }

    summary_df: pd.DataFrame = experiment["summary_df"]
    backtest_df: pd.DataFrame = backtest["summary_df"]
    min_support = int(cfg.get("evaluation", {}).get("minimum_test_positive_support", 5))
    usable_mask = summary_df["support_positive"] >= min_support if not summary_df.empty else pd.Series(dtype=bool)
    usable_patterns = int(summary_df.loc[usable_mask, "pattern"].nunique()) if not summary_df.empty else 0
    total_patterns = len(allowed_patterns_from_cfg(cfg))

    row = {
        "profile": canonical_profile,
        "run_name": cfg["project"]["run_name"],
        "all_patterns_usable": usable_patterns == total_patterns and total_patterns > 0,
        "usable_patterns": usable_patterns,
        "patterns_covered": int(summary_df["pattern"].nunique()) if not summary_df.empty else 0,
        "macro_selection_f2": float(summary_df["selection_f2"].mean()) if not summary_df.empty else 0.0,
        "macro_test_f2": float(summary_df["f2"].mean()) if not summary_df.empty else 0.0,
        "macro_precision": float(summary_df["precision"].mean()) if not summary_df.empty else 0.0,
        "macro_recall": float(summary_df["recall"].mean()) if not summary_df.empty else 0.0,
        "mean_expectancy": float(backtest_df["expectancy"].mean()) if not backtest_df.empty else 0.0,
        "total_pnl": float(backtest_df["total_pnl"].sum()) if not backtest_df.empty else 0.0,
        "total_trades": int(backtest_df["trades"].sum()) if not backtest_df.empty else 0,
        "failed": False,
        "failure_reason": None,
    }
    return row


def _rank_screen_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        by=[
            "all_patterns_usable",
            "macro_selection_f2",
            "mean_expectancy",
            "patterns_covered",
            "total_trades",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def _run_final(
    base_config: str,
    extra_overrides: list[str],
    set_overrides: list[str],
    profile_name: str,
    smoke: bool,
) -> dict[str, Any]:
    canonical_profile = canonicalize_profile_name(profile_name)
    if canonical_profile is None:
        raise ValueError("profile_name is required")
    cfg = _profile_config(base_config, extra_overrides, set_overrides, canonical_profile, smoke)
    base_run_name = run_name_from_cfg(cfg)
    cfg["project"]["run_name"] = final_run_dir_name(base_run_name, canonical_profile)
    experiment = run_experiment_suite(cfg=cfg, smoke=smoke)
    backtest = run_backtest(cfg)
    gallery = render_gallery(cfg)
    return {
        "experiment": experiment,
        "backtest": backtest,
        "gallery": gallery,
        "run_name": cfg["project"]["run_name"],
        "winner_profile": canonical_profile,
        "cfg": cfg,
    }


def _canonical_stage(stage: str) -> str:
    return "final" if stage == "overnight" else stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the screened multi-profile SPY pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--set", dest="set_overrides", action="append", default=[])
    parser.add_argument("--stage", choices=["screen", "full", "final", "overnight"], default="full")
    parser.add_argument("--profile", choices=list(supported_profile_names()), default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    stage = _canonical_stage(args.stage)

    seed_cfg = load_config(args.config, args.config_override, args.set_overrides)
    base_run_name = run_name_from_cfg(seed_cfg)
    report_root = ensure_dir(
        Path(seed_cfg["paths"].get("metrics_dir", "outputs/metrics")) / profile_scan_dir_name(base_run_name)
    )

    ranked_df = pd.DataFrame()
    winner_profile = canonicalize_profile_name(args.profile)

    if stage in {"screen", "full"} or winner_profile is None:
        rows = [
            _screen_profile(args.config, args.config_override, args.set_overrides, profile_name, smoke=args.smoke)
            for profile_name in RULE_PROFILES
        ]
        ranked_df = _rank_screen_results(rows)
        ranked_payload = ranked_df.to_dict(orient="records")
        ranked_df.to_csv(report_root / "profile_comparison.csv", index=False)
        ranked_df.to_csv(report_root / "rule_regime_comparison.csv", index=False)
        _save_json(report_root / "profile_comparison.json", ranked_payload)
        _save_json(report_root / "rule_regime_comparison.json", ranked_payload)
        if ranked_df.empty:
            raise RuntimeError("Rule screening produced no rows.")
        winner_profile = str(ranked_df.iloc[0]["profile"])
        print(f"Profile scan winner: {winner_profile}")

    if stage == "screen":
        return

    final_run = _run_final(
        args.config,
        args.config_override,
        args.set_overrides,
        winner_profile,
        smoke=args.smoke,
    )
    metrics_root = ensure_dir(Path(seed_cfg["paths"].get("metrics_dir", "outputs/metrics")) / final_run["run_name"])
    manifest = {
        "workflow": "run_recall_pipeline",
        "workflow_label": "Screened multi-profile research workflow",
        "winner_profile": winner_profile,
        "final_run_name": final_run["run_name"],
        "overnight_run_name": final_run["run_name"],
        "screen_report": str(report_root),
        "candidate_models": final_run["cfg"].get("model_selection", {}).get("candidate_models", []),
        "primary_selection_metric": final_run["cfg"].get("model_selection", {}).get("primary_selection_metric"),
        "config_stack": {
            "base_config": final_run["cfg"].get("_runtime", {}).get("config_path"),
            "config_overrides": list(final_run["cfg"].get("_runtime", {}).get("config_overrides", [])),
            "set_overrides": list(final_run["cfg"].get("_runtime", {}).get("set_overrides", [])),
        },
    }
    for filename in FINAL_RUN_MANIFEST_FILENAMES:
        _save_json(report_root / filename, manifest)
    _save_json(metrics_root / "pipeline_manifest.json", manifest)
    snapshot_dir = build_dashboard_snapshot(
        cfg=final_run["cfg"],
        run_name=final_run["run_name"],
        output_root=Path("web-dashboard") / ".generated",
        data_only=True,
    )
    print(f"Dashboard snapshot written to: {snapshot_dir}")
    print(f"Final stage finished with profile `{winner_profile}`.")


if __name__ == "__main__":
    main()
