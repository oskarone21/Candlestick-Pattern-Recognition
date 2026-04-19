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
from candlestick.domain import MODEL_LOGREG
from candlestick.project_utils import allowed_patterns_from_cfg, run_name_from_cfg
from scripts.run_backtest import run_backtest
from scripts.run_experiment_suite import run_experiment_suite


RULE_PROFILES = {
    "intraday_conservative": Path("configs/overrides/intraday_conservative.yaml"),
    "intraday_balanced": Path("configs/overrides/intraday_balanced.yaml"),
    "intraday_recall": Path("configs/overrides/intraday_recall.yaml"),
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
) -> dict[str, Any]:
    cfg = load_config(
        base_config,
        [*extra_overrides, str(RULE_PROFILES[profile_name])],
        set_overrides,
    )
    return cfg


def _screen_profile(
    base_config: str,
    extra_overrides: list[str],
    set_overrides: list[str],
    profile_name: str,
    smoke: bool,
) -> dict[str, Any]:
    cfg = _profile_config(base_config, extra_overrides, set_overrides, profile_name)
    base_run_name = run_name_from_cfg(cfg)
    cfg["project"]["run_name"] = f"{base_run_name}_{profile_name}_screen"
    cfg["model_selection"]["candidate_models"] = [MODEL_LOGREG]
    cfg["optuna"]["enabled"] = False

    experiment = run_experiment_suite(cfg=cfg, smoke=smoke)
    backtest = run_backtest(cfg)

    summary_df: pd.DataFrame = experiment["summary_df"]
    backtest_df: pd.DataFrame = backtest["summary_df"]
    min_support = int(cfg.get("evaluation", {}).get("minimum_test_positive_support", 5))
    usable_mask = summary_df["support_positive"] >= min_support if not summary_df.empty else pd.Series(dtype=bool)
    usable_patterns = int(summary_df.loc[usable_mask, "pattern"].nunique()) if not summary_df.empty else 0
    total_patterns = len(allowed_patterns_from_cfg(cfg))

    row = {
        "profile": profile_name,
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


def _run_overnight(
    base_config: str,
    extra_overrides: list[str],
    set_overrides: list[str],
    profile_name: str,
    smoke: bool,
) -> dict[str, Any]:
    cfg = _profile_config(base_config, extra_overrides, set_overrides, profile_name)
    base_run_name = run_name_from_cfg(cfg)
    cfg["project"]["run_name"] = f"{base_run_name}_{profile_name}_overnight"
    return {
        "experiment": run_experiment_suite(cfg=cfg, smoke=smoke),
        "backtest": run_backtest(cfg),
        "run_name": cfg["project"]["run_name"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the two-stage SPY recall-weighted pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--set", dest="set_overrides", action="append", default=[])
    parser.add_argument("--stage", choices=["screen", "full", "overnight"], default="full")
    parser.add_argument("--profile", choices=sorted(RULE_PROFILES), default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    seed_cfg = load_config(args.config, args.config_override, args.set_overrides)
    base_run_name = run_name_from_cfg(seed_cfg)
    report_root = ensure_dir(Path(seed_cfg["paths"].get("metrics_dir", "outputs/metrics")) / f"{base_run_name}_rule_screen")

    ranked_df = pd.DataFrame()
    winner_profile = args.profile

    if args.stage in {"screen", "full"} or winner_profile is None:
        rows = [
            _screen_profile(args.config, args.config_override, args.set_overrides, profile_name, smoke=args.smoke)
            for profile_name in RULE_PROFILES
        ]
        ranked_df = _rank_screen_results(rows)
        ranked_df.to_csv(report_root / "rule_regime_comparison.csv", index=False)
        _save_json(report_root / "rule_regime_comparison.json", ranked_df.to_dict(orient="records"))
        if ranked_df.empty:
            raise RuntimeError("Rule screening produced no rows.")
        winner_profile = str(ranked_df.iloc[0]["profile"])
        print(f"Rule screening winner: {winner_profile}")

    if args.stage == "screen":
        return

    overnight = _run_overnight(
        args.config,
        args.config_override,
        args.set_overrides,
        winner_profile,
        smoke=args.smoke,
    )
    manifest = {
        "winner_profile": winner_profile,
        "overnight_run_name": overnight["run_name"],
        "screen_report": str(report_root),
    }
    _save_json(report_root / "overnight_manifest.json", manifest)
    print(f"Overnight stage finished with profile `{winner_profile}`.")


if __name__ == "__main__":
    main()
