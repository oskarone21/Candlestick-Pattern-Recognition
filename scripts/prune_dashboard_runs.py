from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

try:
    from scripts._bootstrap import REPO_ROOT, ensure_repo_root
except ImportError:
    from _bootstrap import REPO_ROOT, ensure_repo_root

ensure_repo_root()

PROTECTED_RUNS = {
    "dl_rerun_aug_windowminmax_stageAdata_20260419",
    "dl_rerun_baseline_stageAdata_20260419",
    "live_balanced_fix_aug_minmax_optuna_20260419",
    "rough_screen_all_models_2026_04_18_relaxed",
    "stageA_compare_all_no_optuna",
    "stageB_optuna_coarse",
}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _all_close_to_one(values: list[float]) -> bool:
    return bool(values) and all(value >= 0.999 for value in values)


def detect_prune_reasons(metrics_run_dir: Path, macro_f1_floor: float = 0.40) -> list[str]:
    reasons: list[str] = []
    macro = _read_json(metrics_run_dir / "macro_summary.json")
    macro_f1 = macro.get("macro_f1")
    macro_selection_metric = macro.get("macro_selection_metric")
    champions = _read_csv_rows(metrics_run_dir / "champions.csv")

    if isinstance(macro_f1, (int, float)) and float(macro_f1) < macro_f1_floor:
        reasons.append("macro_f1_below_floor")

    if isinstance(macro_selection_metric, (int, float)) and float(macro_selection_metric) >= 0.999:
        reasons.append("perfect_selection_lineage")
    elif champions:
        selection_values: list[float] = []
        for row in champions:
            value = row.get("selection_metric_value")
            if value in (None, ""):
                selection_values = []
                break
            try:
                selection_values.append(float(value))
            except ValueError:
                selection_values = []
                break
        if _all_close_to_one(selection_values):
            reasons.append("perfect_selection_lineage")

    return sorted(set(reasons))


def collect_prune_candidates(metrics_dir: Path, macro_f1_floor: float = 0.40) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {}
    if not metrics_dir.exists():
        return candidates

    for run_dir in sorted(path for path in metrics_dir.iterdir() if path.is_dir()):
        if run_dir.name in PROTECTED_RUNS:
            continue
        reasons = detect_prune_reasons(run_dir, macro_f1_floor=macro_f1_floor)
        if reasons:
            candidates[run_dir.name] = reasons
    return candidates


def run_target_paths(outputs_root: Path, generated_root: Path, run_name: str) -> list[Path]:
    targets: list[Path] = []

    if outputs_root.exists():
        for bucket in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
            candidate = bucket / run_name
            if candidate.exists():
                targets.append(candidate)

    generated_candidate = generated_root / run_name
    if generated_candidate.exists():
        targets.append(generated_candidate)

    return targets


def prune_runs(
    run_names: list[str],
    outputs_root: Path,
    generated_root: Path,
    *,
    delete: bool,
) -> list[Path]:
    removed: list[Path] = []
    for run_name in run_names:
        for target in run_target_paths(outputs_root, generated_root, run_name):
            removed.append(target)
            if delete:
                shutil.rmtree(target)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune poor or unrealistic dashboard run artifacts.")
    parser.add_argument("--outputs-root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--generated-root", default=str(REPO_ROOT / "web-dashboard" / ".generated"))
    parser.add_argument("--macro-f1-floor", type=float, default=0.40)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the selected run folders. Without this flag the script only reports what would be removed.",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    generated_root = Path(args.generated_root)
    metrics_dir = outputs_root / "metrics"
    candidates = collect_prune_candidates(metrics_dir, macro_f1_floor=args.macro_f1_floor)

    if not candidates:
        print("No prune candidates found.")
        return

    print("Prune candidates:")
    for run_name, reasons in candidates.items():
        print(f"- {run_name}: {', '.join(reasons)}")
        for target in run_target_paths(outputs_root, generated_root, run_name):
            print(f"  - {target}")

    removed = prune_runs(sorted(candidates), outputs_root, generated_root, delete=args.delete)
    if args.delete:
        print(f"Deleted {len(removed)} directories across outputs and generated snapshots.")
    else:
        print(f"Dry run only. {len(removed)} directories would be deleted.")


if __name__ == "__main__":
    main()
