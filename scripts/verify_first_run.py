from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from scripts._bootstrap import ensure_repo_root
except ImportError:
    from _bootstrap import ensure_repo_root

ensure_repo_root()

from chart_patterns.run_naming import (
    FINAL_RUN_MANIFEST_FILENAMES,
    accepted_reference_run_aliases,
    candidate_final_run_globs,
    candidate_first_run_reference_paths,
    candidate_profile_scan_dir_names,
    canonicalize_winner_profile,
    resolve_final_run_name_field,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _requested_run_aliases(requested_run_name: str) -> tuple[str, ...]:
    aliases = [requested_run_name]
    if requested_run_name in accepted_reference_run_aliases():
        aliases.extend(accepted_reference_run_aliases())
    return tuple(dict.fromkeys(aliases))


def _resolve_completed_run_name(repo_root: Path, requested_run_name: str) -> str:
    metrics_root = repo_root / "outputs" / "metrics"

    for candidate_name in _requested_run_aliases(requested_run_name):
        direct_pipeline = metrics_root / candidate_name / "pipeline_manifest.json"
        if direct_pipeline.exists():
            return candidate_name

    for candidate_name in _requested_run_aliases(requested_run_name):
        for scan_dir_name in candidate_profile_scan_dir_names(candidate_name):
            scan_dir = metrics_root / scan_dir_name
            for manifest_name in FINAL_RUN_MANIFEST_FILENAMES:
                manifest_path = scan_dir / manifest_name
                if not manifest_path.exists():
                    continue
                final_run_name = resolve_final_run_name_field(_load_json(manifest_path))
                if final_run_name:
                    return final_run_name

    matches: list[Path] = []
    for candidate_name in _requested_run_aliases(requested_run_name):
        for pattern in candidate_final_run_globs(candidate_name):
            matches.extend(metrics_root.glob(pattern))
    if matches:
        ranked = sorted(
            {match.parent for match in matches},
            key=lambda path: (not path.name.endswith("_final"), path.name),
        )
        if len(ranked) == 1:
            return ranked[0].name

    raise FileNotFoundError(
        "Could not resolve a completed reproducibility run for "
        f"`{requested_run_name}` under {metrics_root}."
    )


def _round_metric(value: Any, digits: int) -> float:
    return round(float(value), digits)


def _collect_actual_results(repo_root: Path, requested_run_name: str) -> dict[str, Any]:
    resolved_run_name = _resolve_completed_run_name(repo_root, requested_run_name)
    metrics_root = repo_root / "outputs" / "metrics" / resolved_run_name
    backtest_root = repo_root / "outputs" / "backtest" / resolved_run_name
    snapshot_path = repo_root / "web-dashboard" / ".generated" / resolved_run_name / "data.json"

    required_paths = [
        metrics_root / "repro_manifest.json",
        metrics_root / "pipeline_manifest.json",
        metrics_root / "champions.csv",
        metrics_root / "macro_summary.json",
        backtest_root / "backtest_summary.csv",
        snapshot_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing reproducibility artifacts:\n- " + "\n- ".join(missing))

    repro_manifest = _load_json(metrics_root / "repro_manifest.json")
    pipeline_manifest = _load_json(metrics_root / "pipeline_manifest.json")
    macro_summary = _load_json(metrics_root / "macro_summary.json")
    champions = _read_csv_rows(metrics_root / "champions.csv")
    backtest_rows = _read_csv_rows(backtest_root / "backtest_summary.csv")
    snapshot = _load_json(snapshot_path)

    champion_models = {row["pattern"]: row["champion_model"] for row in champions}
    champion_snapshot_rows = {
        row["pattern"]: row
        for row in snapshot.get("classification", {}).get("champions", [])
    }

    return {
        "requested_run_name": requested_run_name,
        "resolved_run_name": resolved_run_name,
        "processed_15m_sha256": repro_manifest.get("datasets", {}).get("processed_15m", {}).get("sha256"),
        "selected_device": repro_manifest.get("runtime", {}).get("selected_device"),
        "workflow": pipeline_manifest.get("workflow"),
        "winner_profile": pipeline_manifest.get("winner_profile"),
        "visible_patterns": snapshot.get("presentation", {}).get("visible_patterns", []),
        "champion_models": champion_models,
        "macro_metrics": {
            key: _round_metric(macro_summary[key], 4)
            for key in (
                "macro_f1",
                "macro_f2",
                "macro_precision",
                "macro_recall",
                "macro_selection_metric",
            )
            if key in macro_summary
        },
        "total_trades": sum(int(float(row.get("trades", 0) or 0)) for row in backtest_rows),
        "total_pnl": _round_metric(
            sum(float(row.get("total_pnl", 0.0) or 0.0) for row in backtest_rows),
            2,
        ),
        "pattern_support": {
            pattern: {
                "val_positive_support": int(float(row.get("val_positive_support", 0) or 0)),
                "test_positive_support": int(float(row.get("test_positive_support", 0) or 0)),
            }
            for pattern, row in champion_snapshot_rows.items()
        },
    }


def compare_to_reference(actual: dict[str, Any], reference: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []

    if actual.get("processed_15m_sha256") != reference.get("processed_15m_sha256"):
        mismatches.append(
            "processed_15m sha256 mismatch: "
            f"expected {reference.get('processed_15m_sha256')}, got {actual.get('processed_15m_sha256')}"
        )
    if actual.get("selected_device") != "cpu":
        mismatches.append(f"selected_device mismatch: expected cpu, got {actual.get('selected_device')}")
    if actual.get("workflow") != reference.get("workflow"):
        mismatches.append(
            f"workflow mismatch: expected {reference.get('workflow')}, got {actual.get('workflow')}"
        )
    if canonicalize_winner_profile(actual.get("winner_profile")) != canonicalize_winner_profile(
        reference.get("winner_profile")
    ):
        mismatches.append(
            "winner_profile mismatch: "
            f"expected {reference.get('winner_profile')}, got {actual.get('winner_profile')}"
        )
    if actual.get("visible_patterns") != reference.get("visible_patterns"):
        mismatches.append(
            "visible_patterns mismatch: "
            f"expected {reference.get('visible_patterns')}, got {actual.get('visible_patterns')}"
        )
    if actual.get("champion_models") != reference.get("champion_models"):
        mismatches.append(
            "champion_models mismatch: "
            f"expected {reference.get('champion_models')}, got {actual.get('champion_models')}"
        )
    if actual.get("macro_metrics") != reference.get("macro_metrics"):
        mismatches.append(
            "macro_metrics mismatch: "
            f"expected {reference.get('macro_metrics')}, got {actual.get('macro_metrics')}"
        )
    if actual.get("total_trades") != reference.get("total_trades"):
        mismatches.append(
            f"total_trades mismatch: expected {reference.get('total_trades')}, got {actual.get('total_trades')}"
        )
    if actual.get("total_pnl") != reference.get("total_pnl"):
        mismatches.append(
            f"total_pnl mismatch: expected {reference.get('total_pnl')}, got {actual.get('total_pnl')}"
        )
    if actual.get("pattern_support") != reference.get("pattern_support"):
        mismatches.append(
            "pattern_support mismatch: "
            f"expected {reference.get('pattern_support')}, got {actual.get('pattern_support')}"
        )

    return mismatches


def _resolve_reference_path(repo_root: Path, requested_reference: str | None) -> Path:
    if requested_reference:
        requested_path = (repo_root / requested_reference).resolve()
        if requested_path.exists():
            return requested_path
    for path in candidate_first_run_reference_paths(repo_root):
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate a committed first-run reference JSON.")


def verify_run(repo_root: Path, requested_run_name: str, reference_path: Path) -> tuple[str, list[str]]:
    actual = _collect_actual_results(repo_root, requested_run_name)
    reference = _load_json(reference_path)
    mismatches = compare_to_reference(actual, reference)
    return actual["resolved_run_name"], mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that a finished run matches the canonical first-run reference.")
    parser.add_argument("--run-name", required=True, help="Base run name or finished final run name.")
    parser.add_argument(
        "--reference",
        default=str(candidate_first_run_reference_paths()[0]),
        help="Path to the committed reproducibility reference JSON.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    reference_path = _resolve_reference_path(repo_root, args.reference)
    resolved_run_name, mismatches = verify_run(repo_root, args.run_name, reference_path)

    if mismatches:
        print(f"Reproducibility check failed for `{resolved_run_name}`:")
        for item in mismatches:
            print(f"- {item}")
        raise SystemExit(1)

    print(
        "Reproducibility check passed for "
        f"`{resolved_run_name}` against `{reference_path.relative_to(repo_root)}`."
    )


if __name__ == "__main__":
    main()
