from __future__ import annotations

import csv
import json
from pathlib import Path

from chart_patterns.run_naming import canonicalize_profile_name
from scripts.run_recall_pipeline import _with_runtime_context
from scripts.verify_first_run import verify_run


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _seed_reference(repo_root: Path) -> Path:
    reference_path = repo_root / "docs" / "reference" / "first_run_reference.json"
    _write_json(
        reference_path,
        {
            "reference_run_name": "broad_signal_validation_intraday_broad_signal_overnight",
            "accepted_run_aliases": [
                "reference_run_high_coverage_final",
                "broad_signal_validation_intraday_broad_signal_overnight",
            ],
            "processed_15m_sha256": "dataset-hash",
            "workflow": "run_recall_pipeline",
            "winner_profile": "high_coverage",
            "visible_patterns": [
                "double_bottom",
                "double_top",
                "head_shoulders",
                "inverse_head_shoulders",
            ],
            "champion_models": {
                "head_shoulders": "tcn",
                "inverse_head_shoulders": "tcn",
                "double_top": "logreg",
                "double_bottom": "logreg",
            },
            "macro_metrics": {
                "macro_f1": 0.5494,
                "macro_f2": 0.6281,
                "macro_precision": 0.5203,
                "macro_recall": 0.7148,
                "macro_selection_metric": 0.7531,
            },
            "total_trades": 282,
            "total_pnl": 92786.58,
            "pattern_support": {
                "head_shoulders": {"val_positive_support": 104, "test_positive_support": 92},
                "inverse_head_shoulders": {"val_positive_support": 84, "test_positive_support": 82},
                "double_top": {"val_positive_support": 19, "test_positive_support": 30},
                "double_bottom": {"val_positive_support": 12, "test_positive_support": 31},
            },
        },
    )
    return reference_path


def _seed_run_artifacts(
    repo_root: Path,
    *,
    base_run_name: str,
    final_run_name: str,
    manifest_dir_name: str,
    manifest_filename: str,
    manifest_field: str,
    winner_profile: str,
    total_trades: int = 282,
) -> None:
    _write_json(
        repo_root / "outputs" / "metrics" / manifest_dir_name / manifest_filename,
        {manifest_field: final_run_name},
    )
    _write_json(
        repo_root / "outputs" / "metrics" / final_run_name / "repro_manifest.json",
        {
            "datasets": {"processed_15m": {"sha256": "dataset-hash"}},
            "runtime": {"selected_device": "cpu"},
        },
    )
    _write_json(
        repo_root / "outputs" / "metrics" / final_run_name / "pipeline_manifest.json",
        {
            "workflow": "run_recall_pipeline",
            "winner_profile": winner_profile,
            "final_run_name": final_run_name,
            "screen_report": f"outputs/metrics/{manifest_dir_name}",
        },
    )
    _write_json(
        repo_root / "outputs" / "metrics" / final_run_name / "macro_summary.json",
        {
            "macro_f1": 0.5494,
            "macro_f2": 0.6281,
            "macro_precision": 0.5203,
            "macro_recall": 0.7148,
            "macro_selection_metric": 0.7531,
        },
    )
    _write_csv(
        repo_root / "outputs" / "metrics" / final_run_name / "champions.csv",
        [
            {"pattern": "head_shoulders", "champion_model": "tcn"},
            {"pattern": "inverse_head_shoulders", "champion_model": "tcn"},
            {"pattern": "double_top", "champion_model": "logreg"},
            {"pattern": "double_bottom", "champion_model": "logreg"},
        ],
    )
    _write_csv(
        repo_root / "outputs" / "backtest" / final_run_name / "backtest_summary.csv",
        [
            {"pattern": "head_shoulders", "trades": 151, "total_pnl": 51110.60},
            {"pattern": "inverse_head_shoulders", "trades": 33, "total_pnl": 10173.20},
            {"pattern": "double_top", "trades": 48, "total_pnl": 21150.39},
            {"pattern": "double_bottom", "trades": total_trades - 232, "total_pnl": 10352.39},
        ],
    )
    _write_json(
        repo_root / "web-dashboard" / ".generated" / final_run_name / "data.json",
        {
            "presentation": {
                "visible_patterns": [
                    "double_bottom",
                    "double_top",
                    "head_shoulders",
                    "inverse_head_shoulders",
                ]
            },
            "classification": {
                "champions": [
                    {
                        "pattern": "head_shoulders",
                        "model": "tcn",
                        "val_positive_support": 104,
                        "test_positive_support": 92,
                    },
                    {
                        "pattern": "inverse_head_shoulders",
                        "model": "tcn",
                        "val_positive_support": 84,
                        "test_positive_support": 82,
                    },
                    {
                        "pattern": "double_top",
                        "model": "logreg",
                        "val_positive_support": 19,
                        "test_positive_support": 30,
                    },
                    {
                        "pattern": "double_bottom",
                        "model": "logreg",
                        "val_positive_support": 12,
                        "test_positive_support": 31,
                    },
                ]
            },
        },
    )


def test_verify_run_resolves_new_profile_scan_manifest_and_passes(tmp_path):
    repo_root = tmp_path / "repo"
    reference_path = _seed_reference(repo_root)
    _seed_run_artifacts(
        repo_root,
        base_run_name="first_run_check",
        final_run_name="first_run_check_high_coverage_final",
        manifest_dir_name="first_run_check_profile_scan",
        manifest_filename="final_run_manifest.json",
        manifest_field="final_run_name",
        winner_profile="high_coverage",
    )

    resolved_run_name, mismatches = verify_run(repo_root, "first_run_check", reference_path)

    assert resolved_run_name == "first_run_check_high_coverage_final"
    assert mismatches == []


def test_verify_run_resolves_legacy_manifest_and_profile_alias(tmp_path):
    repo_root = tmp_path / "repo"
    reference_path = _seed_reference(repo_root)
    _seed_run_artifacts(
        repo_root,
        base_run_name="team_first_run",
        final_run_name="team_first_run_intraday_broad_signal_overnight",
        manifest_dir_name="team_first_run_rule_screen",
        manifest_filename="overnight_manifest.json",
        manifest_field="overnight_run_name",
        winner_profile="intraday_broad_signal",
    )

    resolved_run_name, mismatches = verify_run(repo_root, "team_first_run", reference_path)

    assert resolved_run_name == "team_first_run_intraday_broad_signal_overnight"
    assert mismatches == []


def test_verify_run_accepts_reference_alias_when_only_legacy_run_exists(tmp_path):
    repo_root = tmp_path / "repo"
    reference_path = _seed_reference(repo_root)
    _seed_run_artifacts(
        repo_root,
        base_run_name="broad_signal_validation",
        final_run_name="broad_signal_validation_intraday_broad_signal_overnight",
        manifest_dir_name="broad_signal_validation_rule_screen",
        manifest_filename="overnight_manifest.json",
        manifest_field="overnight_run_name",
        winner_profile="intraday_broad_signal",
    )

    resolved_run_name, mismatches = verify_run(repo_root, "reference_run_high_coverage_final", reference_path)

    assert resolved_run_name == "broad_signal_validation_intraday_broad_signal_overnight"
    assert mismatches == []


def test_verify_run_reports_reference_mismatch(tmp_path):
    repo_root = tmp_path / "repo"
    reference_path = _seed_reference(repo_root)
    _seed_run_artifacts(
        repo_root,
        base_run_name="first_run_check",
        final_run_name="first_run_check_high_coverage_final",
        manifest_dir_name="first_run_check_profile_scan",
        manifest_filename="final_run_manifest.json",
        manifest_field="final_run_name",
        winner_profile="high_coverage",
        total_trades=280,
    )

    resolved_run_name, mismatches = verify_run(repo_root, "first_run_check", reference_path)

    assert resolved_run_name == "first_run_check_high_coverage_final"
    assert any("total_trades mismatch" in item for item in mismatches)


def test_run_recall_pipeline_runtime_context_records_config_stack():
    cfg = {"project": {"run_name": "first_run_check"}}

    enriched = _with_runtime_context(
        cfg,
        base_config="configs/config.yaml",
        config_overrides=["configs/overrides/first_run_repro.yaml"],
        set_overrides=["project.run_name=first_run_check"],
        smoke=False,
    )

    assert enriched["_runtime"]["config_path"] == "configs/config.yaml"
    assert enriched["_runtime"]["config_overrides"] == ["configs/overrides/first_run_repro.yaml"]
    assert enriched["_runtime"]["set_overrides"] == ["project.run_name=first_run_check"]
    assert enriched["_runtime"]["smoke"] is False


def test_profile_aliases_canonicalize_to_new_names():
    assert canonicalize_profile_name("intraday_broad_signal") == "high_coverage"
    assert canonicalize_profile_name("intraday_conservative") == "conservative"
    assert canonicalize_profile_name("intraday_balanced") == "balanced"
    assert canonicalize_profile_name("intraday_recall") == "high_recall"
