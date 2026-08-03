from __future__ import annotations

import csv
import json

from scripts.prune_dashboard_runs import collect_prune_candidates, run_target_paths


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_collect_prune_candidates_flags_perfect_and_low_macro_runs(tmp_path):
    metrics_dir = tmp_path / "outputs" / "metrics"

    _write_json(
        metrics_dir / "production_repro_handoff_clean_20260419" / "macro_summary.json",
        {"macro_f1": 0.89, "macro_selection_metric": 1.0},
    )
    _write_csv(
        metrics_dir / "production_repro_handoff_clean_20260419" / "champions.csv",
        [{"pattern": "head_shoulders", "selection_metric_value": "1.0"}],
    )

    _write_json(
        metrics_dir / "live_recall_1h_20260418_230820" / "macro_summary.json",
        {"macro_f1": 0.31},
    )
    _write_csv(
        metrics_dir / "live_recall_1h_20260418_230820" / "champions.csv",
        [{"pattern": "head_shoulders", "selection_metric_value": "0.42"}],
    )

    _write_json(
        metrics_dir / "stageB_optuna_coarse" / "macro_summary.json",
        {"macro_f1": 0.77, "macro_selection_metric": 0.83},
    )
    _write_csv(
        metrics_dir / "stageB_optuna_coarse" / "champions.csv",
        [{"pattern": "head_shoulders", "selection_metric_value": "0.74"}],
    )

    candidates = collect_prune_candidates(metrics_dir)

    assert candidates == {
        "live_recall_1h_20260418_230820": ["macro_f1_below_floor"],
        "production_repro_handoff_clean_20260419": ["perfect_selection_lineage"],
    }


def test_run_target_paths_cover_outputs_and_generated_snapshot(tmp_path):
    outputs_root = tmp_path / "outputs"
    generated_root = tmp_path / "web-dashboard" / ".generated"
    run_name = "production_repro_handoff_clean_20260419"

    for root in ("metrics", "backtest", "gallery", "dashboard"):
        (outputs_root / root / run_name).mkdir(parents=True)
    (generated_root / run_name).mkdir(parents=True)

    targets = run_target_paths(outputs_root, generated_root, run_name)

    assert targets == [
        outputs_root / "backtest" / run_name,
        outputs_root / "dashboard" / run_name,
        outputs_root / "gallery" / run_name,
        outputs_root / "metrics" / run_name,
        generated_root / run_name,
    ]
