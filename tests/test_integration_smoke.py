from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def test_integration_smoke_run(tmp_path, base_cfg):
    cfg = base_cfg
    cfg["project"]["run_name"] = "smoke_ci"
    cfg["model_selection"]["candidate_models"] = ["logreg"]
    cfg["optuna"]["enabled"] = False
    cfg["evaluation"]["bootstrap_iterations"] = 30
    cfg["evaluation"]["minimum_test_positive_support"] = 1

    cfg["paths"]["metrics_dir"] = str(tmp_path / "metrics")
    cfg["paths"]["checkpoints_dir"] = str(tmp_path / "checkpoints")
    cfg["paths"]["datasets_dir"] = str(tmp_path / "datasets")

    cfg_path = tmp_path / "smoke_config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "scripts/run_experiment_suite.py",
        "--config",
        str(cfg_path),
        "--smoke",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    run_root = tmp_path / "metrics" / "smoke_ci"
    assert (run_root / "model_comparison_summary.csv").exists()
    assert (run_root / "champions.csv").exists()
    assert (run_root / "macro_summary.json").exists()
    assert (run_root / "runtime_summary.json").exists()
