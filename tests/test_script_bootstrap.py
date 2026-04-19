from __future__ import annotations

import importlib


def test_script_modules_import_with_shared_bootstrap():
    experiment = importlib.import_module("scripts.run_experiment_suite")
    dashboard = importlib.import_module("scripts.build_results_dashboard")

    assert callable(experiment.run_experiment_suite)
    assert callable(dashboard.main)
