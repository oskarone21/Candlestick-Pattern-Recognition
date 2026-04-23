# Chart Pattern Recognition

This project detects four classic chart patterns on 15-minute SPY data and shows the results in a web dashboard.

Patterns covered:

- Head and Shoulders
- Inverse Head and Shoulders
- Double Top
- Double Bottom

## What You Need

- Python with `pip`
- Node.js with `npm` for the dashboard

## First-Time Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install the dashboard dependencies:

```bash
cd web-dashboard
npm install
cd ..
```

## Tracked Grading Dataset

This repository ships the prepared 15-minute grading dataset at:

- `data/processed/spy_15m.csv`

That means the canonical `production_repro` workflow does not require Kaggle for grading or teammate verification. Kaggle is only needed if you want to regenerate the prepared dataset locally.

## Optional Kaggle Setup

The dataset is downloaded through `kagglehub`, so you need a Kaggle API token.

1. Create or download your Kaggle token from your Kaggle account settings.
2. Save it as `~/.kaggle/kaggle.json`.
3. Make sure the dataset settings in [configs/config.yaml](configs/config.yaml) are correct.

Default dataset settings:

- `data_source.kaggle_dataset`: `gratefuldata/intraday-stock-data-1-min-sp-500-200821`
- `data_source.kaggle_file_path`: `1_min_SPY_2008-2021.csv`

If Kaggle changes the CSV name, update `data_source.kaggle_file_path` before running the download step.

## Fastest End-to-End Run

Replace `my_run` with any run name you want to keep.

```bash
python scripts/run_experiment_suite.py --config configs/config.yaml --config-override configs/overrides/production_repro.yaml --set project.run_name=my_run
python scripts/run_backtest.py --config configs/config.yaml --config-override configs/overrides/production_repro.yaml --set project.run_name=my_run
python scripts/render_pattern_gallery.py --config configs/config.yaml --config-override configs/overrides/production_repro.yaml --set project.run_name=my_run
```

Then start the dashboard:

```bash
cd web-dashboard
npm run dev
```

Open `http://localhost:5173`.

## Reproducible Production Run

Use [configs/overrides/production_repro.yaml](configs/overrides/production_repro.yaml) when you want the verified CPU-first production profile rather than the broader research defaults in [configs/config.yaml](configs/config.yaml).

This profile bakes in the live setup that produced the strongest verified local run:

- `intraday_balanced` labeling thresholds
- CPU execution for reproducibility
- candidate models ordered as `tcn`, then `lstm`
- `f1` as the primary selection metric
- `window_minmax` sequence normalization
- training-only positive augmentation
- reduced Optuna budget tuned for the verified local run

The canonical grading-safe path starts from the tracked prepared dataset:

- `data/processed/spy_15m.csv`

Canonical training command:

```bash
python scripts/run_experiment_suite.py \
  --config configs/config.yaml \
  --config-override configs/overrides/production_repro.yaml \
  --set project.run_name=my_run
```

Main reproducibility artifact:

- `outputs/metrics/my_run/repro_manifest.json`

That manifest records:

- git commit SHA and branch
- base config plus override stack
- CLI `--set` overrides
- selected device and torch runtime summary
- exact installed dependency versions from `requirements.txt`
- raw and processed dataset paths, row counts, and SHA-256 hashes

With the same dataset, pinned dependencies, and CPU execution, the expected macro metrics for the verified reference run are approximately:

- F1: `0.8914`
- Precision: `0.9886`
- Recall: `0.8373`

A rerun within about `±0.02` on the macro metrics is a good reproducibility check.

## Step-by-Step

### 1. Use the tracked 15-minute dataset

The canonical first run uses the prepared dataset already included in the repository:

- `data/processed/spy_15m.csv`

This lets a professor or teammate rerun the canonical `production_repro` profile without Kaggle credentials or the raw-data preparation step.

### 2. Run the model training and model comparison

This trains the configured candidate models and selects a champion model for each pattern.

```bash
python scripts/run_experiment_suite.py \
  --config configs/config.yaml \
  --config-override configs/overrides/production_repro.yaml \
  --set project.run_name=my_run
```

Main output folder:

- `outputs/metrics/my_run/`

Important files in that folder:

- `model_comparison_summary.csv`
- `champions.csv`
- `macro_summary.json`
- `repro_manifest.json`

Candidate models in [configs/overrides/production_repro.yaml](configs/overrides/production_repro.yaml):

- `tcn`
- `lstm`

### 3. Run the backtest

This backtests the champion predictions produced in the previous step.

```bash
python scripts/run_backtest.py \
  --config configs/config.yaml \
  --config-override configs/overrides/production_repro.yaml \
  --set project.run_name=my_run
```

Main output folder:

- `outputs/backtest/my_run/`

Important files:

- `backtest_summary.csv`
- `backtest_summary.json`

### 4. Generate gallery images for the dashboard

This creates TP/FP/FN chart images for the champion models.

```bash
python scripts/render_pattern_gallery.py \
  --config configs/config.yaml \
  --config-override configs/overrides/production_repro.yaml \
  --set project.run_name=my_run
```

Main output folder:

- `outputs/gallery/my_run/`

Important file:

- `gallery_summary.json`

### 5. Run the dashboard

Start the dashboard in development mode:

```bash
cd web-dashboard
npm run dev
```

Open:

- `http://localhost:5173`

### 6. Optional: regenerate the prepared dataset from Kaggle

Only use this if you want to rebuild `data/processed/spy_15m.csv` from the raw Kaggle source.

Download the raw dataset:

```bash
python scripts/download_kaggle_intraday.py --config configs/config.yaml
```

Main output:

- `data/raw/spy_1m.csv`

Also writes a raw data quality report to:

- `outputs/data_quality/raw_1m_quality_report.json`

Prepare the canonical 15-minute dataset:

```bash
python scripts/prepare_15m_dataset.py --config configs/config.yaml
```

Main outputs:

- `data/processed/spy_1m_cleaned.csv`
- `data/processed/spy_15m.csv`
- `outputs/data_quality/timezone_report.json`
- `outputs/data_quality/session_coverage_report.csv`
- `outputs/data_quality/prep_quality_report.json`

If you want this step to re-download the raw Kaggle data first, use:

```bash
python scripts/prepare_15m_dataset.py --config configs/config.yaml --refresh-raw
```

## How To Refresh The Live Dashboard Results

The dashboard updates from completed runs automatically. You do not normally need to copy files by hand.

For a run to be visible to the dashboard, all of these must exist for the same run name:

- `outputs/metrics/<run_name>/model_comparison_summary.csv`
- `outputs/metrics/<run_name>/champions.csv`
- `outputs/backtest/<run_name>/backtest_summary.csv`
- `outputs/gallery/<run_name>/gallery_summary.json`

In practice, that means you must run these three commands with the same `project.run_name`:

```bash
python scripts/run_experiment_suite.py --config configs/config.yaml --config-override configs/overrides/production_repro.yaml --set project.run_name=my_run
python scripts/run_backtest.py --config configs/config.yaml --config-override configs/overrides/production_repro.yaml --set project.run_name=my_run
python scripts/render_pattern_gallery.py --config configs/config.yaml --config-override configs/overrides/production_repro.yaml --set project.run_name=my_run
```

After that, refresh the browser page. The dashboard server rebuilds its snapshot automatically when the source files are newer.

Important note: the dashboard does not always show the newest run. It prefers the strongest eligible finished run, then shows the supported subset of patterns from that run. If your latest run does not appear, compare it with older runs in `outputs/metrics/`, `outputs/backtest/`, and `outputs/gallery/`.

## Manual Dashboard Snapshot Rebuild

Only use this if you want to force a dashboard data refresh manually.

```bash
python scripts/build_results_dashboard.py \
  --config configs/config.yaml \
  --run-name my_run \
  --output-root web-dashboard/.generated \
  --data-only
```

This writes the dashboard snapshot to:

- `web-dashboard/.generated/my_run/data.json`

This only rebuilds the cached dashboard data for that run. It does not force the live dashboard to display that run if another finished run ranks higher.

## Most Important Settings

The main file to edit is [configs/config.yaml](configs/config.yaml).

Useful settings:

- `project.run_name`: name of the output run
- `model_selection.candidate_models`: which models to train
- `labeling.allowed_patterns`: which patterns to detect
- `optuna.enabled`: whether Optuna tuning is enabled
- `dashboard.presentation.*`: support floor for the dashboard's supported subset
- `paths.*`: where data and outputs are written

## Quick Checks

Run the test suite:

```bash
pytest -q
```

Run a fast synthetic smoke test without the Kaggle dataset:

```bash
python scripts/run_experiment_suite.py --config configs/config.yaml --smoke
```

## Useful Files

- [configs/config.yaml](configs/config.yaml): main project settings
- [docs/PATTERNS_EXPLAINED.md](docs/PATTERNS_EXPLAINED.md): pattern definitions and assumptions
- [docs/BUSINESS_IMPACT.md](docs/BUSINESS_IMPACT.md): business context
- [docs/BRIEF.md](docs/BRIEF.md): consultancy brief
- [docs/EXPLAINABILITY.md](docs/EXPLAINABILITY.md): SHAP artifacts and workflow
