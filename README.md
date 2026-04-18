# Automated Candlestick Pattern Recognition (SPY 15-Min Prototype)

This repository now implements an end-to-end, config-driven prototype for detecting four classic chart patterns on **15-minute SPY bars** derived from Kaggle 1-minute intraday data:

- Head and Shoulders
- Inverse Head and Shoulders
- Double Top
- Double Bottom

The pipeline includes:

- Kaggle ingestion with schema normalization
- timezone audit and conversion (`America/Denver -> America/New_York`)
- OHLCV cleaning + 1m -> 15m resampling
- breakout-anchored labeling with hard negatives
- one-vs-rest model comparison across 5 model families
- optional Optuna hyperparameter tuning
- post-breakout trade simulation (PnL, Sharpe, win rate, profit factor, drawdown)
- TP/FP/FN chart gallery export for visual verification

## Why this project matters

This prototype is designed as a practical decision-support system for analysts. It shows how objective pattern recognition can reduce manual chart-scanning effort and scale toward broader market coverage while maintaining auditable, reproducible rules.

See business framing in [docs/BUSINESS_IMPACT.md](/Users/oskarrodziewicz/Library/CloudStorage/OneDrive-UniversityofWarwick/Deep%20Learning/Candlestick-Pattern-Recognition/docs/BUSINESS_IMPACT.md).

## Repository structure

- [configs/config.yaml](/Users/oskarrodziewicz/Library/CloudStorage/OneDrive-UniversityofWarwick/Deep%20Learning/Candlestick-Pattern-Recognition/configs/config.yaml): single source of truth for data, labeling, models, tuning, and backtesting
- `candlestick/`: core library modules
- `scripts/`: runnable pipeline entrypoints
- `tests/`: unit + integration smoke tests
- [PATTERNS_EXPLAINED.md](/Users/oskarrodziewicz/Library/CloudStorage/OneDrive-UniversityofWarwick/Deep%20Learning/Candlestick-Pattern-Recognition/PATTERNS_EXPLAINED.md): mathematical definitions and implementation contract

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## End-to-end runbook

### 1) Download raw SPY intraday data from Kaggle

```bash
python scripts/download_kaggle_intraday.py --config configs/config.yaml
```

Notes:
- Uses `kagglehub.load_dataset(...)` and the dataset handle configured in `data_source.kaggle_dataset`.
- Set `data_source.kaggle_file_path` in config to match the exact CSV inside Kaggle.

### 2) Clean, audit timezone, and resample to 15-minute bars

```bash
python scripts/prepare_15m_dataset.py --config configs/config.yaml
```

Outputs:
- `paths.cleaned_1m_path` (default `data/processed/spy_1m_cleaned.csv`)
- `paths.processed_15m_path` (default `data/processed/spy_15m.csv`)
- raw data quality report (`paths.raw_1m_quality_report`)
- timezone audit report (`paths.timezone_report`)
- session coverage report (`paths.session_coverage_report`)
- data quality report (`paths.data_quality_report`)

### 3) Train and compare models (one-vs-rest per pattern)

```bash
python scripts/run_experiment_suite.py --config configs/config.yaml
```

Default candidate models:
- `logreg`
- `hgb`
- `lstm`
- `tcn`
- `transformer`

Optuna settings are controlled under `optuna:` in config.

Outputs (per run name):
- model metrics JSONs
- per-model test predictions
- champion model table (`champions.csv`, selected on validation only)
- macro summary (`macro_summary.json`)

### 4) Backtest champion signals

```bash
python scripts/run_backtest.py --config configs/config.yaml
```

Backtest policy (configurable):
- entry: next bar open after breakout signal
- TP: measured move (primary)
- SL: pattern invalidation plus ATR buffer
- exit: TP / SL / time stop
- metrics: total PnL, Sharpe, win rate, profit factor, max drawdown, expectancy

### 5) Render visual verification gallery

```bash
python scripts/render_pattern_gallery.py --config configs/config.yaml
```

Generates TP/FP/FN images for manual sanity checking.

## Reliability and leakage controls

Implemented controls include:
- strict time-based split
- enforced minimum embargo around split boundaries (`lookback + confirmation horizon`)
- fail-fast behavior when embargo would empty val/test folds (no silent zero-embargo fallback)
- champion selection on validation only (no test-set model picking)
- hard-negative downsampling applied to train only (validation/test stay untouched)
- causal smoothing/extrema detection for pattern events (no future-bar dependence)
- train-only fit behavior for model preprocessing
- breakout-time label anchoring
- confidence intervals (bootstrap) for precision, recall, and F1
- minimum test positive-support gate before reliability claims

## Smoke test (CI/local quick check)

```bash
pytest -q
```

Test suite includes ingestion mapping, timezone conversion, resampling, pattern fixtures, leakage-safe splitting, metric reproducibility, backtest arithmetic, and integration smoke run.

## Config-first experimentation

Use `--config-override` and/or `--set` for controlled experiments:

```bash
python scripts/run_experiment_suite.py \
  --config configs/config.yaml \
  --set project.run_name=overnight_optuna \
  --set optuna.enabled=true \
  --set optuna.n_trials=200
```

## Academic transparency

Pattern math and assumptions are documented in [PATTERNS_EXPLAINED.md](/Users/oskarrodziewicz/Library/CloudStorage/OneDrive-UniversityofWarwick/Deep%20Learning/Candlestick-Pattern-Recognition/PATTERNS_EXPLAINED.md), including breakout confirmation logic and 15-minute timeframe assumptions.
