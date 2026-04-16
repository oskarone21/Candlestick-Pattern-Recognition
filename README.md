# Automated Candlestick Pattern Recognition

This repository contains two related pipelines built on top of 1-minute OHLCV data:

- A research pipeline for rule-based candlestick pattern detection plus sequence-model training.
- A two-stage trading product pipeline that keeps the same pattern detector as Stage 1, then adds a Stage 2 trade-selection model trained on forward trade outcomes.

The current default dataset path is `data/spy_1min_clean.csv`.

## What This Branch Adds

The new product pipeline is run from `run_trading_product.py` and does the following:

1. Loads and resamples the cleaned 1-minute CSV to 15-minute bars.
2. Generates pattern labels using the existing geometric detection logic.
3. Trains the Stage 1 TCN detector on those labels.
4. Tunes a Stage 1 probability threshold with a recall-first objective and a precision floor.
5. Builds Stage 2 trade labels from forward returns, stop-loss, take-profit, and round-trip costs.
6. Trains a Stage 2 tabular model to decide whether a Stage 1 signal is worth trading.
7. Tunes the Stage 2 execution threshold for profitability with a precision guard.
8. Runs a short paper-trading replay on the most recent bars.

This is still a research / paper-trading system. It does not yet connect to a broker or submit live orders.

## Data Setup

The repo expects a cleaned SPY file at:

```text
data/spy_1min_clean.csv
```

If you have the raw Kaggle file `1_min_SPY_2008-2021.csv`, clean it first:

```bash
python scripts/clean_spy_1min.py --input /absolute/path/to/1_min_SPY_2008-2021.csv --output data/spy_1min_clean.csv
```

If you prefer, copy the raw CSV into the repo first and then run:

```bash
python scripts/clean_spy_1min.py --input data/raw/1_min_SPY_2008-2021.csv --output data/spy_1min_clean.csv
```

The cleaner:

- parses timestamps
- removes exact duplicates
- merges conflicting minute rows
- drops zero-volume bars by default
- writes the pipeline-compatible columns `ts_event, Open, High, Low, Close, Volume`

## Prerequisites

- Git
- Docker Desktop or Docker Engine with Compose
- For local non-Docker runs: Python 3.10+

Python 3.10+ matters here because the codebase uses modern type syntax.

## Quick Start With Docker

### 1) Clone the repository

```bash
git clone https://github.com/oskarone21/Candlestick-Pattern-Recognition.git
cd Candlestick-Pattern-Recognition
```

### 2) Build the image

```bash
docker compose build
```

If you are on Apple Silicon and hit an architecture issue:

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose build
```

### 3) Open a shell in the container

GPU/default mode:

```bash
docker compose run --rm app bash
```

CPU mode:

```bash
docker compose -f docker-compose.cpu.yml run --rm app bash
```

### 4) Run one of the pipelines

Inside the container:

Research pipeline, double-bottom focused:

```bash
python run_all.py
```

Research pipeline, all four patterns:

```bash
python train_all_patterns.py
```

Two-stage trading product pipeline:

```bash
python run_trading_product.py
```

## Local Setup Without Docker

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

Then run the same scripts shown above.

## Azure ML Easiest Path

If you want the simplest cloud workflow, use an Azure Machine Learning compute instance plus the notebook:

```text
notebooks/05_azure_run_all_trading_product.ipynb
```

Recommended flow:

1. Create an Azure ML workspace and a compute instance.
2. Open the compute-instance terminal and clone this repo.
3. Copy the dataset to `/home/azureuser/data/spy_1min_clean.csv` on the compute instance.
4. Open `notebooks/05_azure_run_all_trading_product.ipynb`.
5. Leave `INSTALL_REQUIREMENTS = True` for the first run.
6. Press `Run All`.

What the notebook does:

- installs repo dependencies into the current notebook kernel
- runs `run_trading_product.py`
- loads `summary.json`, profitability CSVs, and replay outputs
- renders the key execution and equity charts inline
- uses `configs/azure_compute.yaml` by default so the dataset can live outside the repo mount

Suggested repo location on the compute instance:

```text
~/cloudfiles/code/Users/<your_user_name>/Candlestick-Pattern-Recognition
```

That location works well with Azure ML notebooks and terminals.

Notes:

- `configs/azure_compute.yaml` points to `/home/azureuser/data/spy_1min_clean.csv`.
- If you prefer to keep data inside the repo instead, change `CONFIG_OVERRIDE` in the notebook to `None`.
- The full rolling-window Stage 1 dataset is much heavier than the older selected-slice version.
- A GPU-backed compute instance is recommended if you want the full product run to finish faster.
- After the first successful install, set `INSTALL_REQUIREMENTS = False` in the notebook for later reruns.

## Which Script To Run

`run_all.py`

- Original research script.
- Hardwired to the double-bottom workflow.
- Useful for the older end-to-end detector/backtest path.

`train_all_patterns.py`

- Trains one Stage 1 detector per pattern.
- Good for comparing pattern-detection quality across all four patterns.

`run_trading_product.py`

- The new two-stage trading product pipeline.
- Uses `labeling.active_pattern` from `configs/config.yaml`.
- Produces Stage 1 metrics, Stage 2 trade metrics, profitability comparison, and a paper replay.

## Choosing The Pattern

For `run_trading_product.py`, the active pattern comes from:

```yaml
labeling:
  active_pattern: head_shoulders
```

Supported values:

- `head_shoulders`
- `inverse_head_shoulders`
- `double_top`
- `double_bottom`

You can either:

- edit `configs/config.yaml`, or
- pass `--config-override path/to/override.yaml`

Example:

```bash
python run_trading_product.py --config-override configs/my_override.yaml
```

## Product Config Knobs

The trading-product settings live under `product:` in `configs/config.yaml`.

Important sections:

- `product.purged_split`
- `product.stage1.threshold_search`
- `product.stage2.model`
- `product.stage2.trade_labeling`
- `product.stage2.threshold_search`
- `product.replay`

The main trading assumptions you can change are:

- embargo gap between splits
- Stage 1 detection threshold search range
- Stage 2 model settings
- trade horizon in bars
- take-profit and stop-loss percentages
- round-trip trading costs
- replay capital and per-trade size

## Outputs

### Research scripts

The original research scripts write under `outputs/` using the older layout.

### Two-stage trading product

`run_trading_product.py` writes under:

```text
outputs/product/<active_pattern>/
```

Key files:

- `stage1/threshold_search.csv`
- `stage1/threshold_metrics.json`
- `stage1/checkpoints/best_model.pt`
- `stage2/stage2_model.joblib`
- `stage2/feature_names.json`
- `stage2/threshold_search.csv`
- `stage2/trade_metrics.json`
- `stage2/test_trade_candidates.csv`
- `stage2/profitability_comparison.csv`
- `paper_replay/paper_trades.csv`
- `paper_replay/paper_equity.csv`
- `paper_replay/paper_summary.json`
- `summary.json`

## What The Two-Stage System Is Actually Predicting

Stage 1 predicts:

- whether the latest 80-bar window looks like a confirmed pattern event according to the rule-based labeler

Stage 2 predicts:

- whether taking that Stage 1 signal as a trade is likely to be profitable under the configured forward horizon, stop-loss, take-profit, and cost assumptions

That means the system is trying to move from:

- `pattern detected`

to:

- `pattern detected and worth trading`

## Paper Replay

The replay step in `run_trading_product.py` is a simple paper-trading simulation over the latest bars in the dataset.

It is useful for:

- checking signal frequency
- checking trade logs
- checking equity-curve shape
- validating that the full two-stage decision loop works end to end

It is not yet:

- a live broker integration
- a real order book simulator
- a market-microstructure execution engine

## Suggested First Run

After you have `data/spy_1min_clean.csv` in place:

```bash
python run_trading_product.py
```

Then inspect:

- `outputs/product/<pattern>/summary.json`
- `outputs/product/<pattern>/stage2/profitability_comparison.csv`
- `outputs/product/<pattern>/paper_replay/paper_summary.json`

## Key Files

- `configs/config.yaml` - shared config for both research and product pipelines
- `scripts/clean_spy_1min.py` - raw SPY cleaner
- `run_all.py` - original double-bottom research pipeline
- `train_all_patterns.py` - Stage 1 training across all patterns
- `run_trading_product.py` - new two-stage trading product pipeline
- `src/trading/split.py` - purged split logic
- `src/trading/inference.py` - Stage 1 probability inference and threshold tuning
- `src/trading/trade_labeling.py` - forward trade labels and causal Stage 2 features
- `src/trading/stage2.py` - Stage 2 model and profitability metrics
- `src/trading/paper_trader.py` - replay-style paper trading

## Troubleshooting

### Python version errors

If you see syntax errors around type hints, you are probably using Python older than 3.10. Use Docker or a newer local Python.

### Missing data file

If the run fails on file loading, check that:

- `configs/config.yaml` points to `data/spy_1min_clean.csv`
- that file actually exists

### No GPU detected

Use CPU mode:

```bash
docker compose -f docker-compose.cpu.yml run --rm app bash
```

### Apple Silicon

Apple Silicon does not support CUDA in this setup. Use CPU mode.
