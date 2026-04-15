# Automated Candlestick Chart Pattern Recognition

This repository contains a deep learning project for automated chart-pattern recognition from OHLCV data.

The updated project approach is hybrid:
- Primary path: rule-based pattern labeling + sequence modeling on OHLCV windows
- Optional comparison path: candlestick image rendering + CNN baseline

## Data download

Before running the project, download QQQ daily OHLCV data from Yahoo Finance:

```bash
pip install yfinance
python scripts/download_qqq.py
```

Options (ticker, years, output path):

```bash
python scripts/download_qqq.py --ticker SPY --years 5 --output data/SPY_1d.csv
```

The default command saves `data/QQQ_1d.csv` (10 years of QQQ daily bars).

## Project goal

Investigate whether objective, rule-defined chart patterns can be learned from market data, and compare sequence-based models against optional image-based baselines.

## Modeling approach

1. Smooth daily close prices with Nadaraya-Watson kernel regression and extract extrema from first/second derivative conditions.
2. Build strict geometric labels for head and shoulders, inverse head and shoulders, double top, and double bottom.
3. Enforce volume-confirmation constraints and assign positive labels only on confirmed neckline breakouts.
4. Train a primary sequence model directly on OHLCV windows using one-vs-rest binary runs (one pattern per run).
5. Optionally train an image-CNN baseline on rendered candlestick windows for comparison.
6. Evaluate all approaches on the same time-based split using precision, recall, and F1.

Why this approach:

- Pattern definitions stay auditable and reproducible.
- Labels are tied to academically defined breakout events, reducing target noise.
- Sequence models use raw market structure directly and avoid chart-rendering artifacts.
- Optional image baseline still lets the team test the original vision idea.

## Academic basis and justification

The labeling schema in `configs/config.yaml` and formulas in `PATTERNS_EXPLAINED.md` are based on the following literature:

- Lo, Mamaysky, and Wang (2000): formalized technical pattern detection with nonparametric methods and statistical testing. Link: `https://doi.org/10.1111/0022-1082.00265`
- Osler and Chang (1995): objective algorithmic head-and-shoulders detection with out-of-sample style evaluation. Link: `https://www.newyorkfed.org/research/staff_reports/sr4.html`
- Savin, Weller, and Zvingelis (2007): predictive power evidence for head-and-shoulders in U.S. equities. Link: `https://doi.org/10.1093/jjfinec/nbl012`
- Nadaraya (1964): foundational kernel regression estimator used for smoothing noisy price series. Link: `https://doi.org/10.1137/1109020`
- Hurvich, Simonoff, and Tsai (1998): improved AIC (AICc) for nonparametric smoothing parameter selection. Link: `https://doi.org/10.1111/1467-9868.00125`
- Bulkowski (3rd ed.): empirical breakout and volume confirmation heuristics used as practical bounds in configuration.

## Team development standards

See `TEAM_STANDARDS.md` for shared coding conventions, config usage rules, and AI-assisted development practices. This keeps team contributions consistent and prevents accidental drift from the shared configuration and workflow.

## Configuration

The single source of truth is `configs/config.yaml`.

- Change shared defaults there (timeframe, labels, optimizer, training settings).
- Use local override files for personal experiments.
- `chart_images` controls candlestick rendering and is only used when image input is enabled.

## Double-top PoC workflow

The repository now includes a config-driven proof-of-concept pipeline for `double_top` with:

- staged sensitivity sweep (timeframe + labeling settings),
- breakout-time labeling,
- candlestick visualization with extrema/neckline/breakout overlays,
- sequence model comparison (`tcn`, `lstm`, `transformer`) when PyTorch is available.

### Device resolution order

When `project.device: auto`, the runtime resolves device in this order:

1. `cuda`
2. `mps`
3. `cpu`

On `mps`, the pipeline uses `float32` and disables AMP by default.

### Main commands

Label generation + chart audit:

```bash
python -m candlestick.cli.label \
  --config configs/config.yaml \
  --set labeling.active_pattern=double_top
```

Labeling sensitivity sweep only:

```bash
python -m candlestick.cli.sweep \
  --config configs/config.yaml \
  --profile local_quick \
  --set labeling.active_pattern=double_top
```

Single-config training run:

```bash
python -m candlestick.cli.train \
  --config configs/config.yaml \
  --models tcn lstm transformer \
  --set labeling.active_pattern=double_top
```

End-to-end profile (sweep + visualization + model comparison):

```bash
python -m candlestick.cli.run_profile \
  --config configs/config.yaml \
  --profile local_quick \
  --set labeling.active_pattern=double_top
```

Artifacts are saved under `outputs/<timestamp>_<run_name>/`.

## Tech stack

- Data processing: `numpy`, `pandas`
- Data download: `yfinance`
- Chart rendering and image handling: `matplotlib`, `mplfinance`, `pillow`, `opencv-python-headless`
- Data augmentation: `albumentations`
- Deep learning: `torch`, `torchvision`
- Training and evaluation: `scikit-learn`, `tqdm`, `tensorboard`
- Utilities and notebooks: `pyyaml`, `jupyterlab`, `ipykernel`

All pinned versions are listed in `requirements.txt`.

## Prerequisites

- Git
- Docker Desktop, or Docker Engine with Compose plugin
- For GPU mode: NVIDIA GPU, recent NVIDIA drivers, and NVIDIA container runtime support
- Optional local setup: Python virtual environment support

## Docker workflow (main setup)

Docker is the shared Python environment for this project:

- The image contains Python and all required packages.
- The container runs scripts and Jupyter Lab.
- The repository folder is mounted into the container, so notebooks and outputs persist on your machine.

### 1) Clone the repository

```bash
git clone https://github.com/oskarone21/Candlestick-Pattern-Recognition.git
cd Candlestick-Pattern-Recognition
```

### 2) Download data

```bash
pip install yfinance
python scripts/download_qqq.py
```

### 3) Build the Docker image

```bash
docker compose build
```

If you are using Apple Silicon and hit an architecture issue:

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose build
```

## Run the container

### Option 1: Open a shell

GPU mode:

```bash
docker compose run --rm app bash
```

CPU mode:

```bash
docker compose -f docker-compose.cpu.yml run --rm app bash
```

### Option 2: Run Jupyter Lab (recommended)

GPU mode:

```bash
docker compose run --rm --service-ports app jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

CPU mode:

```bash
docker compose -f docker-compose.cpu.yml run --rm --service-ports app jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Open:

```text
http://localhost:8888
```

Jupyter prints an access token in the terminal when it starts.

## Automatic GPU/CPU selection

The helper script checks Docker runtime support. If NVIDIA runtime is available, it uses GPU mode; otherwise it falls back to CPU mode.

```bash
bash scripts/dev-shell.sh
```

Pass a command directly if needed:

```bash
bash scripts/dev-shell.sh python -c "import torch; print(torch.cuda.is_available())"
```

## Verify CUDA availability

Inside the container, run:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output:

- Mac or CPU mode: `CUDA available: False`
- GPU machines: `CUDA available: True` and a GPU name

## Working with notebooks

Create notebooks inside this repository (for example in `notebooks/`). Because the project folder is mounted into the container, notebook files persist locally and can be committed to Git.

## Optional local setup (without Docker)

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

## Key project files

- `Dockerfile` - CUDA-enabled PyTorch base image and dependency installation
- `docker-compose.yml` - default Docker Compose configuration with GPU access
- `docker-compose.cpu.yml` - Compose override for systems without NVIDIA runtime support
- `scripts/download_qqq.py` - Yahoo Finance data download script
- `scripts/dev-shell.sh` - helper script that automatically selects GPU or CPU mode
- `configs/config.yaml` - shared experiment and pipeline configuration
- `candlestick/cli/run_profile.py` - end-to-end double-top PoC runner
- `candlestick/cli/sweep.py` - staged labeling sensitivity sweep runner
- `candlestick/cli/label.py` - labeling and chart-visualization runner
- `candlestick/cli/train.py` - single-config model training runner
- `TEAM_STANDARDS.md` - team-wide coding, configuration, and collaboration standards
- `.dockerignore` - excludes large or temporary files from Docker build context
- `requirements.txt` - pinned Python package versions

## Troubleshooting

### Docker is not running

Start Docker Desktop (or Docker daemon) before running Compose commands.

### Port 8888 is already in use

Run Jupyter on another port:

```bash
docker compose run --rm --service-ports app jupyter lab --ip=0.0.0.0 --port=8890 --no-browser --allow-root
```

Then open `http://localhost:8890`.

### No GPU detected

Use CPU mode:

```bash
docker compose -f docker-compose.cpu.yml run --rm app bash
```

### Apple Silicon

Apple Silicon does not support CUDA. Use CPU mode.
