# Automated Candlestick Chart Pattern Recognition

This repository contains a deep learning project for automated chart-pattern recognition from futures OHLCV data.

The updated project approach is hybrid:
- Primary path: rule-based pattern labeling + sequence modeling on OHLCV windows
- Optional comparison path: candlestick image rendering + CNN baseline

## Important data file note

Before running the project, manually place `nq_1min.csv` inside the `data/` folder (`data/nq_1min.csv`). This file is too large to store in this GitHub repository.

## Project goal

Investigate whether objective, rule-defined chart patterns can be learned from market data, and compare sequence-based models against optional image-based baselines.

## Updated modeling approach

1. Build objective labels using numeric pattern rules (head and shoulders, inverse head and shoulders, double top, double bottom, no-pattern).
2. Train a primary sequence model directly on OHLCV windows.
3. Optionally train an image-CNN baseline on rendered candlestick windows for comparison.
4. Evaluate all approaches on the same time-based split using per-class precision, recall, and F1.

Why this approach:

- Pattern definitions stay auditable and reproducible.
- Sequence models use raw market structure directly and avoid chart-rendering artifacts.
- Optional image baseline still lets the team test the original vision idea.

## Team development standards

See `TEAM_STANDARDS.md` for shared coding conventions, config usage rules, and AI-assisted development practices. This keeps team contributions consistent and prevents accidental drift from the shared configuration and workflow.

## Configuration

The single source of truth is `configs/config.yaml`.

- Change shared defaults there (timeframe, labels, optimizer, training settings).
- Use local override files for personal experiments.
- `chart_images` controls candlestick rendering and is only used when image input is enabled.

## Tech stack

- Data processing: `numpy`, `pandas`
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

### 2) Build the Docker image

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
- `scripts/dev-shell.sh` - helper script that automatically selects GPU or CPU mode
- `configs/config.yaml` - shared experiment and pipeline configuration
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
