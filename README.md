# Automated Candlestick Chart Pattern Recognition

This repository contains a deep learning project for automated candlestick chart pattern recognition. It transforms OHLCV time-series windows into candlestick chart images, then trains a CNN (for example, a fine-tuned ResNet) to classify patterns such as bull flags and consolidations.

## Important data file note

Before running the project, manually place `nq_1min.csv` inside the `data/` folder (`data/nq_1min.csv`). This file is too large to store in this GitHub repository.

## Project goal

Investigate whether image-based deep learning models can identify technical chart patterns from financial time-series data after transforming OHLCV windows into candlestick chart images.

## Team development standards

See `TEAM_STANDARDS.md` for shared coding conventions, config usage rules, and AI-assisted development practices. This keeps team contributions consistent and prevents accidental drift from the shared configuration and workflow.

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
