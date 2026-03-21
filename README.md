# Automated Candlestick Chart Pattern Recognition

This repository is a group deep learning project that converts OHLCV time-series windows into candlestick-chart images and trains a CNN (for example, a fine-tuned ResNet) to classify chart patterns such as bull flags and consolidations.

## Required libraries

The project uses these Python libraries:

- Data processing: `numpy`, `pandas`
- Chart rendering and image handling: `matplotlib`, `mplfinance`, `pillow`, `opencv-python-headless`
- Data augmentation: `albumentations`
- Deep learning: `torch`, `torchvision`
- Training and evaluation: `scikit-learn`, `tqdm`, `tensorboard`
- Utilities/notebooks: `pyyaml`, `jupyterlab`, `ipykernel`

All pinned versions are in `requirements.txt`.

## Setup guide (Docker)

### 1) Prerequisites

- Docker Desktop (or Docker Engine + Compose plugin)
- For GPU mode (default): NVIDIA GPU, recent NVIDIA driver, and NVIDIA container runtime support in Docker

### 2) Build the image

```bash
docker compose build
```

If you are on Apple Silicon and hit a base-image architecture issue, build with:

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose build
```

### 3) Run with GPU (default)

This is the default mode and is intended for your university lab computers.

```bash
docker compose run --rm app bash
```

### 4) Run with CPU override (Mac and non-NVIDIA systems)

Use the CPU override file when GPU runtime is unavailable:

```bash
docker compose -f docker-compose.cpu.yml run --rm app bash
```

### 5) Automatic fallback script

The script below checks whether Docker has NVIDIA runtime support. If available, it runs GPU mode; otherwise it falls back to CPU mode.

```bash
bash scripts/dev-shell.sh
```

You can also pass a command, for example:

```bash
bash scripts/dev-shell.sh python -c "import torch; print(torch.cuda.is_available())"
```

### 6) Verify CUDA availability

Inside the container:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected result:

- On Mac/CPU mode: `CUDA available: False`
- On lab GPU machines: `CUDA available: True` and a GPU name

### 7) Run Jupyter Lab (optional)

GPU mode:

```bash
docker compose run --rm --service-ports app jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

CPU override mode:

```bash
docker compose -f docker-compose.cpu.yml run --rm --service-ports app jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then open `http://localhost:8888` in your browser.

## Local setup without Docker (optional)

If a team member wants to run directly on their machine:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Docker files included

- `Dockerfile`: CUDA-enabled PyTorch base image + dependency installation
- `docker-compose.yml`: default Compose config that requests GPU access
- `docker-compose.cpu.yml`: CPU override for non-NVIDIA machines
- `scripts/dev-shell.sh`: auto-selects GPU mode when available, CPU mode otherwise
- `.dockerignore`: avoids sending large/temporary files to Docker build context

## Suggested team split

- Member 1: transform OHLCV windows into candlestick images
- Member 2: implement augmentation pipeline and dataset loaders
- Member 3: train/evaluate CNN and report metrics
