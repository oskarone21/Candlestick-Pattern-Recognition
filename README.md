# SL-TCN — Chart-pattern recognition with TCNs (Henry Liao)

This branch is **my individual modelling pipeline** for the WM9B7 group project, in which each team member built their own model independently. It detects four classic chart patterns on 15-minute NQ futures bars: Head & Shoulders, Inverse Head & Shoulders, Double Top and Double Bottom.

## Results (held-out test split, leakage-guarded)

Mean ± std over **3 random seeds** (42, 43, 44); data, splits and augmentation are identical across seeds, only initialisation and batch order change.

| Pattern | Test positives | F1 (positive class) | PR-AUC |
|---|---:|---:|---:|
| Head & Shoulders | 19 | 0.80 ± 0.02 | 0.81 ± 0.03 |
| Inverse H&S | 33 | 0.59 ± 0.04 | 0.75 ± 0.03 |
| Double Top | 19 | 0.59 ± 0.04 | 0.68 ± 0.04 |
| Double Bottom | 36 | 0.57 ± 0.07 | 0.61 ± 0.05 |
| **Mean** | | **0.64 ± 0.03** | **0.71 ± 0.00** |

Positives are rare (~1:10 in validation/test), so the headline metric is **positive-class F1 and PR-AUC**. Macro F1 (about 0.79 at the primary seed) averages in the easy negative class and overstates performance; it is reported in the notebook for reference only. Test sets contain only 19–36 positives per pattern, so per-pattern scores carry wide uncertainty — a single run can move mean F1 by ±0.03, which is why results are averaged over seeds.

### Architecture check: receptive field vs window

The original 4-block TCN (kernel 3, dilations 1–8) has a receptive field of 61 bars, shorter than the 160-bar H&S windows, so the earliest bars could not influence the prediction. A 6-block variant (receptive field 253 bars, covering the full window) was trained under identical conditions and compared on **validation** mean positive-class F1 over the same 3 seeds:

| Variant | Receptive field | Validation F1 | Test F1 | Test PR-AUC |
|---|---:|---:|---:|---:|
| 4 blocks (selected) | 61 | 0.707 ± 0.019 | 0.637 ± 0.027 | 0.712 ± 0.004 |
| 6 blocks | 253 | 0.701 ± 0.012 | 0.586 ± 0.010 | 0.671 ± 0.025 |

The deeper model did not help and overfits earlier (best epoch 2–7 vs 5–23). Windows end at the breakout bar and the confirmation rule requires the breakout within 60 bars of the right shoulder, so the discriminative evidence sits in the last ~60 bars, which the 4-block model already covers. Note the 6-block model also has about twice the parameters, so depth and capacity are confounded in this comparison.

### SHAP (after fixing input normalisation)

An earlier SHAP run explained **raw** OHLCV windows while the model was trained on per-window z-scored inputs; those attributions (mean |SHAP| 80–280) were invalid. After routing SHAP through the same normalisation, mean |SHAP| per feature (test positives, primary seed) is:

| Pattern | open | high | low | close | volume |
|---|---:|---:|---:|---:|---:|
| Head & Shoulders | 0.010 | 0.008 | **0.018** | 0.015 | 0.008 |
| Inverse H&S | 0.007 | 0.008 | 0.007 | **0.010** | 0.007 |
| Double Top | 0.011 | 0.010 | **0.015** | 0.013 | 0.010 |
| Double Bottom | 0.009 | 0.010 | 0.010 | 0.012 | **0.013** |

Differences between features are modest (≤2×), so no single feature dominates. The low price carries the most weight for the two bearish patterns, consistent with their confirmation being a close below the neckline.

## What I built

- **Labelling:** causal Nadaraya–Watson smoothing (no look-ahead) → extrema → geometric, volume and breakout-confirmation rules, following Lo, Mamaysky & Wang (2000). Scanning at 5 bandwidths and de-duplicating raised confirmed samples from 12 (single bandwidth) to several hundred.
- **Model:** one-vs-rest TCN binary classifiers with last-timestep readout; focal loss + class weighting for imbalance; training-only augmentation (jitter, magnitude scaling, time warp); early stopping; `TCN.receptive_field()` is asserted against the window length.
- **Evaluation:** chronological train/val/test split with a one-look-back **embargo** between splits (`dataset.chronological_split_with_embargo`, unit-tested), so no window shares any bar across splits; decision threshold tuned on validation only; architecture selected on validation over 3 seeds; test set used for reporting only.
- **Explainability:** SHAP GradientExplainer over (time step × OHLCV feature), computed on the same normalised inputs the model was trained on.
- **Ablation:** nested-candidate filtering (notebook section 8).

## Known limitations

- **Window anchoring differs between classes.** Positive windows end at the confirmed breakout bar; hard-negative windows (valid geometry, no confirmed breakout within 60 bars) end at the right shoulder. The model can therefore separate the classes partly from how the window ends rather than from the pattern itself, and the task is post-hoc confirmation, not prediction. Aligning both classes to the right shoulder (predict whether a breakout will follow) is the natural next step; it would make the task harder and the scores lower, but more meaningful.
- **Small test sets** (19–36 positives per pattern): per-pattern F1 has wide uncertainty; only the seed-averaged mean should be quoted.
- **Rule-based labels:** the model learns to reproduce the labelling rules, not ground truth. Label parameters (0.5% breakout, 60-bar horizon, 3% symmetry, 1.2× volume) were set by hand and not tuned.
- **Depth vs capacity confounded** in the 4- vs 6-block comparison (see above).

## Repository layout

| Path | Contents |
|---|---|
| `henry_pattern_recognition.ipynb` | End-to-end run with outputs (data → labels → training → evaluation → SHAP → ablation) |
| `src/candlestick/` | Pipeline code: `smoothing`, `labeling`, `dataset`, `model`, `training`, `evaluation` |
| `configs/henry_tcn.yaml` | Experiment config used for the results above (merged on top of `configs/config.yaml`) |
| `tests/test_henry_pipeline.py` | Split-leakage, embargo and metric tests |
| `SL_LSTM_SHAP.ipynb`, `baseline/` | Earlier LSTM + SHAP baseline that preceded the TCN pipeline |

## How to run

```bash
pip install -r requirements.txt          # macOS: requirements.macos.txt
python -m pytest tests/test_henry_pipeline.py
jupyter lab henry_pattern_recognition.ipynb
```

**Data:** the notebook expects 1-minute NQ futures bars at `data/nq_1min_clean.csv` (columns `ts_event, Open, High, Low, Close, Volume`; 2021-01 to 2026-01). The file was shared privately within the team and is not redistributed here.

---

<details>
<summary>Original team README (shared project context)</summary>

## Automated Candlestick Chart Pattern Recognition

This repository contains a deep learning project for automated chart-pattern recognition from futures OHLCV data.

The updated project approach is hybrid:
- Primary path: rule-based pattern labeling + sequence modeling on OHLCV windows
- Optional comparison path: candlestick image rendering + CNN baseline

### Important data file note

Before running the project, manually place `nq_1min.csv` inside the `data/` folder (`data/nq_1min.csv`). This file is too large to store in this GitHub repository.

### Project goal

Investigate whether objective, rule-defined chart patterns can be learned from market data, and compare sequence-based models against optional image-based baselines.

### Modeling approach

1. Smooth 15-minute close prices with Nadaraya-Watson kernel regression and extract extrema from first/second derivative conditions.
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

### Academic basis and justification

The labeling schema in `configs/config.yaml` and formulas in `PATTERNS_EXPLAINED.md` are based on the following literature:

- Lo, Mamaysky, and Wang (2000): formalized technical pattern detection with nonparametric methods and statistical testing. Link: `https://doi.org/10.1111/0022-1082.00265`
- Osler and Chang (1995): objective algorithmic head-and-shoulders detection with out-of-sample style evaluation. Link: `https://www.newyorkfed.org/research/staff_reports/sr4.html`
- Savin, Weller, and Zvingelis (2007): predictive power evidence for head-and-shoulders in U.S. equities. Link: `https://doi.org/10.1093/jjfinec/nbl012`
- Nadaraya (1964): foundational kernel regression estimator used for smoothing noisy price series. Link: `https://doi.org/10.1137/1109020`
- Hurvich, Simonoff, and Tsai (1998): improved AIC (AICc) for nonparametric smoothing parameter selection. Link: `https://doi.org/10.1111/1467-9868.00125`
- Bulkowski (3rd ed.): empirical breakout and volume confirmation heuristics used as practical bounds in configuration.

### Team development standards

See `TEAM_STANDARDS.md` for shared coding conventions, config usage rules, and AI-assisted development practices. This keeps team contributions consistent and prevents accidental drift from the shared configuration and workflow.

### Configuration

The single source of truth is `configs/config.yaml`.

- Change shared defaults there (timeframe, labels, optimizer, training settings).
- Use local override files for personal experiments.
- `chart_images` controls candlestick rendering and is only used when image input is enabled.

### Tech stack

- Data processing: `numpy`, `pandas`
- Chart rendering and image handling: `matplotlib`, `mplfinance`, `pillow`, `opencv-python-headless`
- Data augmentation: `albumentations`
- Deep learning: `torch`, `torchvision`
- Training and evaluation: `scikit-learn`, `tqdm`, `tensorboard`
- Utilities and notebooks: `pyyaml`, `jupyterlab`, `ipykernel`

All pinned versions are listed in `requirements.txt`.

### Prerequisites

- Git
- Docker Desktop, or Docker Engine with Compose plugin
- For GPU mode: NVIDIA GPU, recent NVIDIA drivers, and NVIDIA container runtime support
- Optional local setup: Python virtual environment support

### Docker workflow (main setup)

Docker is the shared Python environment for this project:

- The image contains Python and all required packages.
- The container runs scripts and Jupyter Lab.
- The repository folder is mounted into the container, so notebooks and outputs persist on your machine.

#### 1) Clone the repository

```bash
git clone https://github.com/oskarone21/Candlestick-Pattern-Recognition.git
cd Candlestick-Pattern-Recognition
```

#### 2) Build the Docker image

```bash
docker compose build
```

If you are using Apple Silicon and hit an architecture issue:

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose build
```

### Run the container

#### Option 1: Open a shell

GPU mode:

```bash
docker compose run --rm app bash
```

CPU mode:

```bash
docker compose -f docker-compose.cpu.yml run --rm app bash
```

#### Option 2: Run Jupyter Lab (recommended)

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

### Automatic GPU/CPU selection

The helper script checks Docker runtime support. If NVIDIA runtime is available, it uses GPU mode; otherwise it falls back to CPU mode.

```bash
bash scripts/dev-shell.sh
```

Pass a command directly if needed:

```bash
bash scripts/dev-shell.sh python -c "import torch; print(torch.cuda.is_available())"
```

### Verify CUDA availability

Inside the container, run:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output:

- Mac or CPU mode: `CUDA available: False`
- GPU machines: `CUDA available: True` and a GPU name

### Working with notebooks

Create notebooks inside this repository (for example in `notebooks/`). Because the project folder is mounted into the container, notebook files persist locally and can be committed to Git.

### Optional local setup (without Docker)

#### macOS Apple Silicon

Use the dedicated local requirements file. The main `requirements.txt` is pinned for the team's CUDA/Docker workflow and is not appropriate for Apple Silicon.

```bash
bash scripts/setup_macos.sh
source .venv/bin/activate
```

Quick device check:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

#### Linux / Windows with NVIDIA CUDA

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

### Key project files

- `Dockerfile` - CUDA-enabled PyTorch base image and dependency installation
- `docker-compose.yml` - default Docker Compose configuration with GPU access
- `docker-compose.cpu.yml` - Compose override for systems without NVIDIA runtime support
- `scripts/dev-shell.sh` - helper script that automatically selects GPU or CPU mode
- `configs/config.yaml` - shared experiment and pipeline configuration
- `TEAM_STANDARDS.md` - team-wide coding, configuration, and collaboration standards
- `.dockerignore` - excludes large or temporary files from Docker build context
- `requirements.txt` - pinned Python package versions

### Troubleshooting

#### Docker is not running

Start Docker Desktop (or Docker daemon) before running Compose commands.

#### Port 8888 is already in use

Run Jupyter on another port:

```bash
docker compose run --rm --service-ports app jupyter lab --ip=0.0.0.0 --port=8890 --no-browser --allow-root
```

Then open `http://localhost:8890`.

#### No GPU detected

Use CPU mode:

```bash
docker compose -f docker-compose.cpu.yml run --rm app bash
```

#### Apple Silicon

Apple Silicon does not support CUDA. Use CPU mode.

</details>
