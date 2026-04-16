#!/bin/bash
# ============================================================
# Henry's one-shot local setup for M4 Mac
# Run from project root:
#   cd ~/AAI/DL/Candlestick-Pattern-Recognition
#   bash scripts/setup-henry.sh
# ============================================================

set -e

echo "=== Step 1: Check Python version ==="
# Try common Python locations on Mac
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$candidate"
            echo "Found: $PYTHON ($("$PYTHON" --version))"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    echo "ERROR: Python 3.10+ not found."
    echo ""
    echo "Install with Homebrew (recommended for M4 Mac):"
    echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    echo "  brew install python@3.12"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

echo ""
echo "=== Step 2: Create fresh virtual environment ==="
if [ -d ".venv" ]; then
    echo "Removing old .venv (Python 3.9)..."
    rm -rf .venv
fi
"$PYTHON" -m venv .venv
source .venv/bin/activate
echo "Created .venv with $(python3 --version)"

echo ""
echo "=== Step 3: Upgrade pip ==="
pip install --upgrade pip

echo ""
echo "=== Step 4: Install PyTorch (CPU + MPS for Apple Silicon) ==="
# M4 Mac: use default torch which includes MPS support
pip install torch torchvision

echo ""
echo "=== Step 5: Install project dependencies ==="
# Core deps from requirements.txt (skip CUDA-specific torch since we installed above)
pip install \
    numpy==1.26.4 \
    pandas==2.2.3 \
    matplotlib==3.9.2 \
    mplfinance==0.12.10b0 \
    pillow==11.0.0 \
    "opencv-python-headless==4.10.0.84" \
    scikit-learn==1.5.2 \
    tqdm==4.67.1 \
    tensorboard==2.18.0 \
    pyyaml==6.0.2 \
    jupyterlab==4.3.1 \
    ipykernel==6.29.5 \
    scipy \
    shap

echo ""
echo "=== Step 6: Register Jupyter kernel ==="
python3 -m ipykernel install --user --name=candlestick --display-name="Candlestick (venv)"

echo ""
echo "=== Step 7: Verify installation ==="
python3 -c "
import torch
import numpy as np
import pandas as pd
import sklearn
import yaml

print('PyTorch:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
print('NumPy:', np.__version__)
print('Pandas:', pd.__version__)
print('scikit-learn:', sklearn.__version__)

# Quick model test
from torch import nn
model = nn.Linear(5, 2)
if torch.backends.mps.is_available():
    model = model.to('mps')
    x = torch.randn(4, 5, device='mps')
    out = model(x)
    print('MPS inference: OK')
else:
    print('MPS not available, will use CPU (still fine)')

print()
print('ALL CHECKS PASSED')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To start working:"
echo "  cd $(pwd)"
echo "  source .venv/bin/activate"
echo "  jupyter lab"
echo ""
echo "Then open: henry_pattern_recognition.ipynb"
echo "Make sure to select kernel: 'Candlestick (venv)'"
echo ""
