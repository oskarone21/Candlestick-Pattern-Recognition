"""
ResNet-18 image baseline for candlestick pattern classification.

Renders each OHLCV window as a 224x224 candlestick chart image using
mplfinance, then fine-tunes a pretrained ResNet-18 classifier.

Used only when cfg['approach']['run_image_baseline'] = true.

Follow TEAM_STANDARDS.md: config-driven, single responsibility.
"""

import os
import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import mplfinance as mpf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------

def render_window_to_image(
    window_df: pd.DataFrame,
    image_size: int = 224,
) -> Image.Image:
    """Render one OHLCV window to a PIL image using mplfinance.

    Axes, titles, and dates are removed so the model sees only candle shapes.

    Parameters
    ----------
    window_df : pd.DataFrame
        OHLCV slice with DatetimeIndex and columns Open/High/Low/Close/Volume.
    image_size : int
        Output image width and height in pixels.

    Returns
    -------
    PIL.Image.Image
        RGB image of size (image_size, image_size).
    """
    buf = io.BytesIO()
    dpi = 72
    fig_size = image_size / dpi

    fig, axes = mpf.plot(
        window_df,
        type="candle",
        volume=True,
        style="charles",
        figsize=(fig_size, fig_size),
        axisoff=True,
        tight_layout=True,
        returnfig=True,
        warn_too_much_data=len(window_df) + 1,
    )
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((image_size, image_size))
    return img


def build_image_dataset(
    labeled_windows: list,
    df_ohlcv: pd.DataFrame,
    cfg: dict,
    cache_dir: str = "outputs/images",
) -> tuple[list[Image.Image], np.ndarray]:
    """Render all labeled windows to images.

    Caches rendered PNGs to disk so re-runs are fast.

    Parameters
    ----------
    labeled_windows : list[LabeledWindow]
    df_ohlcv : pd.DataFrame
        Full OHLCV DataFrame (un-normalised).
    cfg : dict
    cache_dir : str

    Returns
    -------
    images : list[PIL.Image.Image]
    labels : np.ndarray
    """
    os.makedirs(cache_dir, exist_ok=True)
    lookback  = cfg["windowing"]["lookback_bars"]
    img_size  = cfg["chart_images"]["width_px"]
    col_cfg   = cfg["data_source"]["columns"]

    ohlcv_cols = [col_cfg["open"], col_cfg["high"],
                  col_cfg["low"],  col_cfg["close"], col_cfg["volume"]]

    images, labels = [], []
    total = len(labeled_windows)

    for i, lw in enumerate(labeled_windows):
        cache_path = os.path.join(cache_dir, f"w{lw.anchor_bar}_l{lw.label}.png")

        if os.path.exists(cache_path):
            img = Image.open(cache_path).convert("RGB")
        else:
            start = lw.anchor_bar - lookback + 1
            end   = lw.anchor_bar + 1
            slice_df = df_ohlcv[ohlcv_cols].iloc[start:end].copy()
            slice_df.columns = ["Open", "High", "Low", "Close", "Volume"]
            img = render_window_to_image(slice_df, image_size=img_size)
            img.save(cache_path)

        images.append(img)
        labels.append(lw.label)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"[resnet] Rendered {i+1}/{total} images", end="\r")

    print()
    return images, np.array(labels)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImagePatternDataset(Dataset):
    """Dataset wrapping PIL images + labels with ImageNet transforms."""

    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, images: list, labels: np.ndarray) -> None:
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        return self._transform(self.images[idx]), self.labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_resnet(cfg: dict, device: torch.device) -> nn.Module:
    """Load pretrained ResNet-18 and replace the final FC layer.

    Parameters
    ----------
    cfg : dict
    device : torch.device

    Returns
    -------
    nn.Module
    """
    num_classes = cfg["model"]["num_classes"]
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze all layers except last residual block + FC
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[resnet] ResNet-18 baseline — {n_params:,} trainable params | device: {device}")
    return model


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_resnet(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> dict:
    """Train ResNet baseline. Same structure as TCN trainer."""
    opt_cfg   = cfg["optimizer"]
    epochs    = cfg["training"]["epochs"]
    patience  = cfg["training"]["early_stopping_patience"]

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"],
    )
    pos_weight = torch.tensor([1.0, 3.0]).to(device)
    criterion  = nn.CrossEntropyLoss(weight=pos_weight)

    best_f1, patience_ctr = -1.0, 0
    history = {"train_loss": [], "val_f1": []}
    ckpt = os.path.join(cfg["paths"]["checkpoints_dir"], "resnet_best.pt")
    os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)

    print(f"\n[resnet] Training ResNet-18 baseline — {epochs} epochs")
    print(f"{'Epoch':>6}  {'TrainLoss':>10}  {'ValF1':>7}")
    print("-" * 28)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_b)
        train_loss = total_loss / len(train_loader.dataset)

        val_f1 = _eval_f1(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_f1)
        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_f1:>7.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), ckpt)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[resnet] Early stopping at epoch {epoch}")
                break

    history["best_val_f1"] = best_f1
    history["checkpoint"]  = ckpt
    print(f"[resnet] Best val F1: {best_f1:.4f}")
    return history


def evaluate_resnet(
    model: nn.Module,
    test_loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> dict:
    """Load best checkpoint and report test metrics."""
    ckpt = os.path.join(cfg["paths"]["checkpoints_dir"], "resnet_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            p = model(X_b.to(device)).argmax(1).cpu().numpy()
            preds.extend(p)
            labels.extend(y_b.numpy())

    report = classification_report(
        labels, preds,
        target_names=["Negative", "Double Bottom"],
        output_dict=True, zero_division=0,
    )
    print("\n[resnet] === ResNet-18 Test Results ===")
    print(classification_report(labels, preds,
          target_names=["Negative", "Double Bottom"], zero_division=0))
    return report


def _eval_f1(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            p = model(X_b.to(device)).argmax(1).cpu().numpy()
            preds.extend(p); labels.extend(y_b.numpy())
    return f1_score(labels, preds, average="macro", zero_division=0)
