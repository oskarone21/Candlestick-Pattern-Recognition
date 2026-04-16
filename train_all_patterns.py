"""
Train TCN models for all four candlestick patterns sequentially.
Saves checkpoints and metrics per pattern under outputs/<pattern>/.
"""

import argparse
import os, json, copy, warnings
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# ── project root ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, ROOT)

from src.data.loader    import load_raw_ohlcv
from src.data.resampler import resample_ohlcv
from src.labeling.smoother import smooth_ohlcv
from src.labeling.extrema  import find_extrema
from src.labeling.head_shoulders         import label_head_shoulders,         windows_to_arrays
from src.labeling.inverse_head_shoulders import label_inverse_head_shoulders
from src.labeling.double_top             import label_double_top
from src.labeling.double_bottom          import label_double_bottom
from src.training.dataset  import time_split, make_dataloaders
from src.training.augment  import augment_positives
from src.training.trainer  import train, evaluate_test
from src.models.tcn        import build_model


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


parser = argparse.ArgumentParser(description="Train TCN on all 4 candlestick patterns.")
parser.add_argument("--config-override", default=None,
                    help="Path to a YAML override file merged on top of configs/config.yaml")
args = parser.parse_args()

# ── load config & data once ───────────────────────────────────────────────────
with open(os.path.join(ROOT, "configs/config.yaml"), encoding="utf-8") as f:
    BASE_CFG = yaml.safe_load(f)

if args.config_override:
    with open(args.config_override, encoding="utf-8") as f:
        override = yaml.safe_load(f)
    BASE_CFG = _deep_merge(BASE_CFG, override)
    print(f"Config override loaded: {args.config_override}")

device = torch.device(
    "cuda" if torch.cuda.is_available() and BASE_CFG["project"]["device"] != "cpu"
    else "cpu"
)
print(f"\n{'='*60}")
print(f"Device: {device}")
print(f"{'='*60}\n")

print("Loading data ...")
df_1min  = load_raw_ohlcv(BASE_CFG)
df_15min = resample_ohlcv(df_1min, BASE_CFG)
df_smooth, bw = smooth_ohlcv(df_15min, BASE_CFG)
extrema  = find_extrema(df_smooth, BASE_CFG)
print(f"Data ready | {len(df_15min):,} 15-min bars | bandwidth={bw:.1f}\n")

# ── pattern registry ──────────────────────────────────────────────────────────
PATTERNS = [
    ("head_shoulders",         label_head_shoulders),
    ("inverse_head_shoulders", label_inverse_head_shoulders),
    ("double_top",             label_double_top),
    ("double_bottom",          label_double_bottom),
]

PATTERN_LABELS = {
    "head_shoulders":         ("Other", "Head & Shoulders"),
    "inverse_head_shoulders": ("Other", "Inverse H&S"),
    "double_top":             ("Other", "Double Top"),
    "double_bottom":          ("Other", "Double Bottom"),
}

all_results = {}

for pat_name, label_fn in PATTERNS:
    print(f"\n{'='*60}")
    print(f"  Pattern: {pat_name.upper().replace('_',' ')}")
    print(f"{'='*60}")

    cfg = copy.deepcopy(BASE_CFG)
    cfg["labeling"]["active_pattern"] = pat_name

    # ── 1. Label ──────────────────────────────────────────────────────────────
    labeled = label_fn(df_smooth, extrema, cfg)
    labeled_s = sorted(labeled, key=lambda w: w.anchor_bar)

    pos = sum(w.label == 1 for w in labeled_s)
    neg = sum(w.label == 0 for w in labeled_s)
    print(f"Dataset: {len(labeled_s)} windows | {pos} positive | {neg} negative")

    if pos < 5:
        print(f"  [SKIP] Too few positive samples ({pos}) — skipping training.")
        continue

    # ── 2. Windows → arrays ───────────────────────────────────────────────────
    from src.labeling import head_shoulders as _hs_mod
    X, y = _hs_mod.windows_to_arrays(labeled_s, df_smooth, cfg)

    # ── 3. Time-based split ───────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = time_split(X, y, cfg)

    print(f"Split — train:{len(y_train)} val:{len(y_val)} test:{len(y_test)}")
    print(f"Train pos before aug: {(y_train==1).sum()}")

    # ── 4. Augment training set ───────────────────────────────────────────────
    if cfg.get("augmentation", {}).get("enabled", False):
        X_train, y_train = augment_positives(X_train, y_train, cfg)

    # ── 5. DataLoaders ────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = make_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, cfg
    )

    # ── 6. Dirs ───────────────────────────────────────────────────────────────
    ckpt_dir    = os.path.join(ROOT, "outputs", pat_name, "checkpoints")
    metrics_dir = os.path.join(ROOT, "outputs", pat_name, "metrics")
    os.makedirs(ckpt_dir,    exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    cfg["paths"]["checkpoints_dir"] = ckpt_dir
    cfg["paths"]["metrics_dir"]     = metrics_dir

    # ── 7. Build & train ──────────────────────────────────────────────────────
    model   = build_model(cfg, device)
    history = train(model, train_loader, val_loader, cfg, device)

    # ── 8. Save training curves ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train", color="steelblue")
    axes[0].plot(history["val_loss"],   label="Val",   color="orange")
    axes[0].set_title(f"{pat_name} — Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history["val_f1"], label="Val F1", color="green")
    axes[1].set_title(f"{pat_name} — Val F1"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "training_curves.png"), dpi=120)
    plt.close(fig)

    # ── 9. Test evaluation ────────────────────────────────────────────────────
    test_metrics = evaluate_test(model, test_loader, cfg, device)

    # Full classification report
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds = model(Xb.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    neg_name, pos_name = PATTERN_LABELS[pat_name]
    try:
        report = classification_report(
            all_labels, all_preds,
            target_names=[neg_name, pos_name],
            output_dict=True, zero_division=0,
        )
    except Exception:
        report = {}

    with open(os.path.join(metrics_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay(cm, display_labels=[neg_name, pos_name]).plot(ax=ax, colorbar=False)
    ax.set_title(f"{pat_name} — Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    all_results[pat_name] = {
        "dataset": {"total": len(labeled_s), "positive": pos, "negative": neg},
        "test":    test_metrics,
        "report":  report,
        "history_best_val_f1": history.get("best_val_f1", 0.0),
    }

    print(f"\n[{pat_name}] Test F1 (macro): {test_metrics['f1']:.4f} | "
          f"Accuracy: {test_metrics['accuracy']:.4f}")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print("  ALL PATTERNS SUMMARY")
print(f"{'='*60}")
print(f"{'Pattern':<28} {'F1':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'Pos':>5}")
print("-" * 60)
for pat, res in all_results.items():
    t = res["test"]
    print(f"{pat:<28} {t['f1']:>6.3f} {t['accuracy']:>6.3f} "
          f"{t['precision']:>6.3f} {t['recall']:>6.3f} {res['dataset']['positive']:>5}")

# Save summary
with open(os.path.join(ROOT, "outputs", "all_patterns_summary.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSummary saved to outputs/all_patterns_summary.json")
