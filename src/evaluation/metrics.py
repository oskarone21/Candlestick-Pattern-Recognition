"""
评估工具：混淆矩阵、逐类别报告、训练曲线图。

遵循 TEAM_STANDARDS.md：配置驱动、单一职责。
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")   # 非交互式后端，避免在无显示器的服务器上报错
import matplotlib.pyplot as plt


def full_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    history: dict,
    save_dir: str = "outputs/metrics",
) -> dict:
    """运行完整评估流程并保存所有图表和 JSON 报告。

    生成文件：
      - confusion_matrix.png     混淆矩阵热图
      - training_curves.png      训练损失和验证 F1 曲线
      - classification_report.json  逐类别精确率/召回率/F1

    参数
    ----
    model : nn.Module
        已加载最优检查点的训练模型。
    test_loader : DataLoader
    cfg : dict
    device : torch.device
    history : dict
        trainer.train() 的返回值，包含 train_loss、val_loss、val_f1。
    save_dir : str
        输出文件保存目录。

    返回
    ----
    dict
        完整的分类报告字典。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 收集测试集的预测结果
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.to(device))
            # softmax 转为概率，取正类（index=1）的概率
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()  # 取概率最大的类别
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # ------------------------------------------------------------------ #
    # 1. 混淆矩阵                                                        #
    # ------------------------------------------------------------------ #
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative (0)", "Double Bottom (1)"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"[eval] Saved: {cm_path}")

    # ------------------------------------------------------------------ #
    # 2. 训练曲线图                                                      #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：训练损失 vs 验证损失
    axes[0].plot(history["train_loss"], label="Train Loss", color="steelblue")
    axes[0].plot(history["val_loss"],   label="Val Loss",   color="orange")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 右图：验证集 F1，并标注最优 epoch
    axes[1].plot(history["val_f1"], label="Val F1 (macro)", color="green")
    best_ep = int(np.argmax(history["val_f1"]))
    axes[1].axvline(best_ep, color="red", ls="--",
                    label=f"Best epoch {best_ep+1}  F1={history['val_f1'][best_ep]:.3f}")
    axes[1].set_title("Validation F1 per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 (macro)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("TCN Training — Double Bottom Pattern")
    plt.tight_layout()
    curve_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(curve_path, dpi=120)
    plt.close()
    print(f"[eval] Saved: {curve_path}")

    # ------------------------------------------------------------------ #
    # 3. 分类报告（逐类别精确率/召回率/F1）                             #
    # ------------------------------------------------------------------ #
    report = classification_report(
        y_true, y_pred,
        target_names=["Negative", "Double Bottom"],
        output_dict=True,   # 返回字典，方便保存为 JSON
        zero_division=0,
    )
    report_path = os.path.join(save_dir, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[eval] Saved: {report_path}")

    # 在控制台打印可读格式
    print("\n[eval] === Per-Class Report ===")
    print(classification_report(
        y_true, y_pred,
        target_names=["Negative", "Double Bottom"],
        zero_division=0,
    ))

    return report
