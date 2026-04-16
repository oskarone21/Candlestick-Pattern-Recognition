"""
训练循环：支持早停、混合精度（AMP）和指标记录。

所有超参数从 cfg 读取，此处不硬编码任何值。
最优检查点（按验证集 F1 保存）存放在 cfg['paths']['checkpoints_dir']。

遵循 TEAM_STANDARDS.md：配置驱动、无数据泄漏、单一职责。
"""

import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> dict:
    """运行完整的训练循环。

    参数
    ----
    model : nn.Module
        已移动到 `device` 上的 TCN（或任意 nn.Module）。
    train_loader : DataLoader
    val_loader : DataLoader
    cfg : dict
        从 configs/config.yaml 加载的完整配置。
    device : torch.device

    返回
    ----
    dict
        历史记录字典，包含键 'train_loss'、'val_loss'、'val_f1'
        以及最优检查点的路径 'checkpoint'。
    """
    train_cfg = cfg["training"]
    opt_cfg   = cfg["optimizer"]
    paths_cfg = cfg["paths"]
    seed      = cfg["project"]["seed"]

    torch.manual_seed(seed)  # 固定随机种子，保证可复现

    epochs   = train_cfg["epochs"]                     # 最大训练轮数
    patience = train_cfg["early_stopping_patience"]    # 早停忍耐轮数
    # 仅在 GPU 上启用混合精度（AMP），CPU 不支持
    use_amp  = train_cfg["mixed_precision"] and device.type == "cuda"

    # 构建优化器
    optimizer = _build_optimizer(model, opt_cfg)

    train_labels = train_loader.dataset.y.cpu().numpy()
    n_pos = int((train_labels == 1).sum())
    n_neg = int((train_labels == 0).sum())

    # 类别权重：从 config 读取；当使用 "auto" 时按训练集类别比例计算
    w_neg = _resolve_class_weight(train_cfg.get("class_weight_negative", 1.0), n_neg, n_pos)
    w_pos = _resolve_class_weight(train_cfg.get("class_weight_positive", 6.0), n_pos, n_neg)
    pos_weight = torch.tensor([w_neg, w_pos], dtype=torch.float32).to(device)
    criterion  = nn.CrossEntropyLoss(weight=pos_weight)

    # AMP 梯度缩放器（仅 GPU 使用）
    scaler = GradScaler() if use_amp else None

    os.makedirs(paths_cfg["checkpoints_dir"], exist_ok=True)
    ckpt_path = os.path.join(paths_cfg["checkpoints_dir"], "best_model.pt")

    # 初始化历史记录
    history      = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_f1  = -1.0
    patience_ctr = 0

    print(f"\n[trainer] Starting training — {epochs} epochs | "
          f"AMP: {use_amp} | patience: {patience}")
    print(f"[trainer] Train samples — pos: {n_pos} | neg: {n_neg} | weights: [{w_neg:.3f}, {w_pos:.3f}]")
    print(f"{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>8}  "
          f"{'ValAcc':>7}  {'ValPrec':>8}  {'ValRec':>7}  {'ValF1':>6}")
    print("-" * 62)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # 训练一个 epoch，返回平均训练损失
        train_loss  = _train_epoch(model, train_loader, optimizer, criterion,
                                   scaler, device, use_amp)
        # 在验证集上推理，返回各项指标
        val_metrics = _evaluate(model, val_loader, criterion, device)

        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])

        print(
            f"{epoch:>6}  {train_loss:>10.4f}  {val_metrics['loss']:>8.4f}  "
            f"{val_metrics['accuracy']:>7.3f}  {val_metrics['precision']:>8.3f}  "
            f"{val_metrics['recall']:>7.3f}  {val_metrics['f1']:>6.3f}"
            f"  ({time.time()-t0:.1f}s)"
        )

        # 保存最优检查点（按验证集 F1）
        if val_metrics["f1"] > best_val_f1:
            best_val_f1  = val_metrics["f1"]
            torch.save(model.state_dict(), ckpt_path)  # 保存模型权重
            patience_ctr = 0  # 重置早停计数器
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                # 连续 patience 轮验证 F1 未提升，触发早停
                print(f"\n[trainer] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    history["best_val_f1"] = best_val_f1
    history["checkpoint"]  = ckpt_path
    print(f"\n[trainer] Best val F1: {best_val_f1:.4f} — checkpoint: {ckpt_path}")
    return history


# ---------------------------------------------------------------------------
# 测试集评估
# ---------------------------------------------------------------------------

def evaluate_test(
    model: nn.Module,
    test_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    checkpoint_path: str | None = None,
) -> dict:
    """加载最优检查点并在测试集上评估。

    参数
    ----
    model : nn.Module
    test_loader : DataLoader
    cfg : dict
    device : torch.device
    checkpoint_path : str, optional
        若为 None，则使用 cfg['paths']['checkpoints_dir']/best_model.pt。

    返回
    ----
    dict
        测试集指标：accuracy, precision, recall, f1。
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            cfg["paths"]["checkpoints_dir"], "best_model.pt"
        )
    # 加载训练时保存的最优权重
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_labels = test_loader.dataset.y.cpu().numpy()
    n_pos = int((test_labels == 1).sum())
    n_neg = int((test_labels == 0).sum())
    w_neg = _resolve_class_weight(cfg["training"].get("class_weight_negative", 1.0), n_neg, n_pos)
    w_pos = _resolve_class_weight(cfg["training"].get("class_weight_positive", 6.0), n_pos, n_neg)
    pos_weight = torch.tensor([w_neg, w_pos], dtype=torch.float32).to(device)
    criterion  = nn.CrossEntropyLoss(weight=pos_weight)
    metrics    = _evaluate(model, test_loader, criterion, device)

    # 保存测试集指标到 JSON 文件
    os.makedirs(cfg["paths"]["metrics_dir"], exist_ok=True)
    metrics_path = os.path.join(cfg["paths"]["metrics_dir"], "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[trainer] === Test Set Results ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:12s}: {v:.4f}")
    print(f"  Saved to: {metrics_path}")
    return metrics


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    """运行一个训练 epoch，返回平均损失。"""
    model.train()        # 切换到训练模式（启用 Dropout 等）
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()  # 清除上一步的梯度

        if use_amp:
            # 混合精度前向传播（自动使用 float16 计算）
            with autocast():
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
            # 缩放梯度并更新（防止 float16 梯度下溢）
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 普通精度前向传播
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()        # 反向传播计算梯度
            optimizer.step()       # 更新模型参数

        total_loss += loss.item() * len(y_batch)  # 累积加权损失

    return total_loss / len(loader.dataset)  # 返回每样本平均损失


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """在 DataLoader 上做推理，返回损失和分类指标。"""
    model.eval()   # 切换到评估模式（禁用 Dropout 等）
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():  # 不计算梯度，节省内存和时间
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits  = model(X_batch)
            loss    = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1).cpu().numpy()  # 取概率最大类别
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

    if len(loader.dataset) == 0:
        return {"loss": 0.0, "accuracy": 0.0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0}

    avg_loss = total_loss / len(loader.dataset)
    return {
        "loss":      avg_loss,
        "accuracy":  float(accuracy_score(all_labels, all_preds)),
        # macro 平均：各类别指标算术平均，不考虑类别样本量（适合不平衡数据）
        "precision": float(precision_score(all_labels, all_preds,
                                           average="macro", zero_division=0)),
        "recall":    float(recall_score(all_labels, all_preds,
                                        average="macro", zero_division=0)),
        "f1":        float(f1_score(all_labels, all_preds,
                                    average="macro", zero_division=0)),
    }


def _build_optimizer(model: nn.Module, opt_cfg: dict) -> torch.optim.Optimizer:
    """根据配置实例化优化器。"""
    name = opt_cfg["name"].lower()
    lr   = opt_cfg["lr"]           # 学习率
    wd   = opt_cfg["weight_decay"] # L2 正则化强度
    if name == "adamw":
        # AdamW：Adam + 解耦权重衰减，防止过拟合
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: '{name}'. Supported: adamw, adam.")


def _resolve_class_weight(value, this_count: int, other_count: int) -> float:
    """Resolve a numeric or 'auto' class weight."""
    if isinstance(value, str):
        if value.lower() != "auto":
            raise ValueError(f"Unsupported class weight value: {value}")
        if this_count <= 0:
            return 1.0
        return float(other_count / this_count)
    return float(value)
