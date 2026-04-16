"""
PyTorch Dataset 及基于时间的训练/验证/测试划分工具。

必须使用时间顺序划分（配置项 evaluation.split_method: time），
以防止未来数据泄漏到训练集和验证集中。

遵循 TEAM_STANDARDS.md：配置驱动、无数据泄漏、单一职责。
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PatternDataset(Dataset):
    """OHLCV 形态窗口的 PyTorch Dataset 封装。

    参数
    ----
    X : np.ndarray, shape (N, T, 5)
        归一化后的 OHLCV 序列。
    y : np.ndarray, shape (N,)
        二分类标签（0 或 1）。
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        # 转换为 PyTorch 张量并缓存
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 返回单个样本（输入序列, 标签）
        return self.X[idx], self.y[idx]


def time_split(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按时间顺序（不打乱）将数组划分为训练集/验证集/测试集。

    样本必须已按锚点 K 线索引升序排列。
    按时间划分可防止未来数据泄漏到训练过程。

    参数
    ----
    X : np.ndarray, shape (N, T, 5)
    y : np.ndarray, shape (N,)
    cfg : dict
        完整配置，读取 evaluation.train_ratio 和 evaluation.val_ratio。

    返回
    ----
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    eval_cfg    = cfg["evaluation"]
    train_ratio = eval_cfg["train_ratio"]  # 训练集比例，如 0.70
    val_ratio   = eval_cfg["val_ratio"]    # 验证集比例，如 0.15

    n       = len(y)
    n_train = int(n * train_ratio)   # 训练集样本数
    n_val   = int(n * val_ratio)     # 验证集样本数

    # 按时间顺序切片（前70%→训练，中15%→验证，后15%→测试）
    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val   = X[n_train : n_train + n_val]
    y_val   = y[n_train : n_train + n_val]

    X_test  = X[n_train + n_val :]
    y_test  = y[n_train + n_val :]

    print(
        f"[dataset] Split (time-based) — "
        f"train: {len(y_train)} | val: {len(y_val)} | test: {len(y_test)}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def make_dataloaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    cfg: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """将划分好的数组封装为 PyTorch DataLoader。

    训练集 DataLoader 会在训练集内部打乱顺序（防止批次偏差）。
    验证集和测试集 DataLoader 不打乱（保证评估顺序稳定）。

    参数
    ----
    cfg : dict
        完整配置，读取 training.batch_size 和 project.seed。

    返回
    ----
    train_loader, val_loader, test_loader
    """
    batch_size = cfg["training"]["batch_size"]  # 每个梯度步骤的样本数
    seed       = cfg["project"]["seed"]          # 随机种子，保证可复现

    # 用固定种子创建随机数生成器，保证训练打乱顺序可复现
    generator = torch.Generator()
    generator.manual_seed(seed)

    # 训练集：打乱顺序（shuffle=True），提高训练多样性
    train_loader = DataLoader(
        PatternDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    # 验证集：不打乱，保证每次评估结果一致
    val_loader = DataLoader(
        PatternDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    # 测试集：不打乱，保证评估结果可复现
    test_loader = DataLoader(
        PatternDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader
