"""
用于二分类形态识别的时序卷积网络（TCN）。

网络结构概述
------------
输入  : (batch, seq_len, input_features)   例如 (B, 80, 5)
         → 转置为 (batch, features, seq_len) 供 Conv1d 使用

TCN   : 膨胀因果残差块堆叠。
         每个块的膨胀率翻倍，使感受野指数增长：第 k 层膨胀率 = 2^k。
         仅需 3 层即可覆盖完整的 80 根 K 线窗口。

         残差块结构（标准 TCN，Bai et al. 2018）：
           Conv1d（因果、膨胀）→ WeightNorm → ReLU → Dropout
           Conv1d（因果、膨胀）→ WeightNorm → ReLU → Dropout
           + 残差 1×1 卷积（当通道数发生变化时）

输出头 : 时间维度全局平均池化 → FC(num_channels[-1], num_classes)

所有超参数从 cfg['model']['tcn'] 和 cfg['model'] 读取。
遵循 TEAM_STANDARDS.md：配置驱动、无数据泄漏、单一职责。
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class _CausalConv1d(nn.Module):
    """因果膨胀 Conv1d：仅在左侧填充，确保时刻 t 只能看到 t-k..t-1。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        # 计算左侧需要的填充量，保证输出长度等于输入长度
        self.padding = (kernel_size - 1) * dilation
        # weight_norm：权重归一化，稳定训练（替代 BatchNorm，适合时序）
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=self.padding,  # 在两侧都填充，但右侧多余部分会被裁剪
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # 裁剪掉 nn.Conv1d 在右侧引入的多余填充，确保因果性
        return out[:, :, : -self.padding] if self.padding > 0 else out


class _ResidualBlock(nn.Module):
    """包含两个因果膨胀卷积的 TCN 残差块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # 第一个因果卷积：in_channels → out_channels
        self.conv1   = _CausalConv1d(in_channels,  out_channels, kernel_size, dilation)
        # 第二个因果卷积：out_channels → out_channels
        self.conv2   = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # Dropout 防止过拟合

        # 当输入输出通道数不同时，用 1×1 卷积对残差支路做投影
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差支路（通道数不同时做 1×1 投影）
        residual = x if self.downsample is None else self.downsample(x)

        # 主路：Conv → ReLU → Dropout → Conv → ReLU → Dropout
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)

        # 残差相加后再过一次 ReLU（标准残差连接）
        return self.relu(out + residual)


class TCN(nn.Module):
    """用于序列分类的时序卷积网络。

    参数
    ----
    cfg : dict
        从 configs/config.yaml 加载的完整配置。
        读取：model.tcn.input_features, model.tcn.num_channels,
              model.tcn.kernel_size, model.dropout, model.num_classes。
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        tcn_cfg     = cfg["model"]["tcn"]
        in_features = tcn_cfg["input_features"]  # 5（OHLCV 五个特征）
        channels    = tcn_cfg["num_channels"]     # [32, 64, 64]
        kernel_size = tcn_cfg["kernel_size"]      # 卷积核大小，如 3
        dropout     = cfg["model"]["dropout"]     # Dropout 比例，如 0.2
        num_classes = cfg["model"]["num_classes"] # 2（二分类）

        # 构建残差块序列，膨胀率指数增长（1, 2, 4, ...）
        layers = []
        in_ch  = in_features
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i  # 第 i 层膨胀率：1→2→4→...（感受野指数扩大）
            layers.append(
                _ResidualBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)

        # 分类头：将时序特征映射到类别分数
        self.head = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x : torch.Tensor, shape (batch, seq_len, input_features)

        返回
        ----
        logits : torch.Tensor, shape (batch, num_classes)
        """
        # Conv1d 需要 (batch, channels, seq_len) 格式，转置调整维度
        x = x.permute(0, 2, 1)   # (B, 5, 80)
        x = self.tcn(x)           # (B, 64, 80) — 经过所有残差块
        x = x[:, :, -1]           # 取最后时间步 → (B, 64)
        # 因果模型最后一步已编码全部历史；全局平均池化会消除时序信号
        return self.head(x)       # 线性分类头 → (B, 2)


def build_model(cfg: dict, device: torch.device) -> TCN:
    """实例化 TCN 模型并移动到目标设备。

    参数
    ----
    cfg : dict
        完整配置。
    device : torch.device

    返回
    ----
    TCN
        位于目标设备上的模型。
    """
    model    = TCN(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[tcn] Model built — {n_params:,} trainable parameters | device: {device}")
    return model
