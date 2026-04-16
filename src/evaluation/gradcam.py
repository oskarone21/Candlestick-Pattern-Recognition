"""
TCN 时序序列的梯度加权类激活映射（Grad-CAM）。

Grad-CAM 回答的问题："模型在预测双底形态时，关注的是哪些时间步？"

如果模型真正学到了形态，高注意力区域应与 E1-E3 形成窗口对齐，
而不是随机 K 线。

参考文献：Selvaraju et al. (2017) Grad-CAM。
此处为 1D 时序序列的改编版（原版用于 2D 图像）。

遵循 TEAM_STANDARDS.md：单一职责。
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt


def compute_gradcam_1d(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int = 1,
) -> np.ndarray:
    """计算单条序列的 Grad-CAM 注意力权重。

    通过 Hook 接入最后一个 TCN 残差块，捕获激活值和梯度，
    生成逐时间步的一维重要性图。

    参数
    ----
    model : nn.Module
        已训练的 TCN，必须处于 eval 模式。
    x : torch.Tensor, shape (1, seq_len, features)
        单个输入窗口（batch size = 1）。
    target_class : int
        要解释的目标类别索引（1 = 双底形态）。

    返回
    ----
    cam : np.ndarray, shape (seq_len,)
        每个时间步的重要性分数，已归一化到 [0, 1]。
    """
    model.eval()

    activations = {}  # 存储前向传播的激活值
    gradients   = {}  # 存储反向传播的梯度

    # 在最后一个 TCN 块（index -1）上注册 Hook
    last_block = model.tcn[-1]

    def forward_hook(module, inp, out):
        # 前向 Hook：捕获该层的输出激活值
        activations["feat"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        # 反向 Hook：捕获该层输出对应的梯度
        gradients["feat"] = grad_out[0].detach()

    fh = last_block.register_forward_hook(forward_hook)
    bh = last_block.register_full_backward_hook(backward_hook)

    # 前向传播
    x      = x.requires_grad_(True)
    logits = model(x)
    score  = logits[0, target_class]  # 目标类别的原始分数

    # 对目标类别分数反向传播，计算梯度
    model.zero_grad()
    score.backward()

    # 清除 Hook
    fh.remove()
    bh.remove()

    # Grad-CAM 计算：
    # 激活值形状：(1, C, T) → 压缩 batch 维为 (C, T)
    # 梯度形状  ：(1, C, T) → 压缩 batch 维为 (C, T)
    acts  = activations["feat"].squeeze(0)  # (C, T)
    grads = gradients["feat"].squeeze(0)    # (C, T)

    # 每个通道的权重 = 梯度在时间维度的全局平均池化
    weights = grads.mean(dim=-1)                         # (C,)
    # 加权求和：(C,1) * (C,T) → sum → (T,)
    cam     = (weights[:, None] * acts).sum(dim=0)       # (T,)
    cam     = torch.relu(cam).cpu().numpy()              # ReLU 去负值

    # 归一化到 [0, 1]，方便可视化
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam


def plot_gradcam_examples(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    labeled_windows: list,
    cfg: dict,
    device: torch.device,
    n_examples: int = 6,
    save_dir: str = "outputs/metrics",
) -> None:
    """绘制正样本的 Grad-CAM 注意力叠加在 OHLCV 序列上的可视化图。

    参数
    ----
    model : nn.Module
        已加载最优检查点的 TCN。
    X : np.ndarray, shape (N, T, 5)
        所有窗口（已归一化）。
    y : np.ndarray, shape (N,)
    labeled_windows : list[LabeledWindow]
        与 X、y 顺序相同的带标签窗口列表。
    cfg : dict
    device : torch.device
    n_examples : int
        要可视化的正样本数量。
    save_dir : str
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    model.eval()

    # 找出预测正确的正样本（真正例）
    correct_pos_idx = []
    for i, (xi, yi) in enumerate(zip(X, y)):
        if yi != 1:
            continue  # 跳过负样本
        x_t = torch.tensor(xi[None], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(x_t).argmax(dim=1).item()
        if pred == 1:
            correct_pos_idx.append(i)  # 预测正确的正样本

    n_plot = min(n_examples, len(correct_pos_idx))
    if n_plot == 0:
        print("[gradcam] No correctly predicted positive samples found.")
        return

    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 4 * n_plot))
    if n_plot == 1:
        axes = [axes]

    lookback = cfg["windowing"]["lookback_bars"]  # 窗口长度，如 80

    for ax, idx in zip(axes, correct_pos_idx[:n_plot]):
        xi = X[idx]       # (T, 5) 单个窗口的归一化 OHLCV
        lw = labeled_windows[idx]
        c  = lw.candidate  # 对应的双底候选形态

        # 计算该样本的 Grad-CAM 注意力图
        x_t = torch.tensor(xi[None], dtype=torch.float32).to(device)
        cam = compute_gradcam_1d(model, x_t, target_class=1)  # (T,)

        close_norm = xi[:, 3]    # 归一化收盘价（第 3 列）
        t = np.arange(lookback)

        # 背景：Grad-CAM 注意力热力图（红色柱）
        ax.bar(t, cam, color="red", alpha=0.25, width=1.0, label="Grad-CAM attention")

        # 前景：归一化收盘价折线
        ax.plot(t, close_norm, color="black", lw=1.5, label="Close (normalised)")

        # 标注 E1（第一波谷）、E2（波峰/颈线）、E3（第二波谷）相对窗口的位置
        window_start = lw.anchor_bar - lookback + 1
        for bar, price_col, color, marker, label in [
            (c.e1_bar, 0, "green",  "v", "E1 trough"),          # 第一波谷：绿色向下三角
            (c.e2_bar, 1, "orange", "^", "E2 peak (neckline)"),  # 颈线波峰：橙色向上三角
            (c.e3_bar, 2, "green",  "v", "E3 trough"),          # 第二波谷：绿色向下三角
        ]:
            rel = bar - window_start  # 转换为窗口内相对位置
            if 0 <= rel < lookback:
                ax.scatter([rel], [close_norm[rel]], color=color,
                           s=100, zorder=5, marker=marker, label=label)

        # 标注突破 K 线（窗口最后一根 K 线）
        rel_anchor = lookback - 1
        ax.axvline(rel_anchor, color="purple", lw=1.5, ls="--", label="Breakout bar")

        ax.set_title(f"Grad-CAM  |  anchor={lw.anchor_bar}  "
                     f"neckline={c.neckline:.0f}", fontsize=9)
        ax.set_xlabel("Bar (t=0 is window start, t=79 is breakout)")
        ax.set_ylabel("Normalised price / Attention")
        ax.legend(fontsize=7, loc="upper left")

    plt.suptitle(
        "Grad-CAM: which time steps does the TCN focus on?\n"
        "(Red bars = model attention, peak attention near E1-E3 = pattern learned)",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = os.path.join(save_dir, "gradcam_examples.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[gradcam] Saved: {out_path}")
