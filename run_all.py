"""
run_all.py — 主控脚本：端到端运行完整流水线。

步骤：
  1. 加载 + 重采样数据
  2. 平滑价格 + 提取极值点
  3. 标注双底形态窗口
  4. 训练 TCN 序列模型
  5. 评估：混淆矩阵、逐类别报告、训练曲线
  6. Grad-CAM 注意力可视化
  7. 回测：突破买入模拟 vs 随机基准
  8. （可选）ResNet-18 图像基准对比
  9. 最终汇总对比图

用法：
  python run_all.py
  python run_all.py --image-baseline      # 同时运行 ResNet 图像模型
"""

import argparse
import os
import json
import yaml
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合无显示器环境
import matplotlib.pyplot as plt

# 导入各模块
from src.data.loader    import load_raw_ohlcv
from src.data.resampler import resample_ohlcv
from src.labeling.smoother      import smooth_ohlcv
from src.labeling.extrema       import find_extrema
from src.labeling.double_bottom import label_double_bottom, windows_to_arrays
from src.training.dataset  import time_split, make_dataloaders
from src.models.tcn        import build_model
from src.training.trainer  import train, evaluate_test
from src.evaluation.metrics  import full_evaluation
from src.evaluation.gradcam  import plot_gradcam_examples
from src.evaluation.backtest import run_backtest


# ---------------------------------------------------------------------------
def main(run_image_baseline: bool = False) -> None:
    """运行完整的训练和评估流水线。

    参数
    ----
    run_image_baseline : bool
        若为 True，额外训练 ResNet-18 图像基准模型进行对比。
    """
    # 加载配置文件（所有参数的唯一来源）
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 自动选择运行设备（优先 GPU，无 GPU 时使用 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Candlestick Pattern Recognition — Double Bottom")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # 步骤 1-3：数据流水线 + 形态标注                                    #
    # ------------------------------------------------------------------ #
    print(">>> Step 1-3: Data pipeline + labeling")
    df_1min      = load_raw_ohlcv(cfg)                    # 加载原始 1 分钟 K 线
    df_15min     = resample_ohlcv(df_1min, cfg)           # 重采样为 15 分钟 K 线
    df_smooth, h = smooth_ohlcv(df_15min, cfg)            # Nadaraya-Watson 平滑
    extrema      = find_extrema(df_smooth, cfg)           # 提取波峰/波谷极值点
    labeled      = label_double_bottom(df_smooth, extrema, cfg)  # 检测双底形态并标注
    labeled_sorted = sorted(labeled, key=lambda w: w.anchor_bar)  # 按时间排序（防止泄漏）

    # 转换为 numpy 数组（X: OHLCV 序列，y: 标签）
    X, y  = windows_to_arrays(labeled_sorted, df_smooth, cfg)
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    print(f"    Windows: {len(y)} total  ({n_pos} positive, {n_neg} negative)\n")

    # ------------------------------------------------------------------ #
    # 步骤 4：训练 TCN 模型                                              #
    # ------------------------------------------------------------------ #
    print(">>> Step 4: Train TCN")
    # 按时间顺序划分训练/验证/测试集
    X_tr, y_tr, X_val, y_val, X_te, y_te = time_split(X, y, cfg)
    # 创建 PyTorch DataLoader
    tr_l, val_l, te_l = make_dataloaders(X_tr, y_tr, X_val, y_val, X_te, y_te, cfg)

    tcn_model = build_model(cfg, device)          # 构建 TCN 模型
    history   = train(tcn_model, tr_l, val_l, cfg, device)  # 训练（含早停）

    # ------------------------------------------------------------------ #
    # 步骤 5：评估 TCN 模型                                              #
    # ------------------------------------------------------------------ #
    print("\n>>> Step 5: Evaluate TCN")
    # 加载训练过程中保存的最优检查点权重
    tcn_model.load_state_dict(
        torch.load(history["checkpoint"], map_location=device)
    )
    # 在测试集上做完整评估，生成混淆矩阵、训练曲线、分类报告
    tcn_report = full_evaluation(
        tcn_model, te_l, cfg, device, history,
        save_dir=cfg["paths"]["metrics_dir"],
    )

    # ------------------------------------------------------------------ #
    # 步骤 6：Grad-CAM 注意力可视化                                      #
    # ------------------------------------------------------------------ #
    print("\n>>> Step 6: Grad-CAM")
    # 计算训练集和验证集的样本数，用于定位测试集起始位置
    n_train = int(len(y) * cfg["evaluation"]["train_ratio"])
    n_val   = int(len(y) * cfg["evaluation"]["val_ratio"])
    # 仅对测试集样本生成 Grad-CAM 图
    plot_gradcam_examples(
        tcn_model,
        X[n_train + n_val:], y[n_train + n_val:],
        labeled_sorted[n_train + n_val:],
        cfg, device, n_examples=6,
        save_dir=cfg["paths"]["metrics_dir"],
    )

    # ------------------------------------------------------------------ #
    # 步骤 7：回测模拟                                                    #
    # ------------------------------------------------------------------ #
    print("\n>>> Step 7: Backtest")
    # 模拟不同持仓周期（8/16/32 根 K 线 ≈ 2h/4h/8h）的交易表现
    bt_results = run_backtest(
        labeled_sorted, df_15min, cfg,
        hold_bars_list=[8, 16, 32],
        save_dir=cfg["paths"]["metrics_dir"],
    )

    # ------------------------------------------------------------------ #
    # 步骤 8：（可选）ResNet-18 图像基准                                 #
    # ------------------------------------------------------------------ #
    resnet_report = None
    if run_image_baseline:
        print("\n>>> Step 8: ResNet-18 image baseline")
        from src.models.resnet_baseline import (
            build_image_dataset, ImagePatternDataset,
            build_resnet, train_resnet, evaluate_resnet,
        )
        from torch.utils.data import DataLoader as DL

        # 将每个 OHLCV 窗口渲染为 224×224 K 线图像
        images, img_labels = build_image_dataset(
            labeled_sorted, df_15min, cfg,
            cache_dir="outputs/images",
        )
        n_tr = int(len(img_labels) * cfg["evaluation"]["train_ratio"])
        n_v  = int(len(img_labels) * cfg["evaluation"]["val_ratio"])

        # 创建图像数据集（按时间顺序划分）
        tr_ds  = ImagePatternDataset(images[:n_tr],         img_labels[:n_tr])
        val_ds = ImagePatternDataset(images[n_tr:n_tr+n_v], img_labels[n_tr:n_tr+n_v])
        te_ds  = ImagePatternDataset(images[n_tr+n_v:],     img_labels[n_tr+n_v:])

        bs = cfg["training"]["batch_size"]
        rn_tr_l  = DL(tr_ds,  batch_size=bs, shuffle=True)
        rn_val_l = DL(val_ds, batch_size=bs, shuffle=False)
        rn_te_l  = DL(te_ds,  batch_size=bs, shuffle=False)

        rn_model      = build_resnet(cfg, device)                         # 构建 ResNet-18
        rn_hist       = train_resnet(rn_model, rn_tr_l, rn_val_l, cfg, device)
        resnet_report = evaluate_resnet(rn_model, rn_te_l, cfg, device)

    # ------------------------------------------------------------------ #
    # 步骤 9：最终汇总对比图                                             #
    # ------------------------------------------------------------------ #
    print("\n>>> Step 9: Final summary chart")
    _plot_final_summary(tcn_report, resnet_report, bt_results, history,
                        save_dir=cfg["paths"]["metrics_dir"])

    print(f"\n{'='*60}")
    print("  All outputs saved to:", cfg["paths"]["metrics_dir"])
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------

def _plot_final_summary(tcn_report, resnet_report, bt_results, history,
                         save_dir: str = "outputs/metrics") -> None:
    """生成一页式汇总图：模型指标 + 回测结果 + 训练曲线。"""

    has_resnet = resnet_report is not None
    n_cols = 3  # 固定三列子图
    fig, axes = plt.subplots(1, n_cols, figsize=(18, 5))

    # --- 子图 1：模型 F1 对比柱状图 ---
    ax = axes[0]
    models_names, f1_vals, prec_vals, rec_vals = [], [], [], []

    for name, report in [("TCN (sequence)", tcn_report),
                          ("ResNet-18 (image)", resnet_report)]:
        if report is None:
            continue  # ResNet 未启用则跳过
        db = report.get("Double Bottom", {})
        models_names.append(name)
        f1_vals.append(db.get("f1-score", 0))
        prec_vals.append(db.get("precision", 0))
        rec_vals.append(db.get("recall", 0))

    x = np.arange(len(models_names))
    w = 0.25  # 每组柱的宽度
    ax.bar(x - w, prec_vals, w, label="Precision", color="steelblue")
    ax.bar(x,     rec_vals,  w, label="Recall",    color="orange")
    ax.bar(x + w, f1_vals,   w, label="F1",        color="green")
    ax.set_xticks(x); ax.set_xticklabels(models_names, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Double Bottom — Test Metrics\n(Precision / Recall / F1)")
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- 子图 2：TCN 训练曲线 ---
    ax = axes[1]
    ax.plot(history["val_f1"], color="green", lw=2, label="Val F1 (macro)")
    ax.plot(history["train_loss"], color="steelblue", lw=1.5,
            ls="--", label="Train Loss")
    best_ep = int(np.argmax(history["val_f1"]))
    ax.axvline(best_ep, color="red", ls=":", lw=1.5,
               label=f"Best ep {best_ep+1}")  # 标注最优 epoch
    ax.set_title("TCN Training Curve")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- 子图 3：回测胜率对比柱状图 ---
    ax = axes[2]
    db_rows   = bt_results[bt_results["strategy"] == "Double Bottom signal"]
    rand_rows = bt_results[bt_results["strategy"] == "Random baseline"]

    # 格式化 X 轴标签：显示 K 线数和对应小时数
    hold_labels = [f"{int(h)}b\n({h*15//60}h)" for h in db_rows["hold_bars"]]
    x2 = np.arange(len(hold_labels))
    w2 = 0.35
    ax.bar(x2 - w2/2, db_rows["win_rate_%"].values,   w2,
           label="Double Bottom signal", color="green", alpha=0.8)
    ax.bar(x2 + w2/2, rand_rows["win_rate_%"].values, w2,
           label="Random baseline",      color="gray",  alpha=0.6)
    ax.axhline(50, color="black", lw=1, ls="--", label="50% line")  # 50% 基准线
    ax.set_xticks(x2); ax.set_xticklabels(hold_labels)
    ax.set_ylim(0, 100)
    ax.set_title("Backtest: Win Rate by Holding Period")
    ax.set_ylabel("Win Rate (%)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Double Bottom Pattern Recognition — Summary", fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(save_dir, "final_summary.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[summary] Saved: {out_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-baseline", action="store_true",
                        help="Also train ResNet-18 image baseline")
    args = parser.parse_args()
    main(run_image_baseline=args.image_baseline)
