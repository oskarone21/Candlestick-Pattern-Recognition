"""
双底形态确认信号的买入/卖出回测模拟。

在每个确认突破的 K 线收盘处模拟做多入场，
衡量不同持仓周期内的收益。

将形态触发的交易与随机入场基准对比，
验证形态是否具有统计上显著的预测能力。

遵循 TEAM_STANDARDS.md：配置驱动、单一职责。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt


def run_backtest(
    labeled_windows: list,
    df_ohlcv: pd.DataFrame,
    cfg: dict,
    hold_bars_list: list[int] | None = None,
    save_dir: str = "outputs/metrics",
) -> pd.DataFrame:
    """模拟突破买入交易并计算绩效指标。

    对每个确认的双底形态（label=1），在突破 K 线的收盘价买入，
    持仓 `hold_bars` 根 15 分钟 K 线后离场。

    同时运行随机入场基准（相同交易笔数、相同持仓周期、随机入场点），
    用于对比衡量形态的额外价值。

    参数
    ----
    labeled_windows : list[LabeledWindow]
        按时间排序的带标签窗口（label_double_bottom 的输出）。
    df_ohlcv : pd.DataFrame
        包含原始（未归一化）收盘价的完整 OHLCV DataFrame。
    cfg : dict
        完整配置。
    hold_bars_list : list[int], optional
        要测试的持仓周期（单位：K 线数），默认 [8, 16, 32]（约 2h/4h/8h）。
    save_dir : str

    返回
    ----
    pd.DataFrame
        每个持仓周期的绩效汇总表。
    """
    os.makedirs(save_dir, exist_ok=True)

    if hold_bars_list is None:
        hold_bars_list = [8, 16, 32]   # 15 分钟 K 线：约 2h/4h/8h

    col_close = cfg["data_source"]["columns"]["close"]
    close     = df_ohlcv[col_close].to_numpy(dtype=float)  # 原始收盘价数组
    n_bars    = len(close)

    seed = cfg["project"]["seed"]
    rng  = np.random.default_rng(seed)  # 固定随机种子，保证基准可复现

    # 提取所有正样本（确认双底）的锚点 K 线索引
    pos_anchors = [w.anchor_bar for w in labeled_windows if w.label == 1]

    results = []

    for hold in hold_bars_list:
        # ---- 双底信号交易 ----
        pat_returns = []
        for anchor in pos_anchors:
            exit_bar = anchor + hold          # 持仓结束的 K 线索引
            if exit_bar >= n_bars:
                continue                      # 超出数据范围，跳过
            entry_price = close[anchor]       # 入场价（突破 K 线收盘价）
            exit_price  = close[exit_bar]     # 离场价
            ret = (exit_price - entry_price) / entry_price  # 简单收益率
            pat_returns.append(ret)

        # ---- 随机基准交易（相同笔数、相同持仓周期、随机入场）----
        n_trades     = len(pat_returns)
        valid_starts = np.arange(0, n_bars - hold)          # 所有合法入场点
        rand_anchors = rng.choice(valid_starts, size=n_trades, replace=False)
        rand_returns = []
        for anchor in rand_anchors:
            entry_price = close[anchor]
            exit_price  = close[anchor + hold]
            ret = (exit_price - entry_price) / entry_price
            rand_returns.append(ret)

        pat_arr  = np.array(pat_returns)
        rand_arr = np.array(rand_returns)

        def metrics(arr, label):
            """计算一组交易收益的统计指标。"""
            if len(arr) == 0:
                return {}
            return {
                "strategy":      label,
                "hold_bars":     hold,
                "hold_hours":    hold * 15 / 60,                      # 换算为小时
                "n_trades":      len(arr),                             # 总交易笔数
                "win_rate_%":    round(100 * (arr > 0).mean(), 1),    # 胜率（收益>0）
                "avg_return_%":  round(100 * arr.mean(), 3),          # 平均收益率
                "median_ret_%":  round(100 * np.median(arr), 3),      # 中位数收益率
                "total_ret_%":   round(100 * arr.sum(), 2),           # 总收益（未复利）
                "best_trade_%":  round(100 * arr.max(), 3),           # 最佳单笔收益
                "worst_trade_%": round(100 * arr.min(), 3),           # 最差单笔收益
                "std_%":         round(100 * arr.std(), 3),           # 收益标准差
            }

        results.append(metrics(pat_arr,  "Double Bottom signal"))
        results.append(metrics(rand_arr, "Random baseline"))

    df_results = pd.DataFrame(results)

    # ---- 打印汇总表 ----
    print("\n[backtest] === Trading Simulation Results ===")
    print(df_results.to_string(index=False))

    # ---- 保存 CSV ----
    csv_path = os.path.join(save_dir, "backtest_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[backtest] Saved: {csv_path}")

    # ---- 绘制收益分布对比图 ----
    _plot_backtest(df_results, pos_anchors, close, hold_bars_list, save_dir)

    return df_results


def _plot_backtest(
    df_results: pd.DataFrame,
    pos_anchors: list[int],
    close: np.ndarray,
    hold_bars_list: list[int],
    save_dir: str,
) -> None:
    """保存回测对比图（不同持仓周期的收益分布直方图）。"""

    fig, axes = plt.subplots(1, len(hold_bars_list), figsize=(6 * len(hold_bars_list), 5))
    if len(hold_bars_list) == 1:
        axes = [axes]

    n_bars = len(close)

    for ax, hold in zip(axes, hold_bars_list):
        pat_returns, rand_returns = [], []
        rng = np.random.default_rng(42)  # 固定种子保证图表可复现

        # 计算双底信号的收益列表
        for anchor in pos_anchors:
            exit_bar = anchor + hold
            if exit_bar >= n_bars:
                continue
            pat_returns.append(100 * (close[exit_bar] - close[anchor]) / close[anchor])

        # 计算随机基准的收益列表
        n_trades     = len(pat_returns)
        valid_starts = np.arange(0, n_bars - hold)
        rand_anchors = rng.choice(valid_starts, size=n_trades, replace=False)
        for anchor in rand_anchors:
            rand_returns.append(100 * (close[anchor + hold] - close[anchor]) / close[anchor])

        # 确定直方图的 bin 范围（覆盖双方数据）
        bins = np.linspace(
            min(min(pat_returns, default=0), min(rand_returns, default=0)) - 0.5,
            max(max(pat_returns, default=0), max(rand_returns, default=0)) + 0.5,
            25,
        )
        ax.hist(rand_returns, bins=bins, alpha=0.5, color="gray",  label="Random baseline")
        ax.hist(pat_returns,  bins=bins, alpha=0.7, color="green", label="Double Bottom signal")
        ax.axvline(0, color="black", lw=1, ls="--")  # 零收益线
        # 标注双方平均收益的竖线
        ax.axvline(np.mean(pat_returns)  if pat_returns  else 0, color="green", lw=2,
                   ls="-", label=f"DB mean {np.mean(pat_returns):.2f}%" if pat_returns else "")
        ax.axvline(np.mean(rand_returns) if rand_returns else 0, color="gray",  lw=2,
                   ls="-", label=f"Rand mean {np.mean(rand_returns):.2f}%")
        ax.set_title(f"Hold = {hold} bars ({hold*15//60}h {hold*15%60}m)")
        ax.set_xlabel("Return (%)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.suptitle("Double Bottom Trade Returns vs Random Baseline", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(save_dir, "backtest_returns.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[backtest] Saved: {out_path}")
