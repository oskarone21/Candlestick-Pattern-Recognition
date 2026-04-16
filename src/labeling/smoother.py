"""
Nadaraya-Watson 核平滑器 + AICc 带宽自动选择。

在极值点提取之前对收盘价序列进行平滑处理。
原始 15 分钟价格含有大量微观结构噪声，会产生数百个无意义的微小峰谷。
平滑后只保留大型、结构上显著的转折点，用于形态识别。

数学参考：PATTERNS_EXPLAINED.md §1
学术来源：Nadaraya (1964)，Hurvich、Simonoff、Tsai (1998)。

所有行为由配置字典驱动，此处不硬编码任何值。
遵循 TEAM_STANDARDS.md：配置驱动、无数据泄漏、单一职责。
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter, lfilter_zi


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------

def smooth_ohlcv(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """向 OHLCV DataFrame 添加平滑后的收盘价列。

    先在代表性样本上用 AICc 选出最优高斯带宽，
    再将平滑器应用到完整收盘价序列，
    结果存入新列 'close_smooth'。

    参数
    ----
    df : pd.DataFrame
        来自 resampler.resample_ohlcv() 的 15 分钟 OHLCV DataFrame。
    cfg : dict
        从 configs/config.yaml 加载的完整配置。

    返回
    ----
    pd.DataFrame
        增加了 'close_smooth' 列的 DataFrame。
    float
        AICc 选出的最优带宽 h。
    """
    smoothing_cfg = cfg["labeling"]["extrema_detection"]["smoothing"]
    bw_cfg        = smoothing_cfg["bandwidth"]
    price_col     = cfg["labeling"]["extrema_detection"]["price_field"]  # "close"
    col_close     = cfg["data_source"]["columns"][price_col]             # "Close"

    prices = df[col_close].to_numpy(dtype=float)

    # 从价格序列中间截取代表性样本，用于带宽选择（节省运行时间）
    sample_size = cfg["labeling"]["extrema_detection"]["smoothing"]["bandwidth"].get(
        "lookback_bars", 120
    )
    sample = _draw_representative_sample(prices, sample_size)

    # 在候选带宽网格上搜索 AICc 最小值
    h = select_bandwidth_aicc(
        sample,
        min_h=bw_cfg["min_bandwidth"],
        max_h=bw_cfg["max_bandwidth"],
    )
    print(f"[smoother] AICc-selected bandwidth h = {h:.2f} bars")

    # 用选出的带宽对完整序列做平滑
    smoothed = smooth_prices(prices, h)

    result = df.copy()
    result["close_smooth"] = smoothed  # 新增平滑价格列
    return result, h


# ---------------------------------------------------------------------------
# 带宽选择
# ---------------------------------------------------------------------------

def select_bandwidth_aicc(
    prices: np.ndarray,
    min_h: float,
    max_h: float,
    n_grid: int = 30,
) -> float:
    """在给定价格样本上找到令 AICc 最小的带宽。

    在 min_h 到 max_h 之间的等间距网格上搜索。

    参数
    ----
    prices : np.ndarray
        一维收盘价数组（小样本，≤ 几百根 K 线）。
    min_h : float
        最小候选带宽（单位：K 线数）。
    max_h : float
        最大候选带宽（单位：K 线数）。
    n_grid : int
        候选带宽的网格点数。

    返回
    ----
    float
        使 AICc 分数最低的带宽 h。
    """
    candidates = np.linspace(min_h, max_h, n_grid)          # 等间距候选带宽
    scores     = [_aicc(prices, h) for h in candidates]     # 计算每个候选带宽的 AICc
    best_h     = candidates[int(np.argmin(scores))]          # 选出得分最低的
    return float(best_h)


# ---------------------------------------------------------------------------
# 核心平滑器
# ---------------------------------------------------------------------------

def smooth_prices(prices: np.ndarray, h: float) -> np.ndarray:
    """单侧因果高斯平滑（Causal Nadaraya-Watson）。

    修复说明
    --------
    原始实现使用 gaussian_filter1d（双向对称核），每个时间步的平滑值
    会用到约 ±18 根未来 K 线，存在前瞻偏差（lookahead bias）。

    修复后使用单侧 FIR 滤波器（scipy.signal.lfilter），时间步 t 的
    平滑值仅依赖 t 及 t 之前的数据，完全消除前瞻偏差。

    Fix note
    --------
    The original implementation used gaussian_filter1d (two-sided symmetric
    kernel), which incorporated ~18 future bars per timestep (lookahead bias).
    The fix uses a one-sided causal FIR filter: the smoothed value at bar t
    depends only on bar t and earlier bars -- zero lookahead.

    参数 / Parameters
    ----
    prices : np.ndarray
        原始收盘价一维数组 / Raw 1D close price array.
    h : float
        高斯带宽（K线数）/ Gaussian bandwidth in bars.

    返回 / Returns
    ----
    np.ndarray
        与输入等长的平滑价格序列 / Smoothed prices, same length as input.
    """
    radius = max(1, int(4 * h))                          # 核半径 ≈ 4σ
    k = np.arange(radius + 1, dtype=float)               # [0, 1, 2, ..., radius]
    kernel = np.exp(-0.5 * (k / h) ** 2)                 # 单侧高斯权重
    kernel = kernel / kernel.sum()                        # 归一化

    # lfilter_zi: 用第一个价格值初始化滤波器状态，消除启动瞬态
    zi = lfilter_zi(kernel, [1.0]) * prices[0]
    smoothed, _ = lfilter(kernel, [1.0], prices.astype(float), zi=zi)
    return smoothed


# ---------------------------------------------------------------------------
# AICc 准则（Hurvich、Simonoff、Tsai 1998）
# ---------------------------------------------------------------------------

def _nadaraya_watson_exact(prices: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """精确 NW 平滑器及帽矩阵对角线（仅用于小窗口的带宽选择）。

    构建完整的 (n × n) 核矩阵，仅适用于小数组（n ≤ ~300）。

    参数
    ----
    prices : np.ndarray
        长度为 n 的一维价格数组。
    h : float
        带宽（K 线数）。

    返回
    ----
    smoothed : np.ndarray
        NW 平滑后的值。
    H_diag : np.ndarray
        帽矩阵 H 的对角线（各点的自影响值）。
    """
    n = len(prices)
    t = np.arange(n, dtype=float)

    # 未归一化的高斯核矩阵  K[i,j] = exp(-0.5 * ((i-j)/h)^2)
    # NW 公式中 1/h 因子在分子分母相消，此处省略
    diff = (t[:, None] - t[None, :]) / h        # (n, n) 各点间距离
    K    = np.exp(-0.5 * diff ** 2)             # (n, n) 高斯核矩阵

    row_sums = K.sum(axis=1)                    # (n,) 每行权重之和
    smoothed = (K @ prices) / row_sums          # NW 估计值

    # 帽矩阵对角线：H_ii = K[i,i] / sum_j K[i,j]
    H_diag = K.diagonal() / row_sums            # (n,)

    return smoothed, H_diag


def _aicc(prices: np.ndarray, h: float) -> float:
    """计算给定带宽在小价格样本上的 AICc 分数。

    AICc(h) = log(σ²) + (1 + tr(H)/n) / (1 - (tr(H) + 2)/n)

    其中 σ² = RSS/n，tr(H) = 帽矩阵对角线之和（有效自由度）。

    参数
    ----
    prices : np.ndarray
        一维价格样本。
    h : float
        带宽（K 线数）。

    返回
    ----
    float
        AICc 分数（越低越好）。
    """
    n = len(prices)
    smoothed, H_diag = _nadaraya_watson_exact(prices, h)

    rss     = np.sum((prices - smoothed) ** 2)  # 残差平方和
    sigma2  = rss / n                            # 估计误差方差
    trace_H = H_diag.sum()                       # 有效自由度

    # 分母接近零时 AICc 趋于无穷，说明带宽过小（过拟合）
    denominator = 1.0 - (trace_H + 2.0) / n
    if denominator <= 0.0:
        return np.inf

    return np.log(max(sigma2, 1e-10)) + (1.0 + trace_H / n) / denominator


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _draw_representative_sample(prices: np.ndarray, size: int) -> np.ndarray:
    """从价格序列中间截取连续片段作为代表性样本。

    使用中间段可以避免开盘/收盘时段波动较小的问题，
    给带宽选择器提供更有代表性的波动率样本。

    参数
    ----
    prices : np.ndarray
        完整价格序列。
    size : int
        要截取的 K 线数量。

    返回
    ----
    np.ndarray
        长度为 min(size, len(prices)) 的连续切片。
    """
    size = min(size, len(prices))
    mid  = len(prices) // 2
    lo   = max(0, mid - size // 2)
    return prices[lo: lo + size]
