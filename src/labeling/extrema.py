"""
极值点提取器：从平滑后的价格序列中找出局部波峰和波谷。

基于 PATTERNS_EXPLAINED.md §1.3 中的一阶和二阶导数条件：

    局部极大值（波峰）：在第 τ 根 K 线处  m'(τ) ≈ 0  且  m''(τ) < 0
    局部极小值（波谷）：在第 τ 根 K 线处  m'(τ) ≈ 0  且  m''(τ) > 0

所有行为由配置字典驱动，此处不硬编码任何值。
遵循 TEAM_STANDARDS.md：配置驱动、无数据泄漏、单一职责。
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------

def find_extrema(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """在平滑收盘价序列中找出所有波峰和波谷。

    参数
    ----
    df : pd.DataFrame
        smoother.smooth_ohlcv() 的输出，必须包含 'close_smooth' 列。
    cfg : dict
        从 configs/config.yaml 加载的完整配置。

    返回
    ----
    pd.DataFrame
        每行对应一个极值点，包含以下列：
            bar_idx   : 在 df 中的整数位置索引
            timestamp : DatetimeIndex 中的时间戳
            price     : 该极值点处的平滑价格
            kind      : 'peak'（波峰）或 'trough'（波谷）
    """
    val_cfg = cfg["labeling"]["extrema_detection"]["extrema_validation"]

    prices = df["close_smooth"].to_numpy(dtype=float)

    # 步骤 1 — 计算一阶和二阶导数
    d1 = _central_difference(prices, order=1)
    d2 = _central_difference(prices, order=2)

    # 步骤 2 — 找一阶导数的过零点（候选极值位置）
    candidates = _zero_crossings(d1, val_cfg["zero_crossing_epsilon"])

    # 步骤 3 — 用二阶导数的符号判断是波峰还是波谷
    min_d2 = val_cfg["min_second_derivative_abs"]
    extrema = []
    for idx in candidates:
        if abs(d2[idx]) < min_d2:
            continue                       # 二阶导数近似为零 → 平坦拐点，跳过
        kind = "peak" if d2[idx] < 0 else "trough"  # 二阶导数<0 → 波峰；>0 → 波谷
        extrema.append({
            "bar_idx":   idx,
            "timestamp": df.index[idx],
            "price":     prices[idx],
            "kind":      kind,
        })

    result = pd.DataFrame(extrema)
    if result.empty:
        return result

    # 步骤 4 — 强制相邻极值点之间有最小间隔
    min_sep = val_cfg["min_extrema_separation_bars"]
    result  = _filter_by_separation(result, min_sep)

    # 步骤 5 — 强制严格交替（波峰→波谷→波峰→…）
    if val_cfg["enforce_alternation"]:
        result = _enforce_alternation(result)

    result = result.reset_index(drop=True)
    print(
        f"[extrema] Found {len(result)} extrema  "
        f"({(result['kind']=='peak').sum()} peaks, "
        f"{(result['kind']=='trough').sum()} troughs)"
    )
    return result


# ---------------------------------------------------------------------------
# 导数计算
# ---------------------------------------------------------------------------

def _central_difference(prices: np.ndarray, order: int) -> np.ndarray:
    """用中心差分法计算导数数组。

    参数
    ----
    prices : np.ndarray
        一维价格（或导数）数组。
    order : int
        1 表示一阶导数，2 表示二阶导数。

    返回
    ----
    np.ndarray
        与输入等长的导数数组。
        首尾两端用前向/后向差分估算，保证输出长度不变。
    """
    if order == 1:
        d = np.empty_like(prices)
        # 内部点：中心差分  (prices[i+1] - prices[i-1]) / 2
        d[1:-1] = (prices[2:] - prices[:-2]) / 2.0
        # 边界点：前向/后向差分
        d[0]    = prices[1]  - prices[0]
        d[-1]   = prices[-1] - prices[-2]
        return d

    if order == 2:
        d = np.empty_like(prices)
        # 内部点：标准二阶中心差分  prices[i+1] - 2*prices[i] + prices[i-1]
        d[1:-1] = prices[2:] - 2.0 * prices[1:-1] + prices[:-2]
        # 边界点：复制相邻内部点的值
        d[0]    = d[1]
        d[-1]   = d[-2]
        return d

    raise ValueError(f"order must be 1 or 2, got {order}")


# ---------------------------------------------------------------------------
# 过零点检测
# ---------------------------------------------------------------------------

def _zero_crossings(d1: np.ndarray, epsilon: float) -> list[int]:
    """返回一阶导数穿越（或触及）零点的 K 线索引列表。

    当连续两个 d1 值发生符号变化时，认为发生了过零。
    选取 |d1| 较小的那个点，因为它更接近真实极值位置。

    参数
    ----
    d1 : np.ndarray
        一阶导数数组。
    epsilon : float
        |d1| < epsilon 的点被视为精确零点。

    返回
    ----
    list[int]
        有序的 K 线索引列表（每个过零点对应一个）。
    """
    crossings = []
    n = len(d1)

    i = 0
    while i < n - 1:
        a, b = d1[i], d1[i + 1]

        # 当前点精确为零
        if abs(a) < epsilon:
            crossings.append(i)
            i += 1
            continue

        # 相邻两点符号不同 → 发生过零，选绝对值更小（更接近零）的那个
        if a * b < 0:
            pick = i if abs(a) <= abs(b) else i + 1
            crossings.append(pick)
            i += 2          # 跳过配对点，避免重复计数
            continue

        i += 1

    return sorted(set(crossings))


# ---------------------------------------------------------------------------
# 后处理过滤器
# ---------------------------------------------------------------------------

def _filter_by_separation(df_ext: pd.DataFrame, min_sep: int) -> pd.DataFrame:
    """删除时间上过于接近的极值点。

    当同类型（同为波峰或同为波谷）的两个极值点相距不足 min_sep 根 K 线时，
    保留价格更极端的那个（更高的波峰或更低的波谷）。

    参数
    ----
    df_ext : pd.DataFrame
        含 bar_idx、price、kind 列的极值点 DataFrame。
    min_sep : int
        任意两个极值点之间的最小 K 线间隔。

    返回
    ----
    pd.DataFrame
        过滤后的极值点 DataFrame。
    """
    keep    = np.ones(len(df_ext), dtype=bool)
    indices = df_ext["bar_idx"].to_numpy()
    prices  = df_ext["price"].to_numpy()
    kinds   = df_ext["kind"].to_numpy()

    for i in range(len(df_ext)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(df_ext)):
            if not keep[j]:
                continue
            if indices[j] - indices[i] > min_sep:
                break                          # 间距已超过阈值，后续点无需检查
            # 间距过小：保留更极端的那个
            if kinds[i] == kinds[j] == "peak":
                # 两个波峰：保留更高的
                if prices[i] >= prices[j]:
                    keep[j] = False
                else:
                    keep[i] = False
            elif kinds[i] == kinds[j] == "trough":
                # 两个波谷：保留更低的
                if prices[i] <= prices[j]:
                    keep[j] = False
                else:
                    keep[i] = False

    return df_ext[keep].copy()


def _enforce_alternation(df_ext: pd.DataFrame) -> pd.DataFrame:
    """确保极值点严格交替出现（波峰→波谷→波峰→…）。

    当连续两个极值点类型相同时，删除较不极端的那个：
      - 连续两个波峰 → 保留更高的
      - 连续两个波谷 → 保留更低的

    参数
    ----
    df_ext : pd.DataFrame
        按 bar_idx 排序的极值点 DataFrame。

    返回
    ----
    pd.DataFrame
        保证严格波峰/波谷交替的极值点 DataFrame。
    """
    changed = True
    while changed:
        # 循环直到没有任何修改（处理连锁冲突）
        changed = False
        rows    = df_ext.reset_index(drop=True)
        drop    = set()

        for i in range(len(rows) - 1):
            if i in drop:
                continue
            j = i + 1
            # 跳过已标记删除的点
            while j in drop and j < len(rows):
                j += 1
            if j >= len(rows):
                break

            ki = rows.loc[i, "kind"]
            kj = rows.loc[j, "kind"]

            if ki == kj:
                # 类型相同 → 删除较不极端的那个
                pi = rows.loc[i, "price"]
                pj = rows.loc[j, "price"]
                if ki == "peak":
                    drop.add(i if pi < pj else j)   # 删除较低的波峰
                else:
                    drop.add(i if pi > pj else j)   # 删除较高的波谷
                changed = True

        df_ext = rows.drop(index=list(drop)).reset_index(drop=True)

    return df_ext
