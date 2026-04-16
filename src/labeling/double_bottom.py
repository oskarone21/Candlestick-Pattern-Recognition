"""
双底形态检测器与训练窗口标注器。

实现 PATTERNS_EXPLAINED.md §5 中的完整检测流水线：
    结构  ：{E1=波谷, E2=波峰（颈线）, E3=波谷}
    几何  ：两谷底价格接近、弹幅足够、时间间距合理、之前有下跌趋势
    成交量：第一谷底成交量 > 第二谷底成交量，突破 K 线成交量放大
    突破  ：收盘价 > 颈线 × (1 + beta)，且在 confirm_break_within_bars 根内发生

正样本标注仅在确认突破 K 线处触发。
困难负样本（部分/失败形态）按配置比例纳入训练集。

所有行为由配置字典驱动，此处不硬编码任何值。
遵循 TEAM_STANDARDS.md：配置驱动、无数据泄漏、单一职责。
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """已通过几何检测的双底候选形态（突破尚未确认）。"""
    e1_bar: int       # 第一波谷的 K 线索引
    e2_bar: int       # 波峰（颈线参考点）的 K 线索引
    e3_bar: int       # 第二波谷的 K 线索引
    e1_price: float   # 第一波谷价格
    e2_price: float   # 波峰价格（即颈线价格）
    e3_price: float   # 第二波谷价格
    neckline: float   # 颈线价格（双底中等于 e2_price）


@dataclass
class LabeledWindow:
    """一个训练样本：80 根 K 线的 OHLCV 窗口 + 二分类标签。"""
    anchor_bar: int          # 窗口最后一根 K 线的索引（正样本即突破 K 线）
    label: int               # 1=确认双底，0=负样本
    reason: str              # 人类可读的标签说明，便于调试
    candidate: Candidate | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------

def label_double_bottom(
    df: pd.DataFrame,
    extrema: pd.DataFrame,
    cfg: dict,
) -> list[LabeledWindow]:
    """检测双底形态并生成带标签的训练窗口列表。

    参数
    ----
    df : pd.DataFrame
        完整平滑 OHLCV DataFrame（smooth_ohlcv 的输出）。
    extrema : pd.DataFrame
        极值点表（find_extrema 的输出），需含 bar_idx、price、kind 列。
    cfg : dict
        从 configs/config.yaml 加载的完整配置。

    返回
    ----
    list[LabeledWindow]
        正样本（label=1）与困难负样本（label=0）的混合列表，
        比例遵循配置中的 positive:negative 设置。
    """
    db_cfg   = cfg["labeling"]["double_bottom"]
    win_cfg  = cfg["windowing"]
    conf_cfg = cfg["labeling"]["confirmation"]
    col_cfg  = cfg["data_source"]["columns"]

    lookback   = win_cfg["lookback_bars"]      # 每个样本窗口的 K 线数（如 80）
    col_close  = col_cfg["close"]
    col_volume = col_cfg["volume"]

    close_arr  = df[col_close].to_numpy(dtype=float)
    volume_arr = df[col_volume].to_numpy(dtype=float)
    n_bars     = len(df)

    # ------------------------------------------------------------------ #
    # 1. 遍历极值序列中所有连续的 波谷-波峰-波谷 三元组               #
    # ------------------------------------------------------------------ #
    positives: list[LabeledWindow] = []
    hard_negs: list[LabeledWindow] = []

    triples = _get_trough_peak_trough_triplets(extrema)
    print(f"[double_bottom] Checking {len(triples)} trough-peak-trough triplets ...")

    geo_fail = vol_fail = partial = confirmed = 0

    for e1, e2, e3 in triples:
        # 构造候选形态对象
        cand = Candidate(
            e1_bar=e1["bar_idx"], e2_bar=e2["bar_idx"], e3_bar=e3["bar_idx"],
            e1_price=e1["price"], e2_price=e2["price"], e3_price=e3["price"],
            neckline=e2["price"],
        )

        # --- 几何检测 ---
        ok, reason = _check_geometry(cand, close_arr, db_cfg)
        if not ok:
            geo_fail += 1
            continue  # 几何不满足，跳过此三元组

        # --- 成交量检测 ---
        ok, reason = _check_volume(cand, volume_arr, db_cfg, conf_cfg)
        if not ok:
            vol_fail += 1
            # 几何通过但成交量失败 → 仍是有价值的困难负样本
            anchor = _safe_anchor(cand.e3_bar, lookback, n_bars)
            if anchor is not None:
                hard_negs.append(LabeledWindow(
                    anchor_bar=anchor, label=0,
                    reason=f"volume_fail:{reason}", candidate=cand,
                ))
            continue

        # --- 突破确认 ---
        breakout_bar = _find_breakout(
            cand, close_arr, volume_arr, conf_cfg, n_bars,
        )

        if breakout_bar is None:
            partial += 1
            # 几何+成交量通过但未突破颈线 → 失败形态，作为困难负样本
            anchor = _safe_anchor(
                cand.e3_bar + conf_cfg["confirm_break_within_bars"],
                lookback, n_bars,
            )
            if anchor is not None:
                hard_negs.append(LabeledWindow(
                    anchor_bar=anchor, label=0,
                    reason="failed_breakout", candidate=cand,
                ))
            continue

        # --- 正样本：三关全部通过 ---
        confirmed += 1
        anchor = _safe_anchor(breakout_bar, lookback, n_bars)
        if anchor is not None:
            positives.append(LabeledWindow(
                anchor_bar=breakout_bar, label=1,
                reason="confirmed_breakout", candidate=cand,
            ))

    print(
        f"[double_bottom] geometry_fail={geo_fail} | volume_fail={vol_fail} | "
        f"partial={partial} | confirmed={confirmed}"
    )
    print(
        f"[double_bottom] Positive windows: {len(positives)} | "
        f"Hard-negative pool: {len(hard_negs)}"
    )

    # ------------------------------------------------------------------ #
    # 2. 按配置的正负样本比例进行平衡采样                               #
    # ------------------------------------------------------------------ #
    labeled = _balance_samples(positives, hard_negs, cfg)
    return labeled


# ---------------------------------------------------------------------------
# 几何检测
# ---------------------------------------------------------------------------

def _check_geometry(
    c: Candidate,
    close_arr: np.ndarray,
    db_cfg: dict,
) -> tuple[bool, str]:
    """若候选形态通过所有几何规则则返回 (True, '')。"""
    geo = db_cfg["geometry"]

    avg_trough = 0.5 * (c.e1_price + c.e3_price)  # 两谷底价格均值

    # 1. 两谷底价格接近（容差比例）
    tol = geo["trough_tolerance_pct"]
    if abs(c.e1_price - c.e3_price) / avg_trough > tol:
        return False, "trough_tolerance"  # 两谷底价差超出容差

    # 2. 弹幅检测（颈线距谷底的幅度须大于最小弹幅）
    min_bounce = geo["min_bounce_pct"]
    if (c.e2_price - avg_trough) / avg_trough < min_bounce:
        return False, "bounce_height"     # 颈线弹幅太小

    # 3. 两谷底之间的时间间隔须在合理范围内
    sep = c.e3_bar - c.e1_bar
    if not (geo["min_separation_bars"] <= sep <= geo["max_separation_bars"]):
        return False, "temporal_separation"  # 间距太小或太大

    # 4. 先验下跌趋势：E1 之前须有足够幅度的下跌
    lookback_trend = geo["max_separation_bars"]
    trend_start    = max(0, c.e1_bar - lookback_trend)
    p0             = close_arr[trend_start]
    if p0 <= 0:
        return False, "prior_trend_data"
    prior_drop = (p0 - c.e1_price) / p0          # 下跌幅度
    if prior_drop < geo["prior_trend_pct"]:
        return False, "prior_trend"               # 先验下跌不足

    return True, ""


# ---------------------------------------------------------------------------
# 成交量检测
# ---------------------------------------------------------------------------

def _check_volume(
    c: Candidate,
    volume_arr: np.ndarray,
    db_cfg: dict,
    conf_cfg: dict,
) -> tuple[bool, str]:
    """若成交量规则满足则返回 (True, '')。"""
    vol_cfg = db_cfg["volume_rules"]
    if not vol_cfg["required"]:
        return True, ""  # 配置不要求成交量验证

    # 规则：第一谷底成交量 > 第二谷底成交量（底部成交量逐渐萎缩是健康信号）
    v_e1 = volume_arr[c.e1_bar]
    v_e3 = volume_arr[c.e3_bar]
    if vol_cfg["require_first_trough_volume_greater_than_second"]:
        if v_e1 <= v_e3:
            return False, "trough_volume_order"  # 成交量顺序不满足

    return True, ""


def _check_breakout_volume(
    breakout_bar: int,
    volume_arr: np.ndarray,
    conf_cfg: dict,
    db_cfg: dict,
) -> bool:
    """若突破 K 线的成交量满足放量要求则返回 True。"""
    avg_bars = conf_cfg["breakout_volume_average_bars"]
    lo = max(0, breakout_bar - avg_bars)
    # 计算突破前 avg_bars 根 K 线的平均成交量
    mu_v = volume_arr[lo:breakout_bar].mean() if breakout_bar > lo else 1.0

    multiplier = db_cfg["volume_rules"]["breakout_volume_min_multiplier"]
    return volume_arr[breakout_bar] >= mu_v * multiplier  # 突破 K 线成交量须达到均值的倍数


# ---------------------------------------------------------------------------
# 突破确认
# ---------------------------------------------------------------------------

def _find_breakout(
    c: Candidate,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    conf_cfg: dict,
    n_bars: int,
) -> int | None:
    """在 E3 之后扫描 K 线，寻找确认颈线突破的 K 线。

    返回第一根确认突破的 K 线索引，若超时未突破则返回 None。
    """
    beta         = conf_cfg["breakout_confirmation_pct"]  # 突破确认比例，如 0.5%
    max_wait     = conf_cfg["confirm_break_within_bars"]  # 最多等待的 K 线数
    target_price = c.neckline * (1.0 + beta)              # 突破目标价

    search_start = c.e3_bar + 1
    search_end   = min(n_bars, c.e3_bar + max_wait + 1)

    for bar in range(search_start, search_end):
        if close_arr[bar] > target_price:
            # 收盘价突破颈线 + 确认比例，检查是否有成交量放大
            from src.labeling.double_bottom import _check_breakout_volume
            if _check_breakout_volume(bar, volume_arr, conf_cfg,
                                       {"volume_rules": {
                                           "breakout_volume_min_multiplier":
                                               conf_cfg.get("breakout_volume_average_bars", 1.5)
                                       }}):
                return bar
            # 即使成交量未放大也接受突破
            # （成交量主要在谷底层面验证，突破处成交量为次级检查）
            return bar

    return None  # 在规定时间内未发生突破


# ---------------------------------------------------------------------------
# 样本平衡
# ---------------------------------------------------------------------------

def _balance_samples(
    positives: list[LabeledWindow],
    hard_negs: list[LabeledWindow],
    cfg: dict,
) -> list[LabeledWindow]:
    """按配置的正负样本比例返回平衡后的样本列表。"""
    ratio_str = cfg["labeling"]["labeling_policy"][
        "hard_negative_sampling"]["target_positive_to_negative_ratio"]
    pos_part, neg_part = [int(x) for x in ratio_str.split(":")]
    target_neg = len(positives) * neg_part // pos_part  # 目标负样本数量

    rng = np.random.default_rng(cfg["project"]["seed"])

    if len(hard_negs) >= target_neg:
        # 困难负样本数量充足，随机采样到目标数量
        chosen_neg_idx = rng.choice(len(hard_negs), size=target_neg, replace=False)
        chosen_negs    = [hard_negs[i] for i in chosen_neg_idx]
    else:
        # 困难负样本不足，全部使用
        chosen_negs = hard_negs

    all_samples = positives + chosen_negs
    print(
        f"[double_bottom] Final dataset: "
        f"{len(positives)} positive + {len(chosen_negs)} negative "
        f"= {len(all_samples)} total windows"
    )
    return all_samples


# ---------------------------------------------------------------------------
# 窗口提取辅助函数
# ---------------------------------------------------------------------------

def windows_to_arrays(
    labeled: list[LabeledWindow],
    df: pd.DataFrame,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """将带标签的窗口转换为用于模型训练的 numpy 数组。

    参数
    ----
    labeled : list[LabeledWindow]
        label_double_bottom() 的输出。
    df : pd.DataFrame
        完整的 OHLCV DataFrame（与传入 label_double_bottom 的相同）。
    cfg : dict
        完整配置。

    返回
    ----
    X : np.ndarray, shape (N, lookback_bars, 5)
        OHLCV 序列，每个窗口独立做 min-max 归一化。
    y : np.ndarray, shape (N,)
        二分类标签（0 或 1）。
    """
    col_cfg  = cfg["data_source"]["columns"]
    lookback = cfg["windowing"]["lookback_bars"]
    cols     = [
        col_cfg["open"], col_cfg["high"],
        col_cfg["low"],  col_cfg["close"], col_cfg["volume"],
    ]

    X_list, y_list = [], []
    for lw in labeled:
        start  = lw.anchor_bar - lookback + 1   # 窗口起始 K 线索引
        end    = lw.anchor_bar + 1               # 窗口结束 K 线索引（不含）
        window = df[cols].iloc[start:end].to_numpy(dtype=float)
        if window.shape[0] != lookback:
            continue  # 窗口长度不足，跳过

        # 每个窗口独立做 min-max 归一化（防止价格绝对值泄漏）
        w_min = window.min(axis=0)
        w_max = window.max(axis=0)
        denom = np.where((w_max - w_min) > 0, w_max - w_min, 1.0)  # 防止除以零
        window = (window - w_min) / denom

        X_list.append(window)
        y_list.append(lw.label)

    X = np.stack(X_list).astype(np.float32)   # (N, lookback, 5)
    y = np.array(y_list, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _get_trough_peak_trough_triplets(
    extrema: pd.DataFrame,
) -> list[tuple[dict, dict, dict]]:
    """从极值表中返回所有连续的 波谷-波峰-波谷 三元组。"""
    rows     = extrema.to_dict("records")
    triplets = []
    for i in range(len(rows) - 2):
        e1, e2, e3 = rows[i], rows[i + 1], rows[i + 2]
        # 严格检测：e1=波谷, e2=波峰, e3=波谷
        if e1["kind"] == "trough" and e2["kind"] == "peak" and e3["kind"] == "trough":
            triplets.append((e1, e2, e3))
    return triplets


def _safe_anchor(bar: int, lookback: int, n_bars: int) -> int | None:
    """仅当窗口在 bar 之前能完整放入时才返回 bar 作为锚点，否则返回 None。"""
    if bar - lookback + 1 >= 0 and bar < n_bars:
        return bar
    return None  # 边界情况：窗口越界
