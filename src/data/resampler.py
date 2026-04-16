"""
OHLCV 重采样器：将 1 分钟 K 线聚合为更高时间周期。

从 config.yaml 读取重采样规则，不硬编码任何值。
遵循 TEAM_STANDARDS.md：配置驱动、无数据泄漏、单一职责。

关键规则（来自 TEAM_STANDARDS）：
- open=first, high=max, low=min, close=last, volume=sum
- 不跨收盘间隙进行前向填充或合成 K 线。
- 删除边界处的不完整 K 线。
"""

import pandas as pd


def resample_ohlcv(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """将 1 分钟 OHLCV DataFrame 重采样到目标时间周期。

    参数
    ----
    df : pd.DataFrame
        来自 loader.load_raw_ohlcv() 的干净 1 分钟 OHLCV DataFrame。
        必须有带时区的 DatetimeIndex，列名与 cfg['data_source']['columns'] 一致。
    cfg : dict
        从 configs/config.yaml 加载的完整配置。

    返回
    ----
    pd.DataFrame
        目标时间周期的 OHLCV DataFrame。
        边界处的不完整 K 线已删除，收盘间隙自然保留（无合成 K 线）。
    """
    rs_cfg     = cfg["resampling"]
    col_cfg    = cfg["data_source"]["columns"]
    target_tf  = rs_cfg["target_timeframe"]  # 目标时间周期，如 "15min"

    col_open   = col_cfg["open"]
    col_high   = col_cfg["high"]
    col_low    = col_cfg["low"]
    col_close  = col_cfg["close"]
    col_volume = col_cfg["volume"]

    # 从 config 读取聚合规则
    agg_rules = {
        col_open:   rs_cfg["ohlcv_agg"]["open"],    # "first"：取周期内第一根 open
        col_high:   rs_cfg["ohlcv_agg"]["high"],    # "max"  ：取周期内最高价
        col_low:    rs_cfg["ohlcv_agg"]["low"],     # "min"  ：取周期内最低价
        col_close:  rs_cfg["ohlcv_agg"]["close"],   # "last" ：取周期内最后一根 close
        col_volume: rs_cfg["ohlcv_agg"]["volume"],  # "sum"  ：成交量求和
    }

    # ------------------------------------------------------------------
    # 执行重采样
    # closed="left"  ：09:00 这根 K 线覆盖 [09:00, 09:15)
    # label="left"   ：K 线以其开盘时间戳命名
    # ------------------------------------------------------------------
    resampled = (
        df.resample(target_tf, closed="left", label="left")
        .agg(agg_rules)
    )

    # ------------------------------------------------------------------
    # 删除任意 OHLCV 列为 NaN 的行。
    # 这些行出现在市场收盘间隙（该时段没有 1 分钟 K 线），
    # 通过 dropna 自然保留间隙，无需前向填充。
    # ------------------------------------------------------------------
    before = len(resampled)
    resampled = resampled.dropna()
    gaps_removed = before - len(resampled)

    # ------------------------------------------------------------------
    # 删除不完整 K 线（配置项 drop_incomplete_bars）。
    # 不完整 K 线：包含的 1 分钟 K 线数量少于预期值。
    # 例如 15 分钟 K 线应包含恰好 15 根 1 分钟 K 线。
    # 通过统计每个重采样周期内的源 K 线数量来检测。
    # ------------------------------------------------------------------
    if rs_cfg["drop_incomplete_bars"]:
        # 统计每个目标周期内实际包含的 1 分钟 K 线数
        bar_counts = df.resample(target_tf, closed="left", label="left")[col_close].count()
        expected   = _expected_bars(rs_cfg["base_timeframe"], target_tf)
        # 只保留达到预期数量的完整 K 线
        complete_mask = bar_counts >= expected
        resampled = resampled[complete_mask.reindex(resampled.index, fill_value=False)]

    incomplete_removed = before - gaps_removed - len(resampled)

    print(
        f"[resampler] {rs_cfg['base_timeframe']} → {target_tf} | "
        f"{len(resampled):,} bars | "
        f"gap bars removed: {gaps_removed} | "
        f"incomplete bars removed: {incomplete_removed}"
    )

    return resampled


def _expected_bars(base_tf: str, target_tf: str) -> int:
    """计算一个目标 K 线应包含多少根基础 K 线。

    参数
    ----
    base_tf : str
        基础时间周期字符串，如 '1min'。
    target_tf : str
        目标时间周期字符串，如 '15min'。

    返回
    ----
    int
        每根目标 K 线预期包含的基础 K 线数量。
    """
    base_minutes   = _to_minutes(base_tf)
    target_minutes = _to_minutes(target_tf)
    return target_minutes // base_minutes


def _to_minutes(tf: str) -> int:
    """将时间周期字符串（如 '1min'、'15min'、'1h'）转换为分钟数。

    参数
    ----
    tf : str
        时间周期字符串。

    返回
    ----
    int
        等效的分钟数。

    异常
    ----
    ValueError
        若时间周期字符串格式无法识别则抛出。
    """
    tf = tf.strip().lower()
    if tf.endswith("min"):
        return int(tf[:-3])          # 例如 "15min" → 15
    if tf.endswith("h"):
        return int(tf[:-1]) * 60     # 例如 "1h" → 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24  # 例如 "1d" → 1440
    raise ValueError(f"Unrecognised timeframe format: '{tf}'. Use e.g. '1min', '15min', '1h'.")
