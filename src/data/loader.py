"""
原始 OHLCV CSV 数据加载器。

读取 1 分钟 K 线 CSV 文件，按 config.yaml 中的规则做质量检查，
返回一个时区感知的干净 DataFrame，可直接送入重采样器。

所有行为由配置字典驱动，此处不硬编码任何值。
遵循 TEAM_STANDARDS.md：配置驱动、无数据泄漏、单一职责。
"""

import pandas as pd


def load_raw_ohlcv(cfg: dict) -> pd.DataFrame:
    """加载并校验 config 中定义的原始 OHLCV CSV 文件。

    处理步骤
    --------
    1. 按正确的分隔符和表头设置读取 CSV。
    2. 解析时间戳列并转为带时区的 UTC 时间。
    3. 转换到 config 指定的工作时区。
    4. 按时间升序排序。
    5. 质量检查（去重、删除 NaN 行、强制转为数值类型）。

    参数
    ----
    cfg : dict
        从 configs/config.yaml 加载的完整配置。

    返回
    ----
    pd.DataFrame
        带时区感知 DatetimeIndex 的干净 OHLCV DataFrame。
        列名取自 config（Open, High, Low, Close, Volume）。
    """
    src     = cfg["data_source"]
    ts_cfg  = src["timestamp"]
    col_cfg = src["columns"]
    quality = src["quality"]

    # ------------------------------------------------------------------
    # 1. 读取 CSV 文件
    # ------------------------------------------------------------------
    df = pd.read_csv(
        src["file_path"],
        delimiter=src["delimiter"],
        header=0 if src["has_header"] else None,
    )

    # ------------------------------------------------------------------
    # 2. 解析时间戳列 → 带时区的 UTC 索引
    # ------------------------------------------------------------------
    ts_col = ts_cfg["column"]

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)  # 强制解析为 UTC 时间
    df = df.set_index(ts_col)
    df.index.name = "timestamp"

    # ------------------------------------------------------------------
    # 3. 转换到工作时区（如 America/New_York）
    # ------------------------------------------------------------------
    target_tz = ts_cfg["convert_to_timezone"]
    df.index = df.index.tz_convert(target_tz)

    # ------------------------------------------------------------------
    # 4. 按时间升序排序（保证数据顺序正确）
    # ------------------------------------------------------------------
    if ts_cfg["sort_ascending"]:
        df = df.sort_index()

    # ------------------------------------------------------------------
    # 5. 只保留 OHLCV 五列（按 config 中定义的列名选取）
    # ------------------------------------------------------------------
    ohlcv_cols = [
        col_cfg["open"],
        col_cfg["high"],
        col_cfg["low"],
        col_cfg["close"],
        col_cfg["volume"],
    ]
    df = df[ohlcv_cols].copy()

    # ------------------------------------------------------------------
    # 6. 质量检查
    # ------------------------------------------------------------------
    original_len = len(df)

    if quality["enforce_numeric_ohlcv"]:
        # 将 OHLCV 列强制转为数值，无法转换的变为 NaN
        for col in ohlcv_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if quality["dropna_ohlcv"]:
        # 删除含有 NaN 的行
        df = df.dropna(subset=ohlcv_cols)

    if quality["drop_duplicates"]:
        # 删除重复时间戳，保留第一条
        df = df[~df.index.duplicated(keep="first")]

    removed = original_len - len(df)
    if removed > 0:
        print(f"[loader] Removed {removed} rows during quality checks.")

    print(
        f"[loader] Loaded {len(df):,} rows | "
        f"{df.index[0]} → {df.index[-1]} | "
        f"timezone: {target_tz}"
    )

    return df
