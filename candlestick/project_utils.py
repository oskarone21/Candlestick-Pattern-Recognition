from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from candlestick.config import deep_merge
from candlestick.domain import (
    CFG_DATA_SOURCE,
    CFG_INSTRUMENT,
    CFG_LABELING,
    CFG_MODEL_SELECTION,
    CFG_PROCESSED_15M_PATH,
    CFG_PROJECT,
    CFG_TIMESTAMP,
    CFG_CONVERT_TO_TIMEZONE,
    COLUMN_SYMBOL,
    COLUMN_TS_EVENT,
    DEFAULT_ALLOWED_PATTERNS,
    DEFAULT_CANDIDATE_MODELS,
    DEFAULT_INSTRUMENT,
    DEFAULT_PROCESSED_15M_PATH,
    DEFAULT_RUN_NAME,
    DEFAULT_WORKING_TIMEZONE,
)


def timeframe_to_minutes(rule: str) -> int:
    normalized = str(rule).strip().lower()
    if normalized.endswith("min"):
        return int(normalized[:-3])
    if normalized.endswith("m"):
        return int(normalized[:-1])
    if normalized.endswith("h"):
        return int(normalized[:-1]) * 60
    raise ValueError(f"Unsupported timeframe rule: {rule}")


def instrument_from_cfg(cfg: dict[str, Any]) -> str:
    return str(cfg.get("data_source", {}).get("instrument", DEFAULT_INSTRUMENT))


def working_timezone_from_cfg(cfg: dict[str, Any]) -> str:
    timestamp_cfg = cfg.get("data_source", {}).get("timestamp", {})
    return str(timestamp_cfg.get("convert_to_timezone", DEFAULT_WORKING_TIMEZONE))


def run_name_from_cfg(cfg: dict[str, Any]) -> str:
    return str(cfg.get("project", {}).get("run_name", DEFAULT_RUN_NAME))


def allowed_patterns_from_cfg(cfg: dict[str, Any]) -> list[str]:
    patterns = cfg.get("labeling", {}).get("allowed_patterns")
    if patterns:
        return [str(pattern) for pattern in patterns]
    return list(DEFAULT_ALLOWED_PATTERNS)


def candidate_models_from_cfg(cfg: dict[str, Any]) -> list[str]:
    models = cfg.get("model_selection", {}).get("candidate_models")
    if models:
        return [str(model) for model in models]
    return list(DEFAULT_CANDIDATE_MODELS)


def model_selection_from_cfg(cfg: dict[str, Any], pattern: str | None = None) -> dict[str, Any]:
    selection_cfg = dict(cfg.get("model_selection", {}))
    if not pattern:
        return selection_cfg

    per_pattern_cfg = selection_cfg.get("per_pattern", {})
    if not isinstance(per_pattern_cfg, dict):
        return selection_cfg

    pattern_override = per_pattern_cfg.get(pattern, {})
    if not isinstance(pattern_override, dict):
        return selection_cfg

    return deep_merge(selection_cfg, pattern_override)


def load_processed_prices(
    cfg: dict[str, Any],
    *,
    path_key: str = "processed_15m_path",
    instrument: str | None = None,
) -> pd.DataFrame:
    prices_path = Path(cfg["paths"].get(path_key, DEFAULT_PROCESSED_15M_PATH))
    if not prices_path.exists():
        raise FileNotFoundError(f"Processed prices not found: {prices_path}")

    prices = pd.read_csv(prices_path)
    prices[COLUMN_TS_EVENT] = pd.to_datetime(
        prices[COLUMN_TS_EVENT],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(working_timezone_from_cfg(cfg))
    prices = prices.dropna(subset=[COLUMN_TS_EVENT]).copy()

    selected_instrument = (instrument or instrument_from_cfg(cfg)).upper()
    if COLUMN_SYMBOL in prices.columns:
        prices = prices[prices[COLUMN_SYMBOL].astype(str).str.upper() == selected_instrument].copy()

    return prices.sort_values(COLUMN_TS_EVENT).reset_index(drop=True)
