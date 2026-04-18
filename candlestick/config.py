from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or interpreted."""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base and return a new dictionary."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_override_value(raw: str) -> Any:
    """Parse CLI override values with YAML semantics (bool/int/float/list/etc)."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_set_overrides(cfg: dict[str, Any], assignments: list[str] | None = None) -> dict[str, Any]:
    """Apply dot-notation key=value assignments on top of cfg."""
    if not assignments:
        return cfg

    updated = deepcopy(cfg)
    for item in assignments:
        if "=" not in item:
            raise ConfigError(f"Invalid override '{item}'. Expected key=value format.")

        path, raw_value = item.split("=", 1)
        value = _parse_override_value(raw_value)
        keys = path.split(".")

        cursor: dict[str, Any] = updated
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = value

    return updated


def load_config(
    base_path: str | Path,
    override_paths: list[str | Path] | None = None,
    set_overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Load base config and optional layered overrides."""
    path = Path(base_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    for override_path in override_paths or []:
        override = Path(override_path)
        if not override.exists():
            raise ConfigError(f"Override config not found: {override}")
        with override.open("r", encoding="utf-8") as f:
            layer = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, layer)

    cfg = apply_set_overrides(cfg, set_overrides)
    return cfg


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
