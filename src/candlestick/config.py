"""Configuration loader following TEAM_STANDARDS.md §2.

Merge order: base config → optional local override → CLI overrides.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    base_path: str | Path = "configs/config.yaml",
    override_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> dict:
    """Load and merge configuration following team standards.

    Parameters
    ----------
    base_path : str | Path
        Path to the shared ``config.yaml``.
    override_path : str | Path | None
        Optional local experiment override YAML.
    cli_overrides : dict | None
        Dot-separated key→value pairs that take highest priority.

    Returns
    -------
    dict
        Fully resolved configuration dictionary.
    """
    base_path = Path(base_path)
    with open(base_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if override_path is not None:
        override_path = Path(override_path)
        if override_path.exists():
            with open(override_path, "r", encoding="utf-8") as f:
                override = yaml.safe_load(f) or {}
            config = deep_merge(config, override)

    if cli_overrides:
        for dotted_key, value in cli_overrides.items():
            keys = dotted_key.split(".")
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

    return config


def get_nested(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """Safely retrieve a nested config value using dot notation."""
    keys = dotted_key.split(".")
    d = cfg
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is default:
            return default
    return d
