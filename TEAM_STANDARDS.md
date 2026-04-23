# Team Standards

This document defines the engineering and collaboration standards for this project.

The goal is consistency: reproducible experiments, clear code ownership, and fewer integration issues across team members.

## 1) Core principles

- `configs/config.yaml` is the shared source of truth for repeatable settings.
- Prefer reproducibility over convenience.
- Make small, reviewable changes.
- Keep code and experiment behavior explicit (no hidden defaults).

## 2) Configuration policy

### Shared config

- Shared defaults live in `configs/config.yaml`.
- Only commit changes to shared config when they are intentional team-level decisions.
- Never hardcode experiment values (learning rate, timeframe, symbol, class list) in Python files.

### Local experiment overrides

- For personal experiments, create a local override file, for example:
  - `configs/local/oskar.experiment.yaml`
- Local override files are for temporary tuning and should not be merged unless agreed by the team.
- Merge order should be:
  1. `configs/config.yaml`
  2. local override file
  3. CLI key-value overrides (highest priority)

In-script example (load base config and optional override):

```python
from copy import deepcopy
import yaml


def deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(base_path: str, override_path: str | None = None) -> dict:
    with open(base_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if override_path:
        with open(override_path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f)
        config = deep_merge(config, override)
    return config


cfg = load_config(
    "configs/config.yaml",
    "configs/local/oskar.experiment.yaml",
)

print("Run name:", cfg["project"]["run_name"])
print("Learning rate:", cfg["optimizer"]["lr"])
```

Recommended run examples:

```bash
python scripts/run_experiment_suite.py \
  --config configs/config.yaml \
  --config-override configs/local/alex.experiment.yaml

python scripts/run_experiment_suite.py \
  --config configs/config.yaml \
  --set optimizer.lr=0.001 \
  --set training.epochs=10
```

## 3) Data and time handling standards

- Canonical raw data file: `data/nq_1min.csv`.
- Timestamp column is `ts_event` and is timezone-aware UTC in source data.
- Convert to `America/New_York` in the pipeline, not manually in notebooks.
- Do not forward-fill or synthesize bars across market-close gaps.
- Respect session boundaries when resampling.

Resampling rules:

- `open=first`, `high=max`, `low=min`, `close=last`, `volume=sum`
- Base data remains `1min`; higher timeframes are derived from base.

## 4) Labeling and leakage rules

- Pattern rules must be numeric and configurable.
- Label definitions for all classes must be documented and versioned.
- No data leakage:
  - Use time-based splits.
  - Avoid future information in features unless explicitly part of target confirmation logic.
  - Document any lookahead horizon in config and reports.

## 5) Coding conventions

- Use Python 3.10+ style, follow PEP 8 naming and structure.
- Add type hints to non-trivial functions.
- Add short docstrings to public functions/classes.
- Keep modules single responsibility (data, labeling, training, evaluation separated).
- Do not use notebook-only logic as production pipeline logic.
- Avoid magic numbers; place tunable values in config.

## 6) Experiment tracking standards

- Every training run should log:
  - run name
  - git commit hash
  - full resolved config snapshot
  - metrics and confusion matrix
- Save artifacts under `outputs/` in organized run folders.
- Baseline metrics must be reproducible from config and code in the repository.

## 7) Git and pull request standards

- Branch naming:
  - `feature/<initials>-<topic>`
  - `fix/<initials>-<topic>`
  - `docs/<initials>-<topic>`
- Prefer conventional commit style:
  - `feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`, `test: ...`
- PRs should include:
  - what changed
  - why it changed
  - how it was validated

## 8) AI-assisted coding policy

AI tools are allowed, but output quality and correctness remain the developer's responsibility.

When using AI tools (Claude, ChatGPT, Copilot, etc.):

- Include this instruction in prompts: "Follow `TEAM_STANDARDS.md` and use `configs/config.yaml` keys."
- Ask the tool to list all changed files and assumptions.
- Do not accept generated code that silently changes global defaults.
- Validate generated code locally before committing.
- Prefer incremental generated changes over large one-shot rewrites.

## 9) Definition of done

A task is complete only if:

- Config-driven behavior is preserved.
- No leakage risks are introduced.
- Basic validation or tests run successfully.
- Relevant documentation is updated.
- Changes are understandable to another team member without extra context.
