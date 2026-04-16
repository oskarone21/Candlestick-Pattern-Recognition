# CLAUDE.md — Rules for Claude Code
# Project: Candlestick Pattern Recognition (WM9B7 AIDL)
# Owner: Henry

Read this file before doing anything else in this project.

---

## 1. Mandatory reading on session start

Before making any code or config changes, read these files:

- `TEAM_STANDARDS.md` — coding and collaboration standards
- `PATTERNS_EXPLAINED.md` — academic definitions for all 4 patterns (required before touching labeling logic)
- `configs/config.yaml` — shared base config
- `configs/local/henry.experiment.yaml` — Henry's active overrides

---

## 2. Competition integrity (non-negotiable)

This is a team competition where each member builds their model independently. Performance is compared at the end.

- **Do not read, reference, or infer from other members' config files** (e.g. `oskar.experiment.yaml`, `hongren.experiment.yaml`).
- **Do not suggest approaches based on what other members "might be doing".**
- Henry's model must be independently reproducible from his own code and config alone.

---

## 3. Config rules (enforced, no exceptions)

- **Never modify `configs/config.yaml`** — it is the shared team baseline and is locked.
- All experiment tuning goes exclusively into `configs/local/henry.experiment.yaml`.
- Never hardcode numeric values (learning rate, thresholds, bar counts, etc.) directly in Python files — all tunable values belong in config.
- CLI overrides are allowed for one-off tests but must not replace persisted config changes.

---

## 4. Code change discipline

- After every set of changes, list all modified files and state any assumptions made.
- Prefer incremental, targeted edits over large rewrites.
- Do not refactor, clean up, or "improve" code outside the scope of the current task.
- Do not add docstrings, comments, or type hints to code you didn't change.

---

## 5. Labeling and data integrity

- Do not introduce data leakage. All splits must be time-based.
- Any lookahead horizon used in confirmation logic must be documented in config.
- Before modifying any pattern detector in `src/candlestick/labeling.py`, re-read `PATTERNS_EXPLAINED.md`.
- Do not synthesize or modify raw data files under `data/`.

---

## 6. Environment

- All Python commands must be run inside the project venv.
- Activation: `cd ~/AAI/DL/Candlestick-Pattern-Recognition && source .venv/bin/activate`
- Device is M4 Mac with MPS support. Do not assume CUDA unless testing on Azure VM.

---

## 7. Git

- Branch naming: `feature/henry-<topic>` or `fix/henry-<topic>`
- Commit style: `feat: ...`, `fix: ...`, `refactor: ...`
- Do not push directly to `main` or `master`.
- Do not push anything without Henry's explicit instruction.
