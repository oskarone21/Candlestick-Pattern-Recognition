from __future__ import annotations

from pathlib import Path

CANONICAL_PROFILE_ORDER = (
    "high_coverage",
    "conservative",
    "balanced",
    "high_recall",
)

PROFILE_NAME_ALIASES = {
    "high_coverage": "high_coverage",
    "intraday_broad_signal": "high_coverage",
    "conservative": "conservative",
    "intraday_conservative": "conservative",
    "balanced": "balanced",
    "intraday_balanced": "balanced",
    "high_recall": "high_recall",
    "intraday_recall": "high_recall",
}

PROFILE_LABELS = {
    "high_coverage": "High Coverage",
    "conservative": "Conservative",
    "balanced": "Balanced",
    "high_recall": "High Recall",
}

PROFILE_OVERRIDE_FILES = {
    "high_coverage": Path("configs/overrides/high_coverage.yaml"),
    "conservative": Path("configs/overrides/conservative.yaml"),
    "balanced": Path("configs/overrides/balanced.yaml"),
    "high_recall": Path("configs/overrides/high_recall.yaml"),
}

PROFILE_SCAN_DIR_SUFFIX = "profile_scan"
LEGACY_PROFILE_SCAN_DIR_SUFFIX = "rule_screen"
FINAL_DIR_SUFFIX = "final"
LEGACY_FINAL_DIR_SUFFIX = "overnight"

PROFILE_COMPARISON_FILENAMES = (
    "profile_comparison.csv",
    "rule_regime_comparison.csv",
    "profile_comparison.json",
    "rule_regime_comparison.json",
)
FINAL_RUN_MANIFEST_FILENAMES = (
    "final_run_manifest.json",
    "overnight_manifest.json",
)
FINAL_RUN_NAME_FIELDS = (
    "final_run_name",
    "overnight_run_name",
)

DEFAULT_FIRST_RUN_NAME = "first_run_check"
LEGACY_FIRST_RUN_NAMES = ("team_first_run",)

DEFAULT_FIRST_RUN_OVERRIDE = Path("configs/overrides/first_run_repro.yaml")
LEGACY_FIRST_RUN_OVERRIDE = Path("configs/overrides/broad_signal_repro.yaml")

DEFAULT_FIRST_RUN_REFERENCE = Path("docs/reference/first_run_reference.json")
LEGACY_FIRST_RUN_REFERENCE = Path("docs/reference/broad_signal_repro_reference.json")
DEFAULT_REFERENCE_RUN_NAME = "reference_run_high_coverage_final"
LEGACY_REFERENCE_RUN_NAMES = ("broad_signal_validation_intraday_broad_signal_overnight",)


def canonical_profile_names() -> tuple[str, ...]:
    return CANONICAL_PROFILE_ORDER


def supported_profile_names() -> tuple[str, ...]:
    return tuple(sorted(PROFILE_NAME_ALIASES))


def canonicalize_profile_name(profile_name: str | None) -> str | None:
    if profile_name is None:
        return None
    normalized = str(profile_name).strip()
    if not normalized:
        return None
    return PROFILE_NAME_ALIASES.get(normalized, normalized)


def profile_label(profile_name: str | None) -> str | None:
    canonical = canonicalize_profile_name(profile_name)
    if canonical is None:
        return None
    return PROFILE_LABELS.get(canonical, canonical.replace("_", " ").title())


def profile_override_path(profile_name: str) -> Path:
    canonical = canonicalize_profile_name(profile_name)
    if canonical is None or canonical not in PROFILE_OVERRIDE_FILES:
        raise KeyError(f"Unsupported profile name: {profile_name}")
    return PROFILE_OVERRIDE_FILES[canonical]


def profile_aliases(profile_name: str) -> tuple[str, ...]:
    canonical = canonicalize_profile_name(profile_name)
    if canonical is None:
        return tuple()
    ordered = [canonical]
    for alias, target in PROFILE_NAME_ALIASES.items():
        if target == canonical and alias != canonical:
            ordered.append(alias)
    return tuple(ordered)


def profile_scan_dir_name(base_run_name: str) -> str:
    return f"{base_run_name}_{PROFILE_SCAN_DIR_SUFFIX}"


def legacy_profile_scan_dir_name(base_run_name: str) -> str:
    return f"{base_run_name}_{LEGACY_PROFILE_SCAN_DIR_SUFFIX}"


def candidate_profile_scan_dir_names(base_run_name: str) -> tuple[str, ...]:
    return (
        profile_scan_dir_name(base_run_name),
        legacy_profile_scan_dir_name(base_run_name),
    )


def final_run_dir_name(base_run_name: str, profile_name: str) -> str:
    canonical = canonicalize_profile_name(profile_name)
    if canonical is None:
        raise ValueError("profile_name is required for final runs")
    return f"{base_run_name}_{canonical}_{FINAL_DIR_SUFFIX}"


def candidate_final_run_dir_names(base_run_name: str, profile_name: str) -> tuple[str, ...]:
    names: list[str] = []
    for alias in profile_aliases(profile_name):
        names.append(f"{base_run_name}_{alias}_{FINAL_DIR_SUFFIX}")
        names.append(f"{base_run_name}_{alias}_{LEGACY_FINAL_DIR_SUFFIX}")
    return tuple(dict.fromkeys(names))


def candidate_final_run_globs(base_run_name: str) -> tuple[str, ...]:
    return (
        f"{base_run_name}_*_{FINAL_DIR_SUFFIX}/pipeline_manifest.json",
        f"{base_run_name}_*_{LEGACY_FINAL_DIR_SUFFIX}/pipeline_manifest.json",
    )


def resolve_final_run_name_field(payload: dict[str, object]) -> str | None:
    for field in FINAL_RUN_NAME_FIELDS:
        value = payload.get(field)
        if value:
            return str(value)
    return None


def canonicalize_winner_profile(profile_name: str | None) -> str | None:
    return canonicalize_profile_name(profile_name)


def candidate_first_run_reference_paths(repo_root: Path | None = None) -> tuple[Path, ...]:
    paths = (
        DEFAULT_FIRST_RUN_REFERENCE,
        LEGACY_FIRST_RUN_REFERENCE,
    )
    if repo_root is None:
        return paths
    return tuple(repo_root / path for path in paths)


def candidate_first_run_override_paths() -> tuple[Path, ...]:
    return (
        DEFAULT_FIRST_RUN_OVERRIDE,
        LEGACY_FIRST_RUN_OVERRIDE,
    )


def accepted_reference_run_aliases() -> tuple[str, ...]:
    return (DEFAULT_REFERENCE_RUN_NAME, *LEGACY_REFERENCE_RUN_NAMES)
