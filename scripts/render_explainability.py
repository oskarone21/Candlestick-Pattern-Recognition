from __future__ import annotations

import argparse

try:
    from scripts._bootstrap import ensure_repo_root
except ImportError:
    from _bootstrap import ensure_repo_root

ensure_repo_root()

from chart_patterns.config import load_config
from chart_patterns.explainability import run_explainability_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render SHAP explainability artifacts for champion pattern models")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--set", dest="set_overrides", action="append", default=[])
    parser.add_argument("--run-name", default=None, help="Override the configured project.run_name")
    parser.add_argument(
        "--pattern",
        dest="patterns",
        action="append",
        default=[],
        help="Pattern to explain. May be passed multiple times or as a comma-separated list.",
    )
    parser.add_argument("--explain-split", default="test")
    parser.add_argument("--background-split", default="train")
    parser.add_argument("--explain-size", type=int, default=32)
    parser.add_argument("--background-size", type=int, default=64)
    parser.add_argument("--top-features", type=int, default=15)
    parser.add_argument("--max-waterfalls", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config, args.config_override, args.set_overrides)
    summary = run_explainability_pipeline(
        cfg,
        run_name=args.run_name,
        patterns=args.patterns,
        explain_split=args.explain_split,
        background_split=args.background_split,
        explain_size=args.explain_size,
        background_size=args.background_size,
        top_features=args.top_features,
        max_waterfalls=args.max_waterfalls,
        seed=args.seed,
    )

    print(
        "Explainability generated:",
        f"run={summary['run_name']}",
        f"succeeded={summary['patterns_succeeded']}",
        f"failed={summary['patterns_failed']}",
        f"output={summary['output_root']}",
    )


if __name__ == "__main__":
    main()
