#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.paper_figures import (
    DEFAULT_PROFILE_FEATURES,
    generate_paper_figure_bundle,
    validate_repo_env_python,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper-aligned Hidden RTNN evaluation figures for this repo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Path to a saved experiment directory, e.g. experiments/seq10_h128",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <experiment-dir>/paper_figures",
    )
    parser.add_argument(
        "--experiments-root",
        default=None,
        help="Optional experiments root containing summary.json for ablation heatmaps.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_PROFILE_FEATURES),
        help="Feature names used for representative profile and breakthrough figures.",
    )
    return parser


def main() -> int:
    validate_repo_env_python(REPO_ROOT)

    parser = build_parser()
    args = parser.parse_args()

    manifest = generate_paper_figure_bundle(
        experiment_dir=args.experiment_dir,
        output_dir=args.output_dir,
        experiments_root=args.experiments_root,
        selected_features=args.features,
    )

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
