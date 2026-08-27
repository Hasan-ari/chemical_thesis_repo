from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from conditional_model_v1.config import ExperimentConfig, load_config
from conditional_model_v1.data import DatasetSpec
from conditional_model_v1.preparation import prepare_cache, validate_cache_manifest


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare or validate one explicitly versioned processed-data cache."""
    parser = argparse.ArgumentParser(
        description="Prepare and validate conditional-model data caches"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Parse configured raw TXT datasets and publish the cache",
    )
    prepare_parser.add_argument("--config", required=True, help="Experiment YAML path")

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate an existing cache without reading raw TXT datasets",
    )
    validate_parser.add_argument("--config", required=True, help="Experiment YAML path")
    validate_parser.add_argument(
        "--deep",
        action="store_true",
        help="Stream and verify every artifact SHA-256 after a copy",
    )

    args = parser.parse_args(argv)
    config = load_config(Path(args.config))
    if config.data.cache_name is None:
        raise ValueError(
            "Cache commands require an explicit data.cache_name so raw data and "
            "experiments cannot silently share the wrong directory"
        )
    specs = _dataset_specs(config)

    if args.command == "prepare":
        cache_path = prepare_cache(specs=specs, cache_dir=config.cache_dir)
        print(f"cache_path={cache_path}")
        return

    manifest = validate_cache_manifest(
        cache_dir=config.cache_dir,
        specs=specs,
        verify_artifact_hashes=bool(args.deep),
    )
    print(
        f"cache_dir={config.cache_dir} "
        f"n_runs={manifest['n_runs']} n_timesteps={manifest['n_timesteps']} "
        f"deep={bool(args.deep)}"
    )


def _dataset_specs(config: ExperimentConfig) -> tuple[DatasetSpec, ...]:
    return tuple(
        DatasetSpec(
            name=dataset.name,
            rock=dataset.rock,
            path=dataset.path,
            max_runs=dataset.max_runs,
            input_units=dataset.input_units,
        )
        for dataset in config.data.datasets
    )


if __name__ == "__main__":
    main()
