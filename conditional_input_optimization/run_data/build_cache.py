"""Build or check the RunTable cache from the command line.

Usage, from the repository root:

    python -m conditional_input_optimization.run_data.build_cache --workers 10

The script must run as a module with the __main__ guard, because macOS
starts worker processes with "spawn". Each worker imports this module again.
"""

from __future__ import annotations

import argparse
import pathlib
import time
from collections.abc import Sequence

from conditional_input_optimization.run_data import (
    rock_source,
    run_table,
    run_table_cache,
)

# All files of this work stream stay in its own folder.
_WORK_STREAM = pathlib.Path("conditional_input_optimization")
DEFAULT_CONFIG = _WORK_STREAM / "configs" / "eda_eight_rocks.yaml"
# The cache folder is in the .gitignore file of the work stream.
DEFAULT_CACHE_DIR = _WORK_STREAM / "cache" / "run_table"


def main(argv: Sequence[str] | None = None) -> None:
    """Load the cache, or build it if it is absent or stale.

    Args:
        argv: The command line arguments. None uses sys.argv.
    """
    args = _parse_args(argv)
    sources = rock_source.load_rock_sources(args.config, args.base_dir)
    cache_dir = args.base_dir / args.cache_dir
    start = time.perf_counter()
    fingerprint = run_table_cache.source_fingerprint(sources)
    was_valid = run_table_cache.is_cache_valid(cache_dir, fingerprint)
    print(f"Fingerprint: {time.perf_counter() - start:.1f} s")
    table = run_table_cache.load_or_build_run_table(
        sources, cache_dir, args.workers, args.chunk_size
    )
    action = "Loaded" if was_valid else "Built and saved"
    print(f"{action} in {time.perf_counter() - start:.1f} s: {cache_dir}")
    print(
        f"runs={len(table.rocks)} timesteps={len(table.time_days)} "
        f"conditions={table.conditions.shape} outputs={table.outputs.shape}"
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--base-dir", type=pathlib.Path, default=pathlib.Path.cwd()
    )
    parser.add_argument(
        "--cache-dir", type=pathlib.Path, default=DEFAULT_CACHE_DIR
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--chunk-size", type=int, default=run_table.DEFAULT_CHUNK_SIZE
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
