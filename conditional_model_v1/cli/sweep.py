from __future__ import annotations

import argparse
from pathlib import Path

from conditional_model_v1.cli.train import run_training
from conditional_model_v1.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple conditional_model_v1 configs")
    parser.add_argument("--configs", nargs="+", required=True, help="YAML config paths")
    args = parser.parse_args()

    run_dirs: list[Path] = []
    for config_name in args.configs:
        config_path = Path(config_name)
        print(f"\n=== running {config_path} ===", flush=True)
        run_dirs.append(run_training(config=load_config(config_path), config_path=config_path))

    print("\nCompleted runs:")
    for run_dir in run_dirs:
        print(run_dir)


if __name__ == "__main__":
    main()
