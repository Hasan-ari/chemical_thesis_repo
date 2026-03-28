"""Main entry point for LSTM PHREEQC Surrogate Training.

Usage:
    # Run full experiment matrix (12 configs)
    python run_experiment.py --matrix

    # Run single experiment
    python run_experiment.py --seq_len 10 --hidden_size 128

    # Run with custom epochs
    python run_experiment.py --seq_len 10 --hidden_size 128 --epochs 200

    # Run from saved config
    python run_experiment.py --config experiments/seq10_h128/config.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.training.config import ExperimentConfig, build_experiment_matrix
from src.training.experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="LSTM PHREEQC Surrogate Training")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--matrix", action="store_true", help="Run full experiment matrix")
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="experiments")
    args = parser.parse_args()

    runner = ExperimentRunner(base_output_dir=args.output_dir)

    if args.matrix:
        configs = build_experiment_matrix()
        if args.epochs:
            for c in configs:
                c.epochs = args.epochs
        runner.run_matrix(configs)

    elif args.config:
        config = ExperimentConfig.load(Path(args.config))
        runner.run_single(config)

    else:
        kwargs = {"output_dir": args.output_dir}
        if args.seq_len:
            kwargs["seq_len"] = args.seq_len
        if args.hidden_size:
            kwargs["hidden_size"] = args.hidden_size
        if args.epochs:
            kwargs["epochs"] = args.epochs
        if args.batch_size:
            kwargs["batch_size"] = args.batch_size
        if args.lr:
            kwargs["learning_rate"] = args.lr
        config = ExperimentConfig(**kwargs)
        runner.run_single(config)


if __name__ == "__main__":
    main()
