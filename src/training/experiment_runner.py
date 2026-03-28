from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch

from src.data.loader import load_all_trajectories, load_time_axis
from src.data.split import split_trajectories
from src.data.preprocessing import DataPreprocessor
from src.data.dataset import create_dataloader
from src.models.lstm import PhreeqcLSTM
from src.training.config import ExperimentConfig
from src.training.trainer import Trainer, setup_device
from src.evaluation.autoregressive import evaluate_on_test_set
from src.evaluation.plotting import (
    plot_trajectory_comparison,
    plot_loss_curve,
    plot_experiment_matrix_heatmap,
)
from src.data.constants import FEATURE_NAMES


class ExperimentRunner:
    """Orchestrates loading data once, then running multiple configs."""

    def __init__(self, base_output_dir: str = "experiments"):
        self.base_output_dir = Path(base_output_dir)
        self.device = setup_device()

    def run_matrix(
        self,
        configs: List[ExperimentConfig],
        skip_existing: bool = True,
    ) -> dict:
        """Run all experiments in the config list.

        Data loading, split, and preprocessing happen once.
        Each config gets its own subdirectory.
        """
        print("=" * 70)
        print("LSTM EXPERIMENT MATRIX")
        print(f"Configs: {len(configs)}")
        print("=" * 70)

        # Load data once
        raw_data = load_all_trajectories(
            Path(configs[0].data_dir) / "output"
        )
        time_axis = load_time_axis(Path(configs[0].data_dir) / "output")

        # Split once (same split for all experiments)
        train_raw, test_raw, train_idx, test_idx = split_trajectories(
            raw_data, test_ratio=configs[0].test_ratio, seed=configs[0].seed
        )

        # Fit preprocessor once on training data
        preprocessor = DataPreprocessor(log_cols=configs[0].log_cols)
        train_norm = preprocessor.fit_transform(train_raw)
        test_norm = preprocessor.transform(test_raw)

        all_results = {}

        for i, config in enumerate(configs):
            exp_dir = self.base_output_dir / config.experiment_name

            if skip_existing and (exp_dir / "results.json").exists():
                print(f"\n[{i+1}/{len(configs)}] SKIP {config.experiment_name} (exists)")
                with open(exp_dir / "results.json") as f:
                    all_results[config.experiment_name] = json.load(f)
                continue

            print(f"\n{'='*70}")
            print(f"[{i+1}/{len(configs)}] {config.experiment_name}")
            print(f"  seq_len={config.seq_len}, hidden={config.hidden_size}")
            print(f"{'='*70}")

            result = self._run_single(
                config, train_norm, test_norm, train_raw, test_raw,
                preprocessor, time_axis, exp_dir,
            )
            all_results[config.experiment_name] = result

        # Save summary
        self._save_summary(all_results, configs)
        return all_results

    def run_single(self, config: ExperimentConfig) -> dict:
        """Run one experiment end-to-end (standalone)."""
        raw_data = load_all_trajectories(Path(config.data_dir) / "output")
        time_axis = load_time_axis(Path(config.data_dir) / "output")

        train_raw, test_raw, _, _ = split_trajectories(
            raw_data, test_ratio=config.test_ratio, seed=config.seed
        )

        preprocessor = DataPreprocessor(log_cols=config.log_cols)
        train_norm = preprocessor.fit_transform(train_raw)
        test_norm = preprocessor.transform(test_raw)

        exp_dir = self.base_output_dir / config.experiment_name
        return self._run_single(
            config, train_norm, test_norm, train_raw, test_raw,
            preprocessor, time_axis, exp_dir,
        )

    def _run_single(
        self,
        config: ExperimentConfig,
        train_norm: np.ndarray,
        test_norm: np.ndarray,
        train_raw: np.ndarray,
        test_raw: np.ndarray,
        preprocessor: DataPreprocessor,
        time_axis: np.ndarray,
        exp_dir: Path,
    ) -> dict:
        """Internal: run one experiment with pre-loaded data."""
        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Create DataLoader
        train_loader = create_dataloader(
            train_norm, config.seq_len,
            batch_size=config.batch_size, shuffle=True,
        )

        # Build model
        model = PhreeqcLSTM(
            n_features=config.n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {n_params:,}")

        # Train
        trainer = Trainer(
            model, train_loader, self.device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            grad_clip=config.grad_clip,
        )
        train_result = trainer.train(
            epochs=config.epochs,
            print_every=max(1, config.epochs // 10),
            save_dir=exp_dir,
        )

        # Load best model for evaluation
        model.load_state_dict(torch.load(
            exp_dir / "best_model.pt", weights_only=True, map_location=self.device
        ))
        model.to(self.device)

        # Evaluate
        print("  Evaluating on test set...")
        eval_result = evaluate_on_test_set(
            model, test_norm, test_raw, preprocessor,
            config.seq_len, self.device,
        )

        print(f"  NRMSE (normalized): {eval_result['nrmse_total']:.6f}")
        print(f"  RMSE per var (original scale):")
        for name, val in zip(FEATURE_NAMES, eval_result["rmse_per_var"]):
            print(f"    {name:12s}: {val:.6f}")

        # Save results
        result = {
            "timestamp": datetime.now().isoformat(),
            "seq_len": config.seq_len,
            "hidden_size": config.hidden_size,
            "rmse_total": eval_result["nrmse_total"],
            "rmse_per_var": dict(zip(FEATURE_NAMES, eval_result["rmse_per_var"])),
            "final_loss": train_result.final_loss,
            "best_loss": train_result.best_loss,
            "training_time": train_result.training_time_seconds,
            "n_params": n_params,
            "loss_history": train_result.loss_history,
        }

        exp_dir.mkdir(parents=True, exist_ok=True)
        config.save(exp_dir / "config.json")
        with open(exp_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)

        # Plot loss curve
        plot_loss_curve(
            train_result.loss_history,
            save_path=exp_dir / "loss_curve.png",
            title=f"Loss: {config.experiment_name}",
        )

        # Plot best and worst test trajectory (selected by normalized RMSE)
        rmse_per_traj = eval_result["nrmse_per_trajectory"]
        for label, traj_idx in [
            ("best", int(np.argmin(rmse_per_traj))),
            ("worst", int(np.argmax(rmse_per_traj))),
        ]:
            plot_trajectory_comparison(
                pred=eval_result["trajectories_pred"][traj_idx],
                truth=test_raw[traj_idx],
                time_axis=time_axis,
                seq_len=config.seq_len,
                save_path=exp_dir / f"trajectory_{label}.png",
                title=f"{config.experiment_name} — {label} test trajectory",
            )

        return result

    def _save_summary(self, all_results: dict, configs: list) -> None:
        """Save summary JSON and heatmap."""
        summary_path = self.base_output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved: {summary_path}")

        # Heatmap
        if len(all_results) > 1:
            plot_experiment_matrix_heatmap(
                all_results,
                save_path=self.base_output_dir / "rmse_heatmap.png",
            )

        # Print summary table
        print("\n" + "=" * 70)
        print("EXPERIMENT MATRIX RESULTS")
        print("=" * 70)
        print(f"{'Experiment':<20} {'RMSE':>10} {'Loss':>12} {'Time':>8}")
        print("-" * 55)
        for name, r in sorted(all_results.items()):
            t = r.get("training_time", 0)
            print(
                f"{name:<20} {r['rmse_total']:>10.6f} "
                f"{r['best_loss']:>12.2e} {t:>7.1f}s"
            )
