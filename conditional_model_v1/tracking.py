from __future__ import annotations

import csv
import json
import shutil
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from conditional_model_v1.config import ExperimentConfig


class ExperimentTracker:
    """Writes one self-contained run folder plus global SQLite/CSV summaries."""

    def __init__(self, config: ExperimentConfig) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(config.run_root) / f"{timestamp}_{config.name}"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.plot_dir = self.run_dir / "plots"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.run_dir / "history.csv"
        self.metrics_path = self.run_dir / "metrics.json"
        self.feature_metrics_path = self.run_dir / "feature_metrics.csv"
        self.summary_csv_path = Path(config.run_root) / "summary.csv"
        self.registry_path = Path(config.run_root) / "registry.sqlite"
        self.write_json(self.run_dir / "resolved_config.json", asdict(config))

    def copy_config(self, config_path: Path | str) -> None:
        """Keep the exact YAML file used to launch this run."""
        shutil.copy2(config_path, self.run_dir / "config.yaml")

    def write_history(self, history: list[dict[str, float | int]]) -> None:
        """Write per-epoch train/validation loss and learning rate."""
        if not history:
            return
        with self.history_path.open("w", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    def write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        self.write_json(self.metrics_path, metrics)

    def write_feature_metrics(self, metrics: dict[str, Any]) -> None:
        """Write one CSV row per output feature for fast inspection."""
        rmse = metrics.get("rmse_per_feature_original", {})
        mae = metrics.get("mae_per_feature_original", {})
        final_rmse = metrics.get("final_rmse_per_feature_original", {})
        final_mae = metrics.get("final_mae_per_feature_original", {})
        rows = [
            {
                "feature": feature,
                "rmse_original": rmse.get(feature),
                "mae_original": mae.get(feature),
                "final_rmse_original": final_rmse.get(feature),
                "final_mae_original": final_mae.get(feature),
            }
            for feature in rmse
        ]
        if not rows:
            return
        with self.feature_metrics_path.open("w", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def record_registry(self, config: ExperimentConfig, metrics: dict[str, Any]) -> None:
        """Mirror final run metadata to SQLite and CSV for comparison."""
        row = {
            "run_name": self.run_dir.name,
            "experiment": config.name,
            "run_dir": str(self.run_dir),
            "best_val_loss": metrics.get("best_val_loss"),
            "final_train_loss": metrics.get("final_train_loss"),
            "final_val_loss": metrics.get("final_val_loss"),
            "rmse_mean_original": metrics.get("rmse_mean_original"),
            "mae_mean_original": metrics.get("mae_mean_original"),
        }
        self._record_sqlite(row)
        self._record_summary_csv(row)

    def _record_sqlite(self, row: dict[str, Any]) -> None:
        with sqlite3.connect(self.registry_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_name TEXT PRIMARY KEY,
                    experiment TEXT,
                    run_dir TEXT,
                    best_val_loss REAL,
                    final_train_loss REAL,
                    final_val_loss REAL,
                    rmse_mean_original REAL,
                    mae_mean_original REAL
                )
                """
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO runs VALUES (
                    :run_name, :experiment, :run_dir, :best_val_loss,
                    :final_train_loss, :final_val_loss,
                    :rmse_mean_original, :mae_mean_original
                )
                """,
                row,
            )

    def _record_summary_csv(self, row: dict[str, Any]) -> None:
        file_exists = self.summary_csv_path.exists()
        with self.summary_csv_path.open("a", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
