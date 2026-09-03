from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from conditional_model_v1.config import parse_config
from conditional_model_v1.tracking import ExperimentTracker


class EpochHistoryRegistryTests(unittest.TestCase):
    def test_history_rows_land_in_sqlite_and_rerun_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = parse_config(
                {
                    "experiment": {"name": "hist", "run_root": tmp_dir},
                    "data": {
                        "processed_root": tmp_dir,
                        "datasets": [{"name": "c", "rock": "Calcite", "path": tmp_dir}],
                    },
                }
            )
            tracker = ExperimentTracker(config)
            history = [
                {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "lr": 1e-3},
                {"epoch": 2, "train_loss": 0.5, "val_loss": 0.4, "lr": 1e-3},
            ]
            metrics = {"best_val_loss": 0.4, "rmse_mean_original": 1.0, "mae_mean_original": 0.5}
            tracker.record_registry(config, metrics, [], history=history)
            tracker.record_registry(config, metrics, [], history=history)  # idempotent

            with sqlite3.connect(Path(tmp_dir) / "registry.sqlite") as connection:
                rows = connection.execute(
                    "SELECT epoch, train_loss, val_loss, lr FROM epoch_history "
                    "WHERE run_name = ? ORDER BY epoch",
                    (tracker.run_dir.name,),
                ).fetchall()
                n_runs = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        self.assertEqual(rows, [(1, 1.0, 0.9, 1e-3), (2, 0.5, 0.4, 1e-3)])
        self.assertEqual(n_runs, 1)

    def test_missing_history_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = parse_config(
                {
                    "experiment": {"name": "nohist", "run_root": tmp_dir},
                    "data": {
                        "processed_root": tmp_dir,
                        "datasets": [{"name": "c", "rock": "Calcite", "path": tmp_dir}],
                    },
                }
            )
            ExperimentTracker(config).record_registry(config, {}, [])
            with sqlite3.connect(Path(tmp_dir) / "registry.sqlite") as connection:
                count = connection.execute("SELECT COUNT(*) FROM epoch_history").fetchone()[0]
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
