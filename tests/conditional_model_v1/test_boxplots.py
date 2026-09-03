from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from conditional_model_v1.config import parse_config
from conditional_model_v1.metrics import per_run_rmse
from conditional_model_v1.plotting import plot_loro_unseen_boxplot, plot_rmse_boxplots
from conditional_model_v1.tracking import ExperimentTracker

FEATURES = ("a", "b", "c")


def _arrays():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(7, 5, 3)).astype(np.float32)
    y_pred = y_true + rng.normal(scale=0.1, size=y_true.shape).astype(np.float32)
    rocks = np.asarray(["Calcite"] * 4 + ["Halite"] * 3, dtype=np.str_)
    run_ids = [f"r{i}" for i in range(7)]
    return y_true, y_pred, rocks, run_ids


class PerRunRmseTests(unittest.TestCase):
    def test_rows_match_direct_computation_and_chunking_is_invisible(self) -> None:
        y_true, y_pred, rocks, run_ids = _arrays()
        rows, feature_rmse = per_run_rmse(
            y_true=y_true, y_pred=y_pred, y_true_norm=y_true * 2, y_pred_norm=y_pred * 2,
            rocks=rocks, run_ids=run_ids, chunk_runs=2,
        )
        self.assertEqual(len(rows), 7)
        self.assertEqual(feature_rmse.shape, (7, 3))
        direct = np.sqrt(np.mean((y_pred[3].astype(np.float64) - y_true[3]) ** 2))
        self.assertAlmostEqual(rows[3]["rmse_original"], float(direct), places=10)
        self.assertAlmostEqual(rows[3]["rmse_normalized"], float(direct * 2), places=10)
        self.assertEqual(rows[3]["rock"], "Calcite")
        self.assertEqual(rows[6]["run_id"], "r6")
        direct_feature = np.sqrt(np.mean(((y_pred[0] - y_true[0]).astype(np.float64) * 2) ** 2, axis=0))
        np.testing.assert_allclose(feature_rmse[0], direct_feature, rtol=1e-6)

    def test_misaligned_metadata_is_rejected(self) -> None:
        y_true, y_pred, rocks, run_ids = _arrays()
        with self.assertRaisesRegex(ValueError, "align"):
            per_run_rmse(y_true=y_true, y_pred=y_pred, y_true_norm=y_true, y_pred_norm=y_pred,
                         rocks=rocks[:-1], run_ids=run_ids)


class BoxplotTests(unittest.TestCase):
    def test_three_pngs_are_written_and_absent_rocks_are_skipped(self) -> None:
        y_true, y_pred, rocks, run_ids = _arrays()
        rows, feature_rmse = per_run_rmse(
            y_true=y_true, y_pred=y_pred, y_true_norm=y_true, y_pred_norm=y_pred,
            rocks=rocks, run_ids=run_ids,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            written = plot_rmse_boxplots(
                run_rows=rows, feature_rmse_normalized=feature_rmse, output_features=FEATURES,
                rock_order=("Calcite", "Dolomite", "Halite"), output_dir=tmp_dir,
            )
            self.assertEqual(
                [path.name for path in written],
                ["rmse_boxplot_by_rock.png", "rmse_boxplot_by_feature.png",
                 "rmse_boxplot_by_rock_and_feature.png"],
            )
            for path in written:
                self.assertGreater(path.stat().st_size, 0)
            with self.assertRaisesRegex(ValueError, "n_runs, n_outputs"):
                plot_rmse_boxplots(
                    run_rows=rows, feature_rmse_normalized=feature_rmse[:, :2],
                    output_features=FEATURES, rock_order=("Calcite",), output_dir=tmp_dir,
                )

    def test_loro_unseen_boxplot_with_and_without_reference(self) -> None:
        unseen = {"Calcite": np.array([1.0, 1.2, 0.9]), "Halite": np.array([2.0, 1.5])}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "box.png"
            plot_loro_unseen_boxplot(unseen=unseen, reference=None, rocks=("Calcite", "Halite"), path=path)
            self.assertGreater(path.stat().st_size, 0)
            plot_loro_unseen_boxplot(
                unseen=unseen, reference={"Calcite": np.array([0.5, 0.4])},
                rocks=("Calcite", "Halite"), path=path,
            )
            self.assertGreater(path.stat().st_size, 0)


class RunRmseRegistryTests(unittest.TestCase):
    def test_run_rmse_rows_land_in_csv_and_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = parse_config({
                "experiment": {"name": "rr", "run_root": tmp_dir},
                "data": {"processed_root": tmp_dir,
                         "datasets": [{"name": "c", "rock": "Calcite", "path": tmp_dir}]},
            })
            tracker = ExperimentTracker(config)
            rows = [
                {"run_id": "1", "rock": "Calcite", "rmse_normalized": 0.5, "rmse_original": 2.0},
                {"run_id": "2", "rock": "Calcite", "rmse_normalized": 0.7, "rmse_original": 3.0},
            ]
            tracker.write_run_rmse(rows)
            tracker.record_registry(config, {}, [], run_rmse_rows=rows)
            self.assertTrue(tracker.run_rmse_path.is_file())
            with sqlite3.connect(Path(tmp_dir) / "registry.sqlite") as connection:
                got = connection.execute(
                    "SELECT run_id, rock, rmse_normalized FROM run_rmse ORDER BY run_id"
                ).fetchall()
        self.assertEqual(got, [("1", "Calcite", 0.5), ("2", "Calcite", 0.7)])


if __name__ == "__main__":
    unittest.main()
