from __future__ import annotations

import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from conditional_model_v1.config import parse_config
from conditional_model_v1.cli.train import _write_eval_predictions
from conditional_model_v1.data import OUTPUT_FEATURES
from conditional_model_v1.metrics import evaluate_by_rock
from conditional_model_v1.plotting import plot_rock_overviews
from conditional_model_v1.tracking import ExperimentTracker


class FourRockEvaluationTests(unittest.TestCase):
    def _evaluation_fixture(self):
        rock_order = ("Calcite", "Dolomite", "Halite", "Trona")
        rocks = np.repeat(np.asarray(rock_order, dtype=np.str_), 2)
        run_ids = [f"{rock}:{run}" for rock in rock_order for run in (1, 2)]
        shape = (len(run_ids), 3, len(OUTPUT_FEATURES))
        y_true = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100.0
        y_pred = y_true.copy()
        y_true_norm = np.zeros(shape, dtype=np.float32)
        y_pred_norm = np.zeros(shape, dtype=np.float32)
        for rock_index in range(len(rock_order)):
            first = rock_index * 2
            second = first + 1
            y_pred[first] += rock_index + 1.0
            y_pred[second] += rock_index + 2.0
            y_pred_norm[first] = 0.1
            y_pred_norm[second] = 0.2
        return (
            rock_order,
            rocks,
            run_ids,
            y_true,
            y_pred,
            y_true_norm,
            y_pred_norm,
        )

    def test_metrics_rank_runs_and_build_rock_by_feature_rows_and_overviews(self) -> None:
        (
            rock_order,
            rocks,
            run_ids,
            y_true,
            y_pred,
            y_true_norm,
            y_pred_norm,
        ) = self._evaluation_fixture()

        per_rock, rows, overviews = evaluate_by_rock(
            y_true=y_true,
            y_pred=y_pred,
            y_true_norm=y_true_norm,
            y_pred_norm=y_pred_norm,
            rocks=rocks,
            run_ids=run_ids,
            output_features=OUTPUT_FEATURES,
            rock_order=rock_order,
        )

        self.assertEqual(tuple(per_rock), rock_order)
        self.assertEqual(len(rows), 4 * len(OUTPUT_FEATURES))
        self.assertEqual(set(overviews), set(rock_order))
        for rock_index, rock in enumerate(rock_order):
            first = rock_index * 2
            second = first + 1
            self.assertEqual(per_rock[rock]["n_runs"], 2)
            self.assertEqual(per_rock[rock]["best_run_id"], run_ids[first])
            self.assertEqual(per_rock[rock]["worst_run_id"], run_ids[second])
            self.assertAlmostEqual(
                per_rock[rock]["best_run_rmse_normalized"],
                0.1,
                places=6,
            )
            self.assertEqual(
                set(per_rock[rock]["rmse_per_feature_original"]),
                set(OUTPUT_FEATURES),
            )
            self.assertEqual(
                set(per_rock[rock]["rmse_per_feature_normalized"]),
                set(OUTPUT_FEATURES),
            )
            np.testing.assert_allclose(
                overviews[rock]["mean"]["y_true"],
                np.mean(y_true[[first, second]], axis=0),
            )

    def test_plotter_requests_exactly_twelve_all_output_overviews(self) -> None:
        fixture = self._evaluation_fixture()
        rock_order, rocks, run_ids, y_true, y_pred, y_true_norm, y_pred_norm = fixture
        _metrics, _rows, overviews = evaluate_by_rock(
            y_true=y_true,
            y_pred=y_pred,
            y_true_norm=y_true_norm,
            y_pred_norm=y_pred_norm,
            rocks=rocks,
            run_ids=run_ids,
            output_features=OUTPUT_FEATURES,
            rock_order=rock_order,
        )

        def touch_overview(**kwargs):
            self.assertEqual(len(kwargs["selected_features"]), len(OUTPUT_FEATURES))
            kwargs["path"].touch()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            with mock.patch(
                "conditional_model_v1.plotting._plot_all_outputs_grid",
                side_effect=touch_overview,
            ) as plot_mock:
                plot_rock_overviews(
                    time_axis=np.array([0.0, 1.0, 2.0]),
                    output_features=OUTPUT_FEATURES,
                    overviews=overviews,
                    output_dir=output_dir,
                )
            expected_names = {
                f"{rock}_{kind}_overview.png"
                for rock in rock_order
                for kind in ("best", "worst", "mean")
            }
            self.assertEqual(
                {path.name for path in output_dir.glob("*.png")},
                expected_names,
            )
            self.assertEqual(plot_mock.call_count, 12)

    def test_tracker_writes_rock_csv_and_sqlite_rows(self) -> None:
        fixture = self._evaluation_fixture()
        rock_order, rocks, run_ids, y_true, y_pred, y_true_norm, y_pred_norm = fixture
        per_rock, rows, _overviews = evaluate_by_rock(
            y_true=y_true,
            y_pred=y_pred,
            y_true_norm=y_true_norm,
            y_pred_norm=y_pred_norm,
            rocks=rocks,
            run_ids=run_ids,
            output_features=OUTPUT_FEATURES,
            rock_order=rock_order,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = parse_config(
                {
                    "experiment": {"name": "four_rock_eval", "run_root": tmp_dir},
                    "data": {
                        "processed_root": str(Path(tmp_dir) / "processed"),
                        "datasets": [
                            {"name": rock.lower(), "rock": rock, "path": "/tmp/data"}
                            for rock in rock_order
                        ],
                    },
                }
            )
            tracker = ExperimentTracker(config)
            metrics = {
                "best_val_loss": 0.1,
                "final_train_loss": 0.2,
                "final_val_loss": 0.3,
                "rmse_mean_original": 1.0,
                "mae_mean_original": 0.5,
                "per_rock": per_rock,
            }
            tracker.write_rock_feature_metrics(rows)
            tracker.record_registry(config, metrics, rows)

            with tracker.rock_feature_metrics_path.open(newline="") as file_obj:
                csv_rows = list(csv.DictReader(file_obj))
            self.assertEqual(len(csv_rows), 4 * len(OUTPUT_FEATURES))
            with sqlite3.connect(tracker.registry_path) as connection:
                count = connection.execute(
                    "SELECT COUNT(*) FROM rock_feature_metrics"
                ).fetchone()[0]
                run_count = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            self.assertEqual(count, 4 * len(OUTPUT_FEATURES))
            self.assertEqual(run_count, 1)

    def test_eval_prediction_archive_is_pickle_free_and_metadata_aligned(self) -> None:
        fixture = self._evaluation_fixture()
        rock_order, rocks, run_ids, y_true, y_pred, y_true_norm, y_pred_norm = fixture
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / "eval_predictions.npz"
            _write_eval_predictions(
                archive_path,
                y_true=y_true,
                y_pred=y_pred,
                y_true_norm=y_true_norm,
                y_pred_norm=y_pred_norm,
                time_axis=np.array([0.0, 1.0, 2.0]),
                run_ids=run_ids,
                rocks=rocks,
                output_features=OUTPUT_FEATURES,
                eval_split="test",
            )

            with np.load(archive_path, allow_pickle=False) as archive:
                self.assertEqual(archive["y_true"].shape[0], len(run_ids))
                self.assertEqual(archive["run_ids"].tolist(), run_ids)
                self.assertEqual(archive["rocks"].tolist(), rocks.tolist())
                self.assertEqual(archive["output_features"].tolist(), list(OUTPUT_FEATURES))
                self.assertEqual(archive["eval_split"].tolist(), ["test"])
                for name in ("run_ids", "rocks", "output_features", "eval_split"):
                    self.assertIn(archive[name].dtype.kind, {"U", "S"})


if __name__ == "__main__":
    unittest.main()
