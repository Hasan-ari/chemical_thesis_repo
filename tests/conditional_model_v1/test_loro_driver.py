"""Tests for the leave-one-rock-out driver, its bookkeeping, plots and configs."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from conditional_model_v1 import loro
from conditional_model_v1.cli import loro as loro_cli
from conditional_model_v1.config import load_config, parse_config
from conditional_model_v1.data import OUTPUT_FEATURES
from conditional_model_v1.plotting import plot_loro_feature_heatmap, plot_loro_rock_bars

ROCKS = ("Calcite", "Dolomite", "Halite")


def _loro_config(tmp_dir: Path, held_out_rocks="all"):
    return parse_config(
        {
            "experiment": {"name": "loro_driver_test", "run_root": str(tmp_dir / "runs")},
            "data": {
                "processed_root": str(tmp_dir / "processed"),
                "datasets": [
                    {"name": rock.lower(), "rock": rock, "path": str(tmp_dir / rock)}
                    for rock in ROCKS
                ],
                "split": {
                    "strategy": "leave_one_rock_out",
                    "held_out_rocks": held_out_rocks,
                    "train": 0.8,
                    "val": 0.1,
                    "test": 0.1,
                    "seed": 42,
                },
            },
            "model": {"hidden_size": 4, "num_layers": 1},
            "training": {"epochs": 1, "batch_size": 2},
        }
    )


def _fake_fold_run(run_root: Path, fold_name: str, held_out_rock: str, *, seen: bool = True) -> Path:
    """Write the files a real ``run_training`` fold leaves behind."""
    run_dir = run_root / f"20260903_000000_{fold_name}"
    overviews = run_dir / "plots" / "rock_overviews"
    overviews.mkdir(parents=True)
    unseen = {
        "n_runs": 4,
        "rmse_mean_original": 0.5,
        "rmse_mean_normalized": 1.5,
        "mae_mean_original": 0.4,
    }
    metrics = {
        "held_out_rock": held_out_rock,
        "split_strategy": "leave_one_rock_out",
        "test_unseen": unseen,
        "test_seen": {"n_runs": 2, "rmse_mean_original": 0.1, "rmse_mean_normalized": 0.3}
        if seen
        else None,
        "n_train_runs": 6,
        "n_val_runs": 1,
        "n_test_runs": 6 if seen else 4,
        "best_val_loss": 0.02,
        "per_rock": {held_out_rock: {"rmse_mean_original": 0.5, "rmse_mean_normalized": 1.5}},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    with (run_dir / "rock_feature_metrics.csv").open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj, fieldnames=["rock", "feature", "n_runs", "rmse_original", "rmse_normalized"]
        )
        writer.writeheader()
        for rock in (held_out_rock, "Other"):
            for index, feature in enumerate(OUTPUT_FEATURES):
                writer.writerow(
                    {
                        "rock": rock,
                        "feature": feature,
                        "n_runs": 4,
                        "rmse_original": 0.1 * index,
                        "rmse_normalized": 0.01 * index,
                    }
                )
    for kind in loro.HELD_OUT_OVERVIEW_KINDS:
        (overviews / f"{held_out_rock}_{kind}_overview.png").write_bytes(b"png")
    _write_run_rmse(run_dir, {held_out_rock: 4, "Other": 2})
    return run_dir


def _write_run_rmse(run_dir: Path, counts: dict[str, int]) -> None:
    with (run_dir / "run_rmse.csv").open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj, fieldnames=["run_id", "rock", "rmse_normalized", "rmse_original"]
        )
        writer.writeheader()
        for rock, count in counts.items():
            for index in range(count):
                writer.writerow(
                    {"run_id": f"{rock}_{index}", "rock": rock,
                     "rmse_normalized": 1.0 + 0.1 * index, "rmse_original": 0.5}
                )


def _fake_reference_run(run_root: Path) -> Path:
    run_dir = run_root / "20260903_000000_reference"
    run_dir.mkdir(parents=True)
    metrics = {
        "held_out_rock": None,
        "per_rock": {
            rock: {"rmse_mean_original": 0.2, "rmse_mean_normalized": 0.5} for rock in ROCKS
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    _write_run_rmse(run_dir, {rock: 3 for rock in ROCKS})
    return run_dir


class ResolveHeldOutRocksTests(unittest.TestCase):
    def test_all_keeps_dataset_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _loro_config(Path(tmp_dir))
        self.assertEqual(loro.resolve_held_out_rocks(config), ROCKS)

    def test_config_subset_and_cli_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _loro_config(Path(tmp_dir), held_out_rocks=["Halite", "Calcite"])
        self.assertEqual(loro.resolve_held_out_rocks(config), ("Calcite", "Halite"))
        self.assertEqual(loro.resolve_held_out_rocks(config, ("Dolomite",)), ("Dolomite",))
        with self.assertRaisesRegex(ValueError, "missing from data.datasets"):
            loro.resolve_held_out_rocks(config, ("Granite",))

    def test_fold_config_only_changes_the_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _loro_config(Path(tmp_dir))
        fold = loro.fold_config(config, "Calcite")
        self.assertEqual(fold.name, "loro_driver_test_loro_Calcite")
        self.assertEqual(fold.data, config.data)
        self.assertEqual(fold.training, config.training)


class RunLoroTests(unittest.TestCase):
    def _run(self, tmp_dir: Path, *, reference: bool, twice: bool = False) -> Path:
        config = _loro_config(tmp_dir, held_out_rocks=["Calcite", "Dolomite"])
        config_path = tmp_dir / "loro.yaml"
        config_path.write_text("experiment: {name: loro_driver_test}\n")
        run_root = Path(config.run_root)
        reference_dir = _fake_reference_run(run_root) if reference else None

        def fake_run_training(*, config, config_path, held_out_rock):
            self.assertEqual(config.name, f"loro_driver_test_loro_{held_out_rock}")
            return _fake_fold_run(run_root, config.name, held_out_rock, seen=reference)

        with mock.patch.object(loro_cli, "run_training", side_effect=fake_run_training) as patched:
            loro_dir = loro_cli.run_loro(
                config=config, config_path=config_path, reference_run_dir=reference_dir
            )
            if twice:
                # Same fold folders again (re-run after a Colab crash).
                for fold_dir in run_root.glob("*_loro_driver_test_loro_*"):
                    for path in sorted(fold_dir.rglob("*"), reverse=True):
                        path.unlink() if path.is_file() else path.rmdir()
                    fold_dir.rmdir()
                loro_cli.run_loro(config=config, config_path=config_path, reference_run_dir=reference_dir)
        self.assertEqual(patched.call_count, 4 if twice else 2)
        return loro_dir

    def test_two_folds_produce_summary_sqlite_features_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            loro_dir = self._run(Path(tmp_dir), reference=True)

            with (loro_dir / "loro_summary.csv").open(newline="") as file_obj:
                summary = list(csv.DictReader(file_obj))
            self.assertEqual([row["held_out_rock"] for row in summary], ["Calcite", "Dolomite"])
            self.assertEqual(summary[0]["n_test_unseen"], "4")
            self.assertEqual(summary[0]["seen_rmse_mean_normalized"], "0.3")
            self.assertEqual(summary[0]["reference_rmse_mean_normalized"], "0.5")
            self.assertAlmostEqual(float(summary[0]["unseen_over_reference_normalized"]), 3.0)

            with (loro_dir / "loro_rock_feature_metrics.csv").open(newline="") as file_obj:
                features = list(csv.DictReader(file_obj))
            self.assertEqual(len(features), 2 * len(OUTPUT_FEATURES))
            self.assertEqual({row["held_out_rock"] for row in features}, {"Calcite", "Dolomite"})

            payload = json.loads((loro_dir / "loro_metrics.json").read_text())
            self.assertEqual(payload["held_out_rocks"], ["Calcite", "Dolomite"])
            self.assertEqual(len(payload["folds"]), 2)
            self.assertTrue((loro_dir / "config.yaml").is_file())
            self.assertTrue((loro_dir / "loro_config.json").is_file())

            plots = loro_dir / "plots"
            for name in (
                "loro_rmse_normalized_by_rock.png",
                "loro_rmse_original_by_rock.png",
                "loro_unseen_rmse_normalized_heatmap.png",
                "loro_unseen_rmse_boxplot.png",
            ):
                self.assertTrue((plots / name).is_file(), name)
            with (loro_dir / "loro_run_rmse.csv").open(newline="") as file_obj:
                run_rmse = list(csv.DictReader(file_obj))
            self.assertEqual(len(run_rmse), 8)  # 4 unseen runs x 2 folds
            copied = sorted(path.name for path in (plots / "rock_overviews").iterdir())
            self.assertEqual(len(copied), 6)
            self.assertIn("Dolomite_worst_overview.png", copied)

            with sqlite3.connect(Path(loro_dir.parent) / "registry.sqlite") as connection:
                rows = connection.execute(
                    "SELECT loro_name, held_out_rock, n_test_unseen, unseen_rmse_mean_normalized, "
                    "seen_rmse_mean_normalized FROM loro_folds ORDER BY held_out_rock"
                ).fetchall()
            self.assertEqual(
                rows,
                [(loro_dir.name, "Calcite", 4, 1.5, 0.3), (loro_dir.name, "Dolomite", 4, 1.5, 0.3)],
            )

    def test_without_reference_and_seen_split_columns_are_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            loro_dir = self._run(Path(tmp_dir), reference=False)
            with (loro_dir / "loro_summary.csv").open(newline="") as file_obj:
                summary = list(csv.DictReader(file_obj))
            self.assertEqual(summary[0]["seen_rmse_mean_normalized"], "")
            self.assertEqual(summary[0]["reference_rmse_mean_normalized"], "")
            self.assertEqual(summary[0]["unseen_over_reference_normalized"], "")
            self.assertEqual(summary[0]["n_test_seen"], "0")
            self.assertTrue((loro_dir / "plots" / "loro_rmse_normalized_by_rock.png").is_file())

    def test_rerun_appends_a_second_loro_dir_without_duplicating_sqlite_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch("conditional_model_v1.cli.loro.datetime") as fake_datetime:
                fake_datetime.now.return_value.strftime.side_effect = ["20260903_000001"] * 2
                loro_dir = self._run(Path(tmp_dir), reference=True, twice=False)
            with sqlite3.connect(loro_dir.parent / "registry.sqlite") as connection:
                before = connection.execute("SELECT COUNT(*) FROM loro_folds").fetchone()[0]
                # Re-recording the same loro_name replaces instead of duplicating.
                from conditional_model_v1.tracking import record_loro_folds

                folds = json.loads((loro_dir / "loro_metrics.json").read_text())["folds"]
                record_loro_folds(
                    loro_dir.parent / "registry.sqlite",
                    loro_name=loro_dir.name,
                    loro_dir=loro_dir,
                    folds=folds,
                )
                after = connection.execute("SELECT COUNT(*) FROM loro_folds").fetchone()[0]
            self.assertEqual((before, after), (2, 2))

    def test_rejects_non_loro_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = parse_config(
                {
                    "experiment": {"name": "plain", "run_root": tmp_dir},
                    "data": {
                        "processed_root": tmp_dir,
                        "datasets": [{"name": "c", "rock": "Calcite", "path": tmp_dir}],
                    },
                }
            )
            with self.assertRaisesRegex(ValueError, "leave_one_rock_out"):
                loro_cli.run_loro(config=config, config_path=Path(tmp_dir) / "x.yaml")

    def test_read_fold_result_checks_the_rock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = _fake_fold_run(Path(tmp_dir), "fold", "Calcite")
            with self.assertRaisesRegex(ValueError, "held_out_rock"):
                loro.read_fold_result(run_dir, "Dolomite")
            result = loro.read_fold_result(run_dir, "Calcite")
            self.assertEqual(result.n_test_unseen, 4)
            self.assertEqual(result.n_test_seen, 2)


class LoroPlotTests(unittest.TestCase):
    def test_bar_plots_write_two_pngs(self) -> None:
        rows = [
            {
                "held_out_rock": rock,
                "unseen_rmse_mean_normalized": 1.0,
                "seen_rmse_mean_normalized": 0.3,
                "reference_rmse_mean_normalized": None,
                "unseen_rmse_mean_original": 0.5,
                "seen_rmse_mean_original": 0.1,
                "reference_rmse_mean_original": None,
            }
            for rock in ROCKS
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            normalized = Path(tmp_dir) / "n.png"
            original = Path(tmp_dir) / "o.png"
            plot_loro_rock_bars(rows=rows, path_normalized=normalized, path_original=original)
            self.assertGreater(normalized.stat().st_size, 0)
            self.assertGreater(original.stat().st_size, 0)
        with self.assertRaisesRegex(ValueError, "at least one fold"):
            plot_loro_rock_bars(rows=[], path_normalized=normalized, path_original=original)

    def test_heatmap_writes_png_and_rejects_incomplete_grid(self) -> None:
        feature_rows = [
            {"held_out_rock": rock, "feature": feature, "rmse_normalized": 0.1}
            for rock in ROCKS
            for feature in OUTPUT_FEATURES
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "heat.png"
            plot_loro_feature_heatmap(
                feature_rows=feature_rows, rocks=ROCKS, output_features=OUTPUT_FEATURES, path=path
            )
            self.assertGreater(path.stat().st_size, 0)
            with self.assertRaisesRegex(ValueError, "missing"):
                plot_loro_feature_heatmap(
                    feature_rows=feature_rows[:-1],
                    rocks=ROCKS,
                    output_features=OUTPUT_FEATURES,
                    path=path,
                )
            with self.assertRaisesRegex(ValueError, "Unexpected heatmap cell"):
                plot_loro_feature_heatmap(
                    feature_rows=feature_rows + [{"held_out_rock": "Granite", "feature": OUTPUT_FEATURES[0], "rmse_normalized": 0.0}],
                    rocks=ROCKS,
                    output_features=OUTPUT_FEATURES,
                    path=path,
                )


class LoroConfigFileTests(unittest.TestCase):
    def test_colab_loro_config_parses_and_matches_reference_run(self) -> None:
        root = Path(__file__).resolve().parents[2] / "configs" / "conditional_model_v1"
        env = {"RUN_ROOT": "/tmp/r", "PROCESSED_ROOT": "/tmp/p", "DATA_ROOT": "/tmp/d"}
        with mock.patch.dict(os.environ, env):
            loro_config = load_config(root / "loro_colab_eight_rocks.yaml")
            reference = load_config(root / "full_colab_eight_rocks.yaml")
        self.assertEqual(loro_config.data.split.strategy, "leave_one_rock_out")
        self.assertEqual(loro_config.data.split.held_out_rocks, "all")
        self.assertEqual(loro_config.name, "eight_rocks_loro_v1")
        self.assertEqual(loro_config.data.datasets, reference.data.datasets)
        self.assertEqual(loro_config.data.cache_name, reference.data.cache_name)
        self.assertEqual(loro_config.model, reference.model)
        self.assertEqual(loro_config.training, reference.training)
        self.assertEqual(len(loro.resolve_held_out_rocks(loro_config)), 8)

    def test_local_smoke_loro_config_parses(self) -> None:
        root = Path(__file__).resolve().parents[2] / "configs" / "conditional_model_v1"
        config = load_config(root / "smoke_loro_local.yaml")
        self.assertEqual(loro.resolve_held_out_rocks(config), ("Calcite", "Dolomite"))


if __name__ == "__main__":
    unittest.main()
