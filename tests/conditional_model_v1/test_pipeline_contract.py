from __future__ import annotations

import tempfile
import unittest
import csv
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from conditional_model_v1.cli.train import _load_or_build_bundle
from conditional_model_v1.config import parse_config
from conditional_model_v1.data import (
    CONDITION_FEATURES,
    OUTPUT_FEATURES,
    DatasetSpec,
    FullTrajectoryDataset,
    TrajectoryBundle,
    build_condition_time_tensor,
    load_cached_bundle,
    load_input_parameters,
    load_output_trajectory,
    write_processed_bundle,
)
from conditional_model_v1.models import ConditionTimeLSTM
from conditional_model_v1.plotting import plot_trajectory_examples
from conditional_model_v1.preprocessing import ConditionScaler, OutputScaler
from conditional_model_v1.splitting import rock_aware_split
from conditional_model_v1.tracking import ExperimentTracker


class PipelineContractTests(unittest.TestCase):
    def test_processed_bundle_round_trip_is_pickle_free_and_skips_empty_outputs_csv(self) -> None:
        bundle = TrajectoryBundle(
            conditions=np.arange(len(CONDITION_FEATURES), dtype=np.float64).reshape(1, -1),
            trajectories=np.arange(3 * len(OUTPUT_FEATURES), dtype=np.float64).reshape(
                1, 3, -1
            ),
            time_axis=np.array([0.0, 0.6, 1.2], dtype=np.float64),
            run_ids=["calcite:1"],
            rocks=np.array(["Calcite"]),
            condition_features=CONDITION_FEATURES,
            output_features=OUTPUT_FEATURES,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            processed_dir = Path(tmp_dir) / "processed"
            processed_dir.mkdir()
            (processed_dir / "outputs.csv").write_text("stale,data\n")
            cache_path = write_processed_bundle(
                bundle=bundle,
                inputs=pd.DataFrame(),
                outputs=pd.DataFrame(),
                inventory=pd.DataFrame(),
                processed_dir=processed_dir,
            )

            self.assertFalse((processed_dir / "outputs.csv").exists())
            with np.load(cache_path, allow_pickle=False) as payload:
                self.assertEqual(payload["run_ids"].tolist(), ["calcite:1"])
                self.assertEqual(payload["rocks"].tolist(), ["Calcite"])
                self.assertEqual(payload["condition_features"].tolist(), list(CONDITION_FEATURES))
                self.assertEqual(payload["output_features"].tolist(), list(OUTPUT_FEATURES))

            restored = load_cached_bundle(cache_path)

        np.testing.assert_array_equal(restored.conditions, bundle.conditions)
        np.testing.assert_array_equal(restored.trajectories, bundle.trajectories)
        np.testing.assert_array_equal(restored.time_axis, bundle.time_axis)
        np.testing.assert_array_equal(restored.rocks, bundle.rocks)
        self.assertEqual(restored.run_ids, bundle.run_ids)
        self.assertEqual(restored.condition_features, bundle.condition_features)
        self.assertEqual(restored.output_features, bundle.output_features)

    def test_config_supports_shared_required_cache(self) -> None:
        config = parse_config(
            {
                "experiment": {"name": "four_rock_lr_1e_3", "run_root": "/tmp/runs"},
                "data": {
                    "processed_root": "/tmp/processed",
                    "cache_name": "four_rocks_v1",
                    "require_cache": True,
                    "datasets": [
                        {"name": "calcite", "rock": "Calcite", "path": "/tmp/calcite"},
                    ],
                },
            }
        )

        self.assertEqual(config.data.cache_name, "four_rocks_v1")
        self.assertTrue(config.data.require_cache)
        self.assertEqual(config.cache_dir, Path("/tmp/processed/four_rocks_v1"))

    def test_required_cache_does_not_fall_back_to_txt_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = parse_config(
                {
                    "experiment": {"name": "four_rock_lr_1e_3", "run_root": "/tmp/runs"},
                    "data": {
                        "processed_root": str(Path(tmp_dir) / "processed"),
                        "cache_name": "four_rocks_v1",
                        "require_cache": True,
                        "datasets": [
                            {
                                "name": "calcite",
                                "rock": "Calcite",
                                "path": str(Path(tmp_dir) / "missing_raw_data"),
                            },
                        ],
                    },
                }
            )

            with mock.patch("conditional_model_v1.cli.train.build_bundle") as build_bundle_mock:
                with self.assertRaisesRegex(FileNotFoundError, "Required data cache"):
                    _load_or_build_bundle(config)
            build_bundle_mock.assert_not_called()

    def test_training_validates_shared_cache_manifest_before_loading_bundle(self) -> None:
        bundle = TrajectoryBundle(
            conditions=np.zeros((1, len(CONDITION_FEATURES)), dtype=np.float64),
            trajectories=np.zeros((1, 3, len(OUTPUT_FEATURES)), dtype=np.float64),
            time_axis=np.array([0.0, 0.6, 1.2], dtype=np.float64),
            run_ids=["Calcite_wat_sat_data_3:1"],
            rocks=np.array(["Calcite"]),
            condition_features=CONDITION_FEATURES,
            output_features=OUTPUT_FEATURES,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = parse_config(
                {
                    "experiment": {"name": "four_rock_train", "run_root": "/tmp/runs"},
                    "data": {
                        "processed_root": tmp_dir,
                        "cache_name": "four_rocks_v1",
                        "require_cache": True,
                        "datasets": [
                            {
                                "name": "Calcite_wat_sat_data_3",
                                "rock": "Calcite",
                                "path": "/offline/raw/Calcite_wat_sat_data_3",
                            },
                            {
                                "name": "Dolomite_wat_sat_data_2",
                                "rock": "Dolomite",
                                "path": "/offline/raw/Dolomite_wat_sat_data_2",
                            },
                        ],
                    },
                }
            )
            config.cache_dir.mkdir()
            cache_path = config.cache_dir / "bundle.npz"
            cache_path.touch()
            manifest = {"n_runs": 1, "n_timesteps": 3}
            with (
                mock.patch(
                    "conditional_model_v1.cli.train.validate_cache_manifest",
                    return_value=manifest,
                ) as validate_mock,
                mock.patch(
                    "conditional_model_v1.cli.train.materialize_mmap_bundle",
                    return_value=(bundle, Path(tmp_dir) / ".runtime" / "four_rocks"),
                ) as materialize_mock,
                mock.patch(
                    "conditional_model_v1.cli.train.load_cached_bundle"
                ) as eager_load_mock,
            ):
                restored = _load_or_build_bundle(config)

        self.assertIs(restored, bundle)
        validate_mock.assert_called_once()
        validation_call = validate_mock.call_args
        cache_dir = validation_call.kwargs.get(
            "cache_dir",
            validation_call.args[0] if validation_call.args else None,
        )
        specs = validation_call.kwargs.get(
            "specs",
            validation_call.args[1] if len(validation_call.args) > 1 else None,
        )
        self.assertEqual(cache_dir, config.cache_dir)
        self.assertEqual(
            [(spec.name, spec.rock, str(spec.path)) for spec in specs],
            [
                (
                    "Calcite_wat_sat_data_3",
                    "Calcite",
                    "/offline/raw/Calcite_wat_sat_data_3",
                ),
                (
                    "Dolomite_wat_sat_data_2",
                    "Dolomite",
                    "/offline/raw/Dolomite_wat_sat_data_2",
                ),
            ],
        )
        self.assertFalse(
            validate_mock.call_args.kwargs.get("verify_artifact_hashes", False)
        )
        materialize_mock.assert_called_once_with(
            cache_dir=config.cache_dir,
            manifest=manifest,
            runtime_root=Path(tmp_dir) / ".runtime",
        )
        eager_load_mock.assert_not_called()

    def test_training_rejects_manifest_shape_that_disagrees_with_loaded_bundle(self) -> None:
        bundle = TrajectoryBundle(
            conditions=np.zeros((1, len(CONDITION_FEATURES)), dtype=np.float64),
            trajectories=np.zeros((1, 3, len(OUTPUT_FEATURES)), dtype=np.float64),
            time_axis=np.array([0.0, 0.6, 1.2], dtype=np.float64),
            run_ids=["Calcite_wat_sat_data_3:1"],
            rocks=np.array(["Calcite"]),
            condition_features=CONDITION_FEATURES,
            output_features=OUTPUT_FEATURES,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = parse_config(
                {
                    "experiment": {"name": "four_rock_train", "run_root": "/tmp/runs"},
                    "data": {
                        "processed_root": tmp_dir,
                        "cache_name": "four_rocks_v1",
                        "require_cache": True,
                        "datasets": [
                            {
                                "name": "Calcite_wat_sat_data_3",
                                "rock": "Calcite",
                                "path": "/offline/raw/Calcite_wat_sat_data_3",
                            },
                        ],
                    },
                }
            )
            config.cache_dir.mkdir()
            (config.cache_dir / "bundle.npz").touch()
            mismatch_cases = (
                ({"n_runs": 2, "n_timesteps": 3}, "manifest.*n_runs"),
                ({"n_runs": 1, "n_timesteps": 4}, "manifest.*n_timesteps"),
            )
            for manifest, expected_error in mismatch_cases:
                with self.subTest(manifest=manifest):
                    with (
                        mock.patch(
                            "conditional_model_v1.cli.train.validate_cache_manifest",
                            return_value=manifest,
                        ),
                        mock.patch(
                            "conditional_model_v1.cli.train.materialize_mmap_bundle",
                            return_value=(
                                bundle,
                                Path(tmp_dir) / ".runtime" / "four_rocks",
                            ),
                        ) as materialize_mock,
                    ):
                        with self.assertRaisesRegex(ValueError, expected_error):
                            _load_or_build_bundle(config)
                        materialize_mock.assert_called_once_with(
                            cache_dir=config.cache_dir,
                            manifest=manifest,
                            runtime_root=Path(tmp_dir) / ".runtime",
                        )

    def test_training_rejects_shared_cache_contract_mismatch_before_npz_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = parse_config(
                {
                    "experiment": {"name": "four_rock_train", "run_root": "/tmp/runs"},
                    "data": {
                        "processed_root": tmp_dir,
                        "cache_name": "four_rocks_v1",
                        "require_cache": True,
                        "datasets": [
                            {
                                "name": "Calcite_wat_sat_data_3",
                                "rock": "Calcite",
                                "path": "/offline/raw/Calcite_wat_sat_data_3",
                            },
                        ],
                    },
                }
            )
            config.cache_dir.mkdir()
            (config.cache_dir / "bundle.npz").touch()

            with (
                mock.patch(
                    "conditional_model_v1.cli.train.validate_cache_manifest",
                    side_effect=ValueError("cache dataset contract mismatch"),
                ),
                mock.patch(
                    "conditional_model_v1.cli.train.load_cached_bundle"
                ) as eager_load_mock,
                mock.patch(
                    "conditional_model_v1.cli.train.materialize_mmap_bundle"
                ) as materialize_mock,
            ):
                with self.assertRaisesRegex(ValueError, "dataset contract"):
                    _load_or_build_bundle(config)

            eager_load_mock.assert_not_called()
            materialize_mock.assert_not_called()

    def test_cached_bundle_rejects_pickle_backed_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "legacy_object_cache.npz"
            np.savez_compressed(
                cache_path,
                conditions=np.array([[1.0]], dtype=np.float64),
                trajectories=np.array([[[2.0]]], dtype=np.float64),
                time_axis=np.array([0.0], dtype=np.float64),
                run_ids=np.array(["calcite:1"], dtype=object),
                rocks=np.array(["Calcite"], dtype=object),
                condition_features=np.array(["temperature"], dtype=object),
                output_features=np.array(["pH"], dtype=object),
            )

            with mock.patch("conditional_model_v1.data.np.load", wraps=np.load) as load_mock:
                with self.assertRaises(ValueError):
                    load_cached_bundle(cache_path)

            load_mock.assert_called_once_with(cache_path, allow_pickle=False)

    def test_config_rejects_conflicting_or_unsafe_cache_settings(self) -> None:
        base_payload = {
            "experiment": {"name": "four_rock", "run_root": "/tmp/runs"},
            "data": {
                "processed_root": "/tmp/processed",
                "datasets": [
                    {"name": "calcite", "rock": "Calcite", "path": "/tmp/calcite"},
                ],
            },
        }

        conflicting = {
            **base_payload,
            "data": {
                **base_payload["data"],
                "cache_name": "four_rocks_v1",
                "require_cache": True,
                "rebuild_cache": True,
            },
        }
        with self.assertRaises(ValueError) as raised:
            parse_config(conflicting)
        self.assertIn("require_cache", str(raised.exception))
        self.assertIn("rebuild_cache", str(raised.exception))

        unsafe = {
            **base_payload,
            "data": {**base_payload["data"], "cache_name": "../outside_processed_root"},
        }
        with self.assertRaisesRegex(ValueError, "cache_name"):
            parse_config(unsafe)

        mutable_shared_cache = {
            **base_payload,
            "data": {
                **base_payload["data"],
                "cache_name": "four_rocks_v1",
                "require_cache": False,
            },
        }
        with self.assertRaisesRegex(ValueError, "cache_name.*require_cache"):
            parse_config(mutable_shared_cache)

        unsafe_experiment = {
            **base_payload,
            "experiment": {"name": "../outside_processed_root", "run_root": "/tmp/runs"},
        }
        with self.assertRaisesRegex(ValueError, "experiment.name"):
            parse_config(unsafe_experiment)

    def test_config_keeps_backward_compatible_cache_defaults(self) -> None:
        config = parse_config(
            {
                "experiment": {"name": "legacy_experiment", "run_root": "/tmp/runs"},
                "data": {
                    "processed_root": "/tmp/processed",
                    "datasets": [
                        {"name": "calcite", "rock": "Calcite", "path": "/tmp/calcite"},
                    ],
                },
            }
        )

        self.assertIsNone(config.data.cache_name)
        self.assertFalse(config.data.require_cache)
        self.assertEqual(config.cache_dir, Path("/tmp/processed/legacy_experiment"))

    def test_legacy_pickle_cache_requires_explicit_rebuild(self) -> None:
        rebuilt_bundle = TrajectoryBundle(
            conditions=np.zeros((1, len(CONDITION_FEATURES)), dtype=np.float64),
            trajectories=np.zeros((1, 3, len(OUTPUT_FEATURES)), dtype=np.float64),
            time_axis=np.array([0.0, 0.6, 1.2], dtype=np.float64),
            run_ids=["calcite:1"],
            rocks=np.array(["Calcite"]),
            condition_features=CONDITION_FEATURES,
            output_features=OUTPUT_FEATURES,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = parse_config(
                {
                    "experiment": {"name": "legacy", "run_root": "/tmp/runs"},
                    "data": {
                        "processed_root": tmp_dir,
                        "datasets": [
                            {"name": "calcite", "rock": "Calcite", "path": "/tmp/calcite"},
                        ],
                    },
                }
            )
            config.cache_dir.mkdir()
            np.savez_compressed(
                config.cache_dir / "bundle.npz",
                conditions=rebuilt_bundle.conditions,
                trajectories=rebuilt_bundle.trajectories,
                time_axis=rebuilt_bundle.time_axis,
                run_ids=np.array(rebuilt_bundle.run_ids, dtype=object),
                rocks=np.array(rebuilt_bundle.rocks, dtype=object),
                condition_features=np.array(CONDITION_FEATURES, dtype=object),
                output_features=np.array(OUTPUT_FEATURES, dtype=object),
            )

            with mock.patch("conditional_model_v1.cli.train.build_bundle") as build_bundle_mock:
                with self.assertRaisesRegex(ValueError, "rebuild_cache=true"):
                    _load_or_build_bundle(config)
            build_bundle_mock.assert_not_called()

            required_payload = {
                "experiment": {"name": "legacy", "run_root": "/tmp/runs"},
                "data": {
                    "processed_root": tmp_dir,
                    "require_cache": True,
                    "datasets": [
                        {"name": "calcite", "rock": "Calcite", "path": "/tmp/calcite"},
                    ],
                },
            }
            required_config = parse_config(required_payload)
            with mock.patch("conditional_model_v1.cli.train.build_bundle") as build_bundle_mock:
                with self.assertRaises(ValueError) as raised:
                    _load_or_build_bundle(required_config)
            build_bundle_mock.assert_not_called()
            self.assertIn("new versioned cache", str(raised.exception))
            self.assertNotIn("rebuild_cache=true", str(raised.exception))

            rebuilding_payload = {
                "experiment": {"name": "legacy", "run_root": "/tmp/runs"},
                "data": {
                    "processed_root": tmp_dir,
                    "rebuild_cache": True,
                    "datasets": [
                        {"name": "calcite", "rock": "Calcite", "path": "/tmp/calcite"},
                    ],
                },
            }
            rebuilding_config = parse_config(rebuilding_payload)
            with (
                mock.patch(
                    "conditional_model_v1.cli.train.build_bundle",
                    return_value=(
                        rebuilt_bundle,
                        pd.DataFrame(),
                        pd.DataFrame(),
                        pd.DataFrame(),
                    ),
                ) as build_bundle_mock,
                mock.patch(
                    "conditional_model_v1.cli.train.write_processed_bundle"
                ) as writer_mock,
            ):
                restored = _load_or_build_bundle(rebuilding_config)

            self.assertIs(restored, rebuilt_bundle)
            build_bundle_mock.assert_called_once()
            writer_mock.assert_called_once()
            self.assertEqual(writer_mock.call_args.args[-1], rebuilding_config.cache_dir)

    def test_cached_bundle_rejects_invalid_scientific_contract(self) -> None:
        def write_cache(path: Path, **overrides) -> None:
            arrays = {
                "conditions": np.zeros((2, len(CONDITION_FEATURES)), dtype=np.float64),
                "trajectories": np.zeros((2, 3, len(OUTPUT_FEATURES)), dtype=np.float64),
                "time_axis": np.array([0.0, 0.6, 1.2], dtype=np.float64),
                "run_ids": np.array(["calcite:1", "dolomite:1"]),
                "rocks": np.array(["Calcite", "Dolomite"]),
                "condition_features": np.array(CONDITION_FEATURES),
                "output_features": np.array(OUTPUT_FEATURES),
            }
            arrays.update(overrides)
            np.savez_compressed(path, **arrays)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            duplicate_path = root / "duplicate.npz"
            write_cache(duplicate_path, run_ids=np.array(["calcite:1", "calcite:1"]))
            with self.assertRaisesRegex(ValueError, "unique"):
                load_cached_bundle(duplicate_path)

            nonfinite_path = root / "nonfinite.npz"
            trajectories = np.zeros((2, 3, len(OUTPUT_FEATURES)), dtype=np.float64)
            trajectories[1, 2, 3] = np.nan
            write_cache(nonfinite_path, trajectories=trajectories)
            with self.assertRaisesRegex(ValueError, "finite"):
                load_cached_bundle(nonfinite_path)

            time_path = root / "time.npz"
            write_cache(time_path, time_axis=np.array([0.0, 1.2, 0.6]))
            with self.assertRaisesRegex(ValueError, "increasing"):
                load_cached_bundle(time_path)

            feature_path = root / "features.npz"
            wrong_features = list(CONDITION_FEATURES)
            wrong_features[0] = "WRONG"
            write_cache(feature_path, condition_features=np.array(wrong_features))
            with self.assertRaisesRegex(ValueError, "condition feature"):
                load_cached_bundle(feature_path)

            output_feature_path = root / "output_features.npz"
            wrong_outputs = list(OUTPUT_FEATURES)
            wrong_outputs[-1] = "WRONG"
            write_cache(output_feature_path, output_features=np.array(wrong_outputs))
            with self.assertRaisesRegex(ValueError, "output feature"):
                load_cached_bundle(output_feature_path)

            nonnumeric_path = root / "nonnumeric.npz"
            write_cache(
                nonnumeric_path,
                conditions=np.full((2, len(CONDITION_FEATURES)), "not-a-number"),
            )
            with self.assertRaisesRegex(ValueError, "numeric"):
                load_cached_bundle(nonnumeric_path)

            nonnumeric_trajectory_path = root / "nonnumeric_trajectory.npz"
            write_cache(
                nonnumeric_trajectory_path,
                trajectories=np.full((2, 3, len(OUTPUT_FEATURES)), "not-a-number"),
            )
            with self.assertRaisesRegex(ValueError, "numeric"):
                load_cached_bundle(nonnumeric_trajectory_path)

            nonnumeric_time_path = root / "nonnumeric_time.npz"
            write_cache(nonnumeric_time_path, time_axis=np.array(["zero", "one", "two"]))
            with self.assertRaisesRegex(ValueError, "numeric"):
                load_cached_bundle(nonnumeric_time_path)

            empty_path = root / "empty.npz"
            write_cache(
                empty_path,
                conditions=np.empty((0, len(CONDITION_FEATURES))),
                trajectories=np.empty((0, 3, len(OUTPUT_FEATURES))),
                run_ids=np.array([], dtype=str),
                rocks=np.array([], dtype=str),
            )
            with self.assertRaisesRegex(ValueError, "empty"):
                load_cached_bundle(empty_path)

            condition_nan_path = root / "condition_nan.npz"
            conditions = np.zeros((2, len(CONDITION_FEATURES)), dtype=np.float64)
            conditions[0, 0] = np.inf
            write_cache(condition_nan_path, conditions=conditions)
            with self.assertRaisesRegex(ValueError, "finite"):
                load_cached_bundle(condition_nan_path)

            time_nan_path = root / "time_nan.npz"
            write_cache(time_nan_path, time_axis=np.array([0.0, np.nan, 1.2]))
            with self.assertRaisesRegex(ValueError, "finite"):
                load_cached_bundle(time_nan_path)

            invalid_metadata = {
                "run_ids": np.array([["calcite:1"], ["dolomite:1"]]),
                "rocks": np.array([["Calcite"], ["Dolomite"]]),
                "condition_features": np.array(CONDITION_FEATURES).reshape(-1, 1),
                "output_features": np.array(OUTPUT_FEATURES).reshape(-1, 1),
            }
            for field, invalid_value in invalid_metadata.items():
                with self.subTest(metadata_field=field):
                    metadata_shape_path = root / f"metadata_shape_{field}.npz"
                    write_cache(metadata_shape_path, **{field: invalid_value})
                    with self.assertRaisesRegex(ValueError, "one-dimensional metadata"):
                        load_cached_bundle(metadata_shape_path)

            complex_arrays = {
                "conditions": np.zeros(
                    (2, len(CONDITION_FEATURES)), dtype=np.complex128
                ),
                "trajectories": np.zeros(
                    (2, 3, len(OUTPUT_FEATURES)), dtype=np.complex128
                ),
                "time_axis": np.array([0.0, 0.6, 1.2], dtype=np.complex128),
            }
            for field, invalid_value in complex_arrays.items():
                with self.subTest(complex_field=field):
                    complex_path = root / f"complex_{field}.npz"
                    write_cache(complex_path, **{field: invalid_value})
                    with self.assertRaisesRegex(ValueError, "real numeric"):
                        load_cached_bundle(complex_path)

    def test_input_parser_maps_rock_specific_mineral_fields_to_generic_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "1_Input.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "{DATABASE} C:\\phreeqc\\database\\phreeqc.dat",
                        "{TEMPERATURE} 43.13",
                        "{POROSITY} 0.16",
                        "{WATER_VOLUME} 32.15",
                        "{GAS_VOLUME} 60.65",
                        "{SOLID_MASS} 2349.61",
                        "{PORE_VOLUME} 0.16",
                        "{ALKALINITY} 0.0126",
                        "{NA} 0.3801",
                        "{MG} 0.000104",
                        "{CL} 0.3801",
                        "{CA} 0.00232",
                        "{S6} 0.00232",
                        "{H2} 1.184",
                        "{CH4} 41.385",
                        "{CO2} 9.492",
                        "{N2} 7.933",
                        "{H2S} 0.0061",
                        "{DOLOMITE_MOLES} 12.74",
                        "{DOLOMITE_AREA} 50.35",
                    ]
                )
            )

            row = load_input_parameters(input_path, DatasetSpec(name="dolo", rock="Dolomite", path=tmp_dir))

        self.assertEqual(row["run_id"], "1")
        self.assertEqual(row["rock"], "Dolomite")
        self.assertEqual(row["mineral_moles"], 12.74)
        self.assertEqual(row["mineral_area"], 50.35)
        for feature in CONDITION_FEATURES:
            self.assertIn(feature, row)

    def test_output_parser_reads_time_axis_and_feature_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "1_Output.txt"
            output_path.write_text(
                "\n".join(
                    [
                        "time_d pH Ptot_atm pH2_atm",
                        "0.0 7.0 4.0 0.0",
                        "0.6 6.5 5.0 1.0",
                    ]
                )
            )

            frame = load_output_trajectory(output_path)

        self.assertEqual(frame["run_id"].iloc[0], "1")
        self.assertEqual(frame["time_d"].tolist(), [0.0, 0.6])
        self.assertEqual(frame["pH"].tolist(), [7.0, 6.5])

    def test_rock_aware_split_keeps_runs_disjoint_and_each_rock_represented(self) -> None:
        rocks = np.array(["Calcite"] * 10 + ["Dolomite"] * 10, dtype=object)

        split = rock_aware_split(rocks=rocks, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

        train = set(split.train.tolist())
        val = set(split.val.tolist())
        test = set(split.test.tolist())
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertEqual(train | val | test, set(range(20)))
        self.assertEqual(set(rocks[split.val]), {"Calcite", "Dolomite"})
        self.assertEqual(set(rocks[split.test]), {"Calcite", "Dolomite"})

    def test_scaling_and_dataset_contract_do_not_include_rock_label_in_x(self) -> None:
        conditions = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        time_axis = np.array([0.0, 0.6, 1.2], dtype=np.float64)
        trajectories = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)

        condition_scaler = ConditionScaler().fit(conditions[:1])
        output_scaler = OutputScaler(log_feature_indices=()).fit(trajectories[:1])

        x = build_condition_time_tensor(
            condition_scaler.transform(conditions),
            time_axis,
            time_mean=float(time_axis.mean()),
            time_std=float(time_axis.std()),
        )
        y = output_scaler.transform(trajectories)
        dataset = FullTrajectoryDataset(x=x, y=y, run_ids=["c:1", "d:1"], rocks=["Calcite", "Dolomite"])

        sample_x, sample_y = dataset[0]
        self.assertEqual(tuple(sample_x.shape), (3, 3))
        self.assertEqual(tuple(sample_y.shape), (3, 4))
        self.assertEqual(dataset.rocks[0], "Calcite")

    def test_condition_time_lstm_returns_full_trajectory_predictions(self) -> None:
        model = ConditionTimeLSTM(input_size=20, output_size=32, hidden_size=16, num_layers=1)
        x = torch.zeros((2, 301, 20), dtype=torch.float32)

        y = model(x)

        self.assertEqual(tuple(y.shape), (2, 301, 32))

    def test_plot_config_allows_null_max_runs_to_mean_every_eval_run(self) -> None:
        config = parse_config(
            {
                "experiment": {"name": "plot_all_runs", "run_root": "/tmp/runs"},
                "data": {
                    "processed_root": "/tmp/processed",
                    "datasets": [
                        {"name": "calcite", "rock": "Calcite", "path": "/tmp/calcite"},
                    ],
                },
                "plots": {"max_runs": None, "features": "all"},
            }
        )

        self.assertIsNone(config.plots.max_runs)
        self.assertEqual(config.plots.features, "all")

    def test_plotting_can_render_every_run_and_every_feature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            time_axis = np.array([0.0, 0.6, 1.2], dtype=np.float64)
            y_true = np.zeros((2, 3, 2), dtype=np.float64)
            y_pred = np.ones((2, 3, 2), dtype=np.float64)

            plot_trajectory_examples(
                time_axis=time_axis,
                y_true=y_true,
                y_pred=y_pred,
                output_features=("pH", "Ptot_atm"),
                run_ids=["calcite:1", "dolomite:1"],
                output_dir=Path(tmp_dir),
                max_runs=None,
                feature_names=None,
            )

            png_names = sorted(path.name for path in Path(tmp_dir).rglob("*.png"))

        self.assertIn("calcite_1_pH.png", png_names)
        self.assertIn("dolomite_1_Ptot_atm.png", png_names)
        self.assertIn("calcite_1_all_outputs.png", png_names)
        self.assertIn("dolomite_1_all_outputs.png", png_names)

    def test_tracker_writes_per_feature_metrics_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = parse_config(
                {
                    "experiment": {"name": "metrics_csv", "run_root": tmp_dir},
                    "data": {
                        "processed_root": "/tmp/processed",
                        "datasets": [
                            {"name": "calcite", "rock": "Calcite", "path": "/tmp/calcite"},
                        ],
                    },
                }
            )
            tracker = ExperimentTracker(config)
            self.assertTrue((tracker.run_dir / "config.json").is_file())

            tracker.write_feature_metrics(
                {
                    "rmse_per_feature_original": {"pH": 0.1, "Ptot_atm": 0.2},
                    "mae_per_feature_original": {"pH": 0.01, "Ptot_atm": 0.02},
                    "final_rmse_per_feature_original": {"pH": 0.3, "Ptot_atm": 0.4},
                    "final_mae_per_feature_original": {"pH": 0.03, "Ptot_atm": 0.04},
                }
            )

            with (tracker.run_dir / "feature_metrics.csv").open() as file_obj:
                rows = list(csv.DictReader(file_obj))

        self.assertEqual(rows[0]["feature"], "pH")
        self.assertEqual(rows[1]["feature"], "Ptot_atm")
        self.assertEqual(rows[0]["rmse_original"], "0.1")
        self.assertEqual(rows[1]["final_mae_original"], "0.04")


if __name__ == "__main__":
    unittest.main()
