from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from conditional_model_v1.cli.train import _load_or_build_bundle
from conditional_model_v1.config import parse_config
from conditional_model_v1.data import (
    CONDITION_FEATURES,
    OUTPUT_FEATURES,
    SCALAR_CONDITION_FEATURES,
    DatasetSpec,
    load_cached_bundle,
    load_input_parameters,
    load_output_trajectory,
)
from conditional_model_v1.preparation import prepare_cache, validate_cache_manifest


DATASETS: tuple[tuple[str, str], ...] = (
    ("Calcite_wat_sat_data_3", "Calcite"),
    ("Dolomite_wat_sat_data_2", "Dolomite"),
    ("Halite_wat_sat_data_2", "Halite"),
    ("Trona_par_sat_data_3", "Trona"),
)


class DataPreparationTests(unittest.TestCase):
    def test_prepare_cache_builds_one_bundle_and_sharded_csvs_for_four_rocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            specs = self._write_four_datasets(root / "raw")
            cache_dir = root / "processed" / "four_rocks_v1"

            with (
                # These spies intentionally define the preparation module's parser boundary:
                # every matched input/output TXT must pass through each shared parser once.
                mock.patch(
                    "conditional_model_v1.preparation.load_input_parameters",
                    wraps=load_input_parameters,
                ) as input_parser,
                mock.patch(
                    "conditional_model_v1.preparation.load_output_trajectory",
                    wraps=load_output_trajectory,
                ) as output_parser,
            ):
                cache_path = prepare_cache(specs, cache_dir, progress_every=0)

            self.assertEqual(input_parser.call_count, 8)
            self.assertEqual(output_parser.call_count, 8)

            self.assertEqual(cache_path, cache_dir / "bundle.npz")
            self.assertTrue((cache_dir / "_SUCCESS").is_file())
            bundle = load_cached_bundle(cache_path)
            self.assertEqual(bundle.conditions.shape, (8, len(CONDITION_FEATURES)))
            self.assertEqual(bundle.trajectories.shape, (8, 3, len(OUTPUT_FEATURES)))
            self.assertEqual(bundle.time_axis.tolist(), [0.0, 0.6, 1.2])
            self.assertEqual(
                bundle.rocks.tolist(),
                [rock for _, rock in DATASETS for _ in range(2)],
            )
            self.assertEqual(
                bundle.run_ids,
                [
                    f"{dataset_name}:{run_id}"
                    for dataset_name, _ in DATASETS
                    for run_id in (1, 2)
                ],
            )

            inputs = pd.read_csv(cache_dir / "inputs.csv")
            inventory = pd.read_csv(cache_dir / "run_inventory.csv", dtype={"run_id": str})
            self.assertEqual(len(inputs), 8)
            self.assertEqual(set(CONDITION_FEATURES), set(inputs.columns) & set(CONDITION_FEATURES))
            self.assertEqual(
                inventory.columns.tolist(),
                ["dataset", "rock", "run_id", "status", "input_path", "output_path"],
            )
            self.assertEqual(
                inventory["status"].value_counts().to_dict(),
                {"matched": 8, "failed": 4},
            )
            matched_rows = inventory[inventory["status"] == "matched"]
            self.assertEqual(matched_rows.iloc[0]["input_path"], "input/1_Input.txt")
            self.assertEqual(matched_rows.iloc[0]["output_path"], "output/1_Output.txt")
            failed_rows = inventory[inventory["status"] == "failed"]
            self.assertEqual(set(failed_rows["input_path"]), {"failed/99_Input.txt"})
            self.assertTrue((failed_rows["output_path"].fillna("") == "").all())

            for dataset_name, rock in DATASETS:
                output_path = cache_dir / "outputs" / f"{dataset_name}.csv.gz"
                self.assertTrue(output_path.is_file())
                with gzip.open(output_path, "rt") as file_obj:
                    output_frame = pd.read_csv(file_obj)
                self.assertEqual(len(output_frame), 6)
                self.assertEqual(output_frame["dataset"].unique().tolist(), [dataset_name])
                self.assertEqual(output_frame["rock"].unique().tolist(), [rock])
                self.assertEqual(output_frame["run_id"].astype(str).unique().tolist(), ["1", "2"])
                self.assertEqual(
                    output_frame.columns.tolist(),
                    ["dataset", "rock", "run_id", "timestep_index", "time_d", *OUTPUT_FEATURES],
                )

            manifest = json.loads((cache_dir / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(
                [(item["name"], item["rock"]) for item in manifest["datasets"]],
                list(DATASETS),
            )
            self.assertEqual(manifest["arrays"]["conditions"]["shape"], [8, len(CONDITION_FEATURES)])
            self.assertEqual(
                manifest["arrays"]["trajectories"]["shape"],
                [8, 3, len(OUTPUT_FEATURES)],
            )
            self.assertEqual(len(manifest["source_inventory_signature"]), 64)
            expected_artifacts = {
                "bundle.npz",
                "inputs.csv",
                "run_inventory.csv",
                *(f"outputs/{dataset_name}.csv.gz" for dataset_name, _ in DATASETS),
            }
            self.assertEqual(set(manifest["artifacts"]), expected_artifacts)
            for relative_path, metadata in manifest["artifacts"].items():
                self.assertEqual(
                    metadata["size_bytes"],
                    (cache_dir / relative_path).stat().st_size,
                )
                self.assertEqual(len(metadata["sha256"]), 64)
            self.assertFalse((cache_dir / "outputs.csv").exists())

            restored_manifest = validate_cache_manifest(
                cache_dir,
                specs,
                verify_artifact_hashes=True,
            )

        self.assertEqual(restored_manifest["n_runs"], 8)

    def test_manifest_validation_rejects_wrong_dataset_contract_and_missing_success_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            specs = self._write_four_datasets(root / "raw")
            cache_dir = root / "processed" / "four_rocks_v1"
            prepare_cache(specs, cache_dir, progress_every=0)

            with self.assertRaisesRegex(ValueError, "dataset contract"):
                validate_cache_manifest(cache_dir, specs[:-1])

            (cache_dir / "_SUCCESS").unlink()
            with self.assertRaisesRegex(ValueError, "_SUCCESS"):
                validate_cache_manifest(cache_dir, specs)

    def test_manifest_validation_rejects_missing_or_size_changed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            specs = self._write_four_datasets(root / "raw")
            cache_dir = root / "processed" / "four_rocks_v1"
            prepare_cache(specs, cache_dir, progress_every=0)
            output_path = cache_dir / "outputs" / "Calcite_wat_sat_data_3.csv.gz"

            original_bytes = output_path.read_bytes()
            output_path.unlink()
            with self.assertRaisesRegex(ValueError, "artifact"):
                validate_cache_manifest(cache_dir, specs)

            output_path.write_bytes(original_bytes + b"tampered")
            with self.assertRaisesRegex(ValueError, "size"):
                validate_cache_manifest(cache_dir, specs)

            tampered_same_size = bytearray(original_bytes)
            tampered_same_size[len(tampered_same_size) // 2] ^= 1
            output_path.write_bytes(tampered_same_size)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_cache_manifest(
                    cache_dir,
                    specs,
                    verify_artifact_hashes=True,
                )

    def test_source_inventory_signature_is_portable_and_tracks_paths_and_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first_specs = self._write_four_datasets(root / "raw_a")
            second_specs = self._write_four_datasets(root / "raw_b")
            first_dir = root / "processed" / "four_rocks_v1_a"
            second_dir = root / "processed" / "four_rocks_v1_b"
            changed_dir = root / "processed" / "four_rocks_v1_changed"
            prepare_cache(first_specs, first_dir, progress_every=0)
            prepare_cache(second_specs, second_dir, progress_every=0)
            first_manifest = json.loads((first_dir / "manifest.json").read_text())
            second_manifest = json.loads((second_dir / "manifest.json").read_text())
            self.assertEqual(
                first_manifest["source_inventory_signature"],
                second_manifest["source_inventory_signature"],
            )

            failed_path = Path(second_specs[0].path) / "failed" / "99_Input.txt"
            original_size = failed_path.stat().st_size
            failed_path.write_text("{FAILED} fail\n")
            self.assertEqual(failed_path.stat().st_size, original_size)
            content_changed_dir = root / "processed" / "four_rocks_v1_content_changed"
            prepare_cache(second_specs, content_changed_dir, progress_every=0)
            content_changed_manifest = json.loads(
                (content_changed_dir / "manifest.json").read_text()
            )
            self.assertEqual(
                first_manifest["source_inventory_signature"],
                content_changed_manifest["source_inventory_signature"],
            )

            failed_path.write_text(failed_path.read_text() + "{DETAIL} changed-size\n")
            prepare_cache(second_specs, changed_dir, progress_every=0)
            changed_manifest = json.loads((changed_dir / "manifest.json").read_text())

        self.assertNotEqual(
            first_manifest["source_inventory_signature"],
            changed_manifest["source_inventory_signature"],
        )

    def test_manifest_validation_rejects_feature_schema_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            specs = self._write_four_datasets(root / "raw")
            cache_dir = root / "processed" / "four_rocks_v1"
            prepare_cache(specs, cache_dir, progress_every=0)
            manifest_path = cache_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["condition_features"][0] = "WRONG"
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "condition feature"):
                validate_cache_manifest(cache_dir, specs)

    def test_invalid_trajectory_does_not_publish_partial_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            specs = self._write_four_datasets(root / "raw")
            bad_output = Path(specs[2].path) / "output" / "1_Output.txt"
            self._write_output(bad_output, base_value=30.0, time_axis=(0.0, 1.2, 0.6))
            processed_root = root / "processed"
            cache_dir = processed_root / "four_rocks_v1"

            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                prepare_cache(specs, cache_dir, progress_every=0)

            self.assertFalse(cache_dir.exists())
            self._assert_missing_or_empty(processed_root)

    def test_failure_after_artifacts_are_written_does_not_publish_partial_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            specs = self._write_four_datasets(root / "raw")
            processed_root = root / "processed"
            cache_dir = processed_root / "four_rocks_v1"

            def fail_publish(staging_dir: Path, final_dir: Path) -> None:
                self.assertEqual(final_dir, cache_dir)
                self.assertTrue((staging_dir / "bundle.npz").is_file())
                self.assertTrue((staging_dir / "manifest.json").is_file())
                self.assertTrue((staging_dir / "_SUCCESS").is_file())
                raise RuntimeError("injected publish failure")

            with mock.patch(
                "conditional_model_v1.preparation._publish_cache",
                side_effect=fail_publish,
            ):
                with self.assertRaisesRegex(RuntimeError, "injected publish failure"):
                    prepare_cache(specs, cache_dir, progress_every=0)

            self.assertFalse(cache_dir.exists())
            self._assert_missing_or_empty(processed_root)

    def test_existing_valid_cache_is_reused_without_parsing_txt_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            specs = self._write_four_datasets(root / "raw")
            cache_dir = root / "processed" / "four_rocks_v1"
            expected = prepare_cache(specs, cache_dir, progress_every=0)
            (root / "raw").rename(root / "raw_is_offline")

            with (
                mock.patch(
                    "conditional_model_v1.preparation.load_input_parameters",
                    side_effect=AssertionError("input TXT parser should not run"),
                ),
                mock.patch(
                    "conditional_model_v1.preparation.load_output_trajectory",
                    side_effect=AssertionError("output TXT parser should not run"),
                ),
            ):
                actual = prepare_cache(specs, cache_dir, progress_every=0)

            config = self._training_config(root, specs)
            training_bundle = _load_or_build_bundle(config)

        self.assertEqual(actual, expected)
        self.assertEqual(training_bundle.conditions.shape, (8, len(CONDITION_FEATURES)))
        self.assertEqual(
            training_bundle.trajectories.shape,
            (8, 3, len(OUTPUT_FEATURES)),
        )

    def test_training_rejects_real_cache_built_for_a_different_dataset_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            specs = self._write_four_datasets(root / "raw")
            cache_dir = root / "processed" / "four_rocks_v1"
            prepare_cache(specs, cache_dir, progress_every=0)
            (root / "raw").rename(root / "raw_is_offline")
            wrong_config = self._training_config(root, specs[:-1])

            with self.assertRaisesRegex(ValueError, "dataset contract"):
                _load_or_build_bundle(wrong_config)

    def test_each_configured_dataset_must_have_at_least_one_matched_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            specs = list(self._write_four_datasets(root / "raw"))
            empty_root = Path(specs[-1].path)
            for run_id in (1, 2):
                (empty_root / "output" / f"{run_id}_Output.txt").unlink()
            cache_dir = root / "processed" / "four_rocks_v1"

            with self.assertRaisesRegex(ValueError, "Trona_par_sat_data_3.*zero matched"):
                prepare_cache(tuple(specs), cache_dir, progress_every=0)

            self.assertFalse(cache_dir.exists())

    def test_strict_validation_rejects_nonfinite_mismatched_and_malformed_outputs(self) -> None:
        cases = ("nonfinite", "mismatched_time", "missing_output")
        for case_name in cases:
            with self.subTest(case=case_name), tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                specs = self._write_four_datasets(root / "raw")
                output_path = Path(specs[0].path) / "output" / "2_Output.txt"
                if case_name == "nonfinite":
                    self._write_output(output_path, base_value=np.nan)
                    expected_error = "finite"
                elif case_name == "mismatched_time":
                    self._write_output(
                        output_path,
                        base_value=12.0,
                        time_axis=(0.0, 0.7, 1.2),
                    )
                    expected_error = "time axis"
                else:
                    lines = output_path.read_text().splitlines()
                    lines[0] = " ".join(lines[0].split()[:-1])
                    lines[1:] = [" ".join(line.split()[:-1]) for line in lines[1:]]
                    expected_error = "missing outputs"
                    output_path.write_text("\n".join(lines) + "\n")

                processed_root = root / "processed"
                cache_dir = processed_root / "four_rocks_v1"
                with self.assertRaisesRegex(ValueError, expected_error):
                    prepare_cache(specs, cache_dir, progress_every=0)
                self.assertFalse(cache_dir.exists())
                self._assert_missing_or_empty(processed_root)

    def test_unknown_extra_output_columns_are_ignored(self) -> None:
        """Multi-mineral datasets add columns such as `Barite`; only the contract is kept."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            specs = self._write_four_datasets(root / "raw")
            output_path = Path(specs[0].path) / "output" / "2_Output.txt"
            lines = output_path.read_text().splitlines()
            lines[0] += " Barite Calcite"
            lines[1:] = [f"{line} 1.0 2.0" for line in lines[1:]]
            output_path.write_text("\n".join(lines) + "\n")
            cache_dir = root / "processed" / "four_rocks_v1"

            cache_path = prepare_cache(specs, cache_dir, progress_every=0)

            bundle = load_cached_bundle(cache_path)
            self.assertEqual(bundle.output_features, OUTPUT_FEATURES)
            self.assertEqual(bundle.trajectories.shape, (8, 3, len(OUTPUT_FEATURES)))

    def test_duplicate_dataset_identity_is_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            specs = self._write_four_datasets(root / "raw")
            duplicate_specs = (*specs, specs[0])
            cache_dir = root / "processed" / "four_rocks_v1"

            with self.assertRaisesRegex(ValueError, "dataset names must be unique"):
                prepare_cache(duplicate_specs, cache_dir, progress_every=0)

            self.assertFalse(cache_dir.exists())

    def test_same_source_directory_cannot_be_registered_twice_under_different_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            specs = self._write_four_datasets(root / "raw")
            source_path = Path(specs[0].path)
            duplicate_source = replace(
                specs[0],
                name="Calcite_duplicate",
                path=source_path / ".." / source_path.name,
            )
            cache_dir = root / "processed" / "four_rocks_v1"

            with self.assertRaisesRegex(ValueError, "dataset paths must be unique"):
                prepare_cache((*specs, duplicate_source), cache_dir, progress_every=0)

            self.assertFalse(cache_dir.exists())

    def test_cleanup_error_does_not_mask_primary_preparation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            specs = self._write_four_datasets(root / "raw")
            processed_root = root / "processed"
            cache_dir = processed_root / "four_rocks_v1"

            with (
                mock.patch(
                    "conditional_model_v1.preparation._write_output_chunk",
                    side_effect=RuntimeError("primary write failure"),
                ),
                mock.patch.object(
                    Path,
                    "unlink",
                    side_effect=PermissionError("cleanup failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "primary write failure"):
                    prepare_cache(specs, cache_dir, progress_every=0)

            self.assertFalse(cache_dir.exists())

    def _assert_missing_or_empty(self, path: Path) -> None:
        self.assertFalse(path.exists() and any(path.iterdir()))

    @staticmethod
    def _training_config(root: Path, specs: tuple[DatasetSpec, ...]):
        return parse_config(
            {
                "experiment": {"name": "four_rock_train", "run_root": str(root / "runs")},
                "data": {
                    "processed_root": str(root / "processed"),
                    "cache_name": "four_rocks_v1",
                    "require_cache": True,
                    "datasets": [
                        {
                            "name": spec.name,
                            "rock": spec.rock,
                            "path": str(spec.path),
                            "max_runs": spec.max_runs,
                        }
                        for spec in specs
                    ],
                },
            }
        )

    def _write_four_datasets(self, raw_root: Path) -> tuple[DatasetSpec, ...]:
        specs: list[DatasetSpec] = []
        for dataset_index, (dataset_name, rock) in enumerate(DATASETS, start=1):
            dataset_root = raw_root / dataset_name
            for directory_name in ("input", "output", "failed"):
                (dataset_root / directory_name).mkdir(parents=True)
            for run_id in (1, 2):
                self._write_input(
                    dataset_root / "input" / f"{run_id}_Input.txt",
                    rock=rock,
                    base_value=float(dataset_index * 10 + run_id),
                )
                self._write_output(
                    dataset_root / "output" / f"{run_id}_Output.txt",
                    base_value=float(dataset_index * 10 + run_id),
                )
            (dataset_root / "failed" / "99_Input.txt").write_text("{FAILED} true\n")
            specs.append(DatasetSpec(name=dataset_name, rock=rock, path=dataset_root))
        return tuple(specs)

    @staticmethod
    def _write_input(path: Path, *, rock: str, base_value: float) -> None:
        lines = [
            *(
                f"{{{feature}}} {base_value + index / 10}"
                for index, feature in enumerate(SCALAR_CONDITION_FEATURES)
            ),
            f"{{{rock.upper()}_MOLES}} {base_value + 100}",
            f"{{{rock.upper()}_AREA}} {base_value + 200}",
        ]
        path.write_text("\n".join(lines) + "\n")

    @staticmethod
    def _write_output(
        path: Path,
        *,
        base_value: float,
        time_axis: tuple[float, ...] = (0.0, 0.6, 1.2),
    ) -> None:
        header = " ".join(("time_d", *OUTPUT_FEATURES))
        rows = [header]
        for timestep_index, time_value in enumerate(time_axis):
            values = [
                str(base_value + timestep_index + feature_index / 100)
                for feature_index in range(len(OUTPUT_FEATURES))
            ]
            rows.append(" ".join((str(time_value), *values)))
        path.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    unittest.main()
