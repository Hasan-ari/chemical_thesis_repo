from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import unittest
import zipfile
from copy import deepcopy
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from conditional_model_v1.data import (
    CONDITION_FEATURES,
    OUTPUT_FEATURES,
    TrajectoryBundle,
    write_processed_bundle,
)
from conditional_model_v1.runtime_data import (
    RUNTIME_DISK_HEADROOM,
    ZIP_COPY_BUFFER_BYTES,
    estimate_runtime_bytes,
    materialize_mmap_bundle,
)


class RuntimeDataTests(unittest.TestCase):
    def test_materialize_extracts_npz_once_and_returns_read_only_memmaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_dir, manifest, expected = self._write_cache(root)
            original_zip_read = zipfile.ZipExtFile.read

            def bounded_zip_read(zip_member, size=-1):
                self.assertGreater(size, 0)
                self.assertLessEqual(size, ZIP_COPY_BUFFER_BYTES)
                return original_zip_read(zip_member, size)

            def finite_trajectory(values):
                return bool(np.isfinite(values).all())

            with (
                mock.patch.object(
                    zipfile.ZipFile,
                    "read",
                    side_effect=AssertionError("whole ZIP members must not be read into RAM"),
                ),
                mock.patch.object(
                    zipfile.ZipExtFile,
                    "read",
                    autospec=True,
                    side_effect=bounded_zip_read,
                ),
                mock.patch(
                    "conditional_model_v1.runtime_data.np.load",
                    wraps=np.load,
                ) as load_mock,
                mock.patch(
                    "conditional_model_v1.runtime_data._trajectory_is_finite",
                    side_effect=finite_trajectory,
                ) as trajectory_finite_mock,
            ):
                bundle, runtime_dir = materialize_mmap_bundle(
                    cache_dir=cache_dir,
                    manifest=manifest,
                    runtime_root=root / "runtime",
                )

            self.assertEqual(
                runtime_dir.name,
                f"four_rocks_v1-{manifest['artifacts']['bundle.npz']['sha256'][:16]}",
            )
            self.assertTrue((runtime_dir / "_SUCCESS").is_file())
            self.assertTrue((runtime_dir / "runtime_manifest.json").is_file())
            self.assertEqual(
                {path.name for path in runtime_dir.glob("*.npy")},
                {
                    "conditions.npy",
                    "trajectories.npy",
                    "time_axis.npy",
                    "run_ids.npy",
                    "rocks.npy",
                    "condition_features.npy",
                    "output_features.npy",
                },
            )
            for array in (bundle.conditions, bundle.trajectories, bundle.time_axis):
                self.assertIsInstance(array, np.memmap)
                self.assertFalse(array.flags.writeable)
            np.testing.assert_array_equal(bundle.conditions, expected.conditions)
            np.testing.assert_array_equal(bundle.trajectories, expected.trajectories)
            np.testing.assert_array_equal(bundle.time_axis, expected.time_axis)
            self.assertEqual(bundle.run_ids, expected.run_ids)
            self.assertEqual(bundle.rocks.tolist(), expected.rocks.tolist())
            self.assertGreater(load_mock.call_count, 0)
            for load_call in load_mock.call_args_list:
                self.assertEqual(Path(load_call.args[0]).suffix, ".npy")
                self.assertEqual(load_call.kwargs["mmap_mode"], "r")
                self.assertFalse(load_call.kwargs["allow_pickle"])
            trajectory_finite_mock.assert_called_once()
            del array
            del bundle

    def test_complete_runtime_is_reused_without_reopening_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_dir, manifest, _ = self._write_cache(root)
            first_bundle, first_dir = materialize_mmap_bundle(
                cache_dir=cache_dir,
                manifest=manifest,
                runtime_root=root / "runtime",
            )
            max_header_check_values = first_bundle.conditions.size
            original_isfinite = np.isfinite

            def reject_trajectory_scan(values):
                if values.size > max_header_check_values:
                    raise AssertionError("complete runtime trajectories must not be rescanned")
                return original_isfinite(values)

            with (
                mock.patch(
                    "conditional_model_v1.runtime_data.zipfile.ZipFile",
                    side_effect=AssertionError("NPZ archive must not be reopened for reuse"),
                ),
                mock.patch(
                    "conditional_model_v1.runtime_data._file_sha256",
                    side_effect=AssertionError("unchanged NPZ must not be rehashed"),
                ),
                mock.patch(
                    "conditional_model_v1.runtime_data._trajectory_is_finite",
                    side_effect=AssertionError(
                        "complete runtime trajectories must not be rescanned"
                    ),
                ),
                mock.patch(
                    "conditional_model_v1.runtime_data.np.isfinite",
                    side_effect=reject_trajectory_scan,
                ),
            ):
                second_bundle, second_dir = materialize_mmap_bundle(
                    cache_dir=cache_dir,
                    manifest=manifest,
                    runtime_root=root / "runtime",
                )

            self.assertEqual(second_dir, first_dir)
            np.testing.assert_array_equal(
                second_bundle.trajectories,
                first_bundle.trajectories,
            )
            del first_bundle
            del second_bundle

    def test_source_bundle_hash_selects_a_different_runtime_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_dir, manifest, _ = self._write_cache(root)
            first_bundle, first_dir = materialize_mmap_bundle(
                cache_dir=cache_dir,
                manifest=manifest,
                runtime_root=root / "runtime",
            )
            changed_cache_dir, changed_manifest, _ = self._write_cache(
                root,
                cache_parent="changed_cache",
                trajectory_offset=1.0,
            )

            second_bundle, second_dir = materialize_mmap_bundle(
                cache_dir=changed_cache_dir,
                manifest=changed_manifest,
                runtime_root=root / "runtime",
            )

            first_hash = manifest["artifacts"]["bundle.npz"]["sha256"]
            second_hash = changed_manifest["artifacts"]["bundle.npz"]["sha256"]
            self.assertNotEqual(first_hash, second_hash)
            self.assertEqual(second_dir.name, f"four_rocks_v1-{second_hash[:16]}")
            del first_bundle
            del second_bundle

        self.assertNotEqual(first_dir, second_dir)

    def test_declared_bundle_hash_must_match_actual_npz_before_runtime_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_dir, manifest, _ = self._write_cache(root)
            bundle, runtime_dir = materialize_mmap_bundle(
                cache_dir=cache_dir,
                manifest=manifest,
                runtime_root=root / "runtime",
            )
            del bundle
            bundle_path = cache_dir / "bundle.npz"
            original_bytes = bundle_path.read_bytes()
            changed_bytes = bytearray(original_bytes)
            changed_bytes[len(changed_bytes) // 2] ^= 1
            bundle_path.write_bytes(changed_bytes)
            self.assertEqual(bundle_path.stat().st_size, len(original_bytes))
            self.assertNotEqual(
                hashlib.sha256(original_bytes).hexdigest(),
                hashlib.sha256(changed_bytes).hexdigest(),
            )

            with self.assertRaisesRegex(ValueError, "bundle.npz SHA-256"):
                materialize_mmap_bundle(
                    cache_dir=cache_dir,
                    manifest=manifest,
                    runtime_root=root / "runtime",
                )

            self.assertTrue(runtime_dir.is_dir())

    def test_missing_or_pickle_backed_npz_member_is_rejected(self) -> None:
        cases = ("missing", "unexpected", "pickle")
        for case_name in cases:
            with self.subTest(case=case_name), tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                cache_dir = root / "cache" / "four_rocks_v1"
                cache_dir.mkdir(parents=True)
                arrays = self._arrays()
                if case_name == "missing":
                    arrays.pop("rocks")
                    expected_error = "members"
                elif case_name == "unexpected":
                    arrays["unexpected"] = np.array([1.0])
                    expected_error = "members"
                else:
                    arrays["run_ids"] = np.array(["calcite:1", "dolomite:1"], dtype=object)
                    expected_error = "pickle"
                np.savez_compressed(cache_dir / "bundle.npz", **arrays)
                manifest = self._manifest(cache_dir / "bundle.npz")

                with self.assertRaisesRegex(ValueError, expected_error):
                    materialize_mmap_bundle(
                        cache_dir=cache_dir,
                        manifest=manifest,
                        runtime_root=root / "runtime",
                    )

                runtime_root = root / "runtime"
                self.assertFalse(runtime_root.exists() and any(runtime_root.iterdir()))

    def test_incomplete_or_corrupt_runtime_is_not_reused(self) -> None:
        for case_name in ("missing", "wrong_shape", "wrong_manifest"):
            with self.subTest(case=case_name), tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                cache_dir, manifest, _ = self._write_cache(root)
                bundle, runtime_dir = materialize_mmap_bundle(
                    cache_dir=cache_dir,
                    manifest=manifest,
                    runtime_root=root / "runtime",
                )
                del bundle
                if case_name == "missing":
                    (runtime_dir / "trajectories.npy").unlink()
                elif case_name == "wrong_shape":
                    np.save(
                        runtime_dir / "trajectories.npy",
                        np.zeros((2, 4, len(OUTPUT_FEATURES)), dtype=np.float64),
                    )
                else:
                    runtime_manifest_path = runtime_dir / "runtime_manifest.json"
                    runtime_manifest = json.loads(runtime_manifest_path.read_text())
                    runtime_manifest["source_bundle_sha256"] = "c" * 64
                    runtime_manifest_path.write_text(json.dumps(runtime_manifest))

                with self.assertRaisesRegex(ValueError, "runtime.*(incomplete|shape|manifest)"):
                    materialize_mmap_bundle(
                        cache_dir=cache_dir,
                        manifest=manifest,
                        runtime_root=root / "runtime",
                    )

    def test_nonfinite_trajectory_and_cross_array_feature_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_dir, manifest, _ = self._write_cache(root, nonfinite_trajectory=True)
            finite_sizes: list[int] = []
            finite_has_nan: list[bool] = []
            original_isfinite = np.isfinite

            def record_finite_check(values):
                finite_sizes.append(values.size)
                finite_has_nan.append(bool(np.isnan(values).any()))
                return original_isfinite(values)

            with (
                mock.patch(
                    "conditional_model_v1.runtime_data.FINITE_CHECK_RUN_CHUNK",
                    1,
                ),
                mock.patch(
                    "conditional_model_v1.runtime_data.np.isfinite",
                    side_effect=record_finite_check,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "trajectory.*finite"):
                    materialize_mmap_bundle(
                        cache_dir=cache_dir,
                        manifest=manifest,
                        runtime_root=root / "runtime",
                    )
            one_run_value_count = 3 * len(OUTPUT_FEATURES)
            self.assertGreaterEqual(len(finite_sizes), 1)
            self.assertTrue(
                all(value_count <= one_run_value_count for value_count in finite_sizes)
            )
            self.assertTrue(any(finite_has_nan))

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_dir = root / "cache" / "four_rocks_v1"
            cache_dir.mkdir(parents=True)
            arrays = self._arrays()
            arrays["conditions"] = np.zeros(
                (2, len(CONDITION_FEATURES) - 1),
                dtype=np.float64,
            )
            np.savez_compressed(cache_dir / "bundle.npz", **arrays)
            manifest = self._manifest(cache_dir / "bundle.npz")
            manifest["arrays"]["conditions"]["shape"] = [
                2,
                len(CONDITION_FEATURES) - 1,
            ]

            with self.assertRaisesRegex(ValueError, "condition feature count"):
                materialize_mmap_bundle(
                    cache_dir=cache_dir,
                    manifest=manifest,
                    runtime_root=root / "runtime",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_dir, manifest, _ = self._write_cache(root)
            wrong_manifest = deepcopy(manifest)
            wrong_manifest["n_timesteps"] = 4
            wrong_manifest["arrays"]["trajectories"]["shape"] = [
                2,
                4,
                len(OUTPUT_FEATURES),
            ]
            wrong_manifest["arrays"]["time_axis"]["shape"] = [4]
            with self.assertRaisesRegex(ValueError, "trajectory.*shape"):
                materialize_mmap_bundle(
                    cache_dir=cache_dir,
                    manifest=wrong_manifest,
                    runtime_root=root / "runtime",
                )

    def test_copy_or_disk_preflight_failure_never_publishes_partial_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_dir, manifest, _ = self._write_cache(root)
            runtime_root = root / "runtime"
            required_bytes = estimate_runtime_bytes(cache_dir / "bundle.npz")
            with zipfile.ZipFile(cache_dir / "bundle.npz") as archive:
                uncompressed_bytes = sum(info.file_size for info in archive.infolist())
            self.assertEqual(
                required_bytes,
                math.ceil(uncompressed_bytes * 1.10),
            )
            self.assertEqual(RUNTIME_DISK_HEADROOM, 1.10)
            usage_type = type(shutil.disk_usage(root))

            with mock.patch(
                "conditional_model_v1.runtime_data.shutil.disk_usage",
                return_value=usage_type(
                    total=required_bytes * 2,
                    used=required_bytes + 1,
                    free=required_bytes - 1,
                ),
            ):
                with self.assertRaisesRegex(OSError, "disk space"):
                    materialize_mmap_bundle(
                        cache_dir=cache_dir,
                        manifest=manifest,
                        runtime_root=runtime_root,
                    )
            self.assertFalse(runtime_root.exists() and any(runtime_root.iterdir()))

            copy_calls = 0

            def fail_after_one_member(*args, **kwargs):
                nonlocal copy_calls
                destination = kwargs.get("destination")
                if destination is None:
                    destination = args[2]
                copy_calls += 1
                if copy_calls == 1:
                    destination.write_bytes(b"partial member")
                    return
                raise OSError("injected copy failure")

            with mock.patch(
                "conditional_model_v1.runtime_data._copy_zip_member",
                side_effect=fail_after_one_member,
            ):
                with self.assertRaisesRegex(OSError, "injected copy failure"):
                    materialize_mmap_bundle(
                        cache_dir=cache_dir,
                        manifest=manifest,
                        runtime_root=runtime_root,
                    )
            self.assertFalse(runtime_root.exists() and any(runtime_root.iterdir()))

    def _write_cache(
        self,
        root: Path,
        *,
        cache_parent: str = "cache",
        trajectory_offset: float = 0.0,
        nonfinite_trajectory: bool = False,
    ) -> tuple[Path, dict, TrajectoryBundle]:
        cache_dir = root / cache_parent / "four_rocks_v1"
        arrays = self._arrays()
        arrays["trajectories"] = arrays["trajectories"] + trajectory_offset
        if nonfinite_trajectory:
            arrays["trajectories"][1, 2, 3] = np.nan
        bundle = TrajectoryBundle(
            conditions=arrays["conditions"],
            trajectories=arrays["trajectories"],
            time_axis=arrays["time_axis"],
            run_ids=["calcite:1", "dolomite:1"],
            rocks=np.array(["Calcite", "Dolomite"]),
            condition_features=CONDITION_FEATURES,
            output_features=OUTPUT_FEATURES,
        )
        write_processed_bundle(
            bundle=bundle,
            inputs=pd.DataFrame(),
            outputs=pd.DataFrame(),
            inventory=pd.DataFrame(),
            processed_dir=cache_dir,
        )
        return cache_dir, self._manifest(cache_dir / "bundle.npz"), bundle

    @staticmethod
    def _arrays() -> dict[str, np.ndarray]:
        return {
            "conditions": np.arange(
                2 * len(CONDITION_FEATURES), dtype=np.float64
            ).reshape(2, len(CONDITION_FEATURES)),
            "trajectories": np.arange(
                2 * 3 * len(OUTPUT_FEATURES), dtype=np.float64
            ).reshape(2, 3, len(OUTPUT_FEATURES)),
            "time_axis": np.array([0.0, 0.6, 1.2], dtype=np.float64),
            "run_ids": np.array(["calcite:1", "dolomite:1"]),
            "rocks": np.array(["Calcite", "Dolomite"]),
            "condition_features": np.array(CONDITION_FEATURES),
            "output_features": np.array(OUTPUT_FEATURES),
        }

    @staticmethod
    def _manifest(bundle_path: Path) -> dict:
        sha256 = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        return {
            "n_runs": 2,
            "n_timesteps": 3,
            "condition_features": list(CONDITION_FEATURES),
            "output_features": list(OUTPUT_FEATURES),
            "arrays": {
                "conditions": {
                    "shape": [2, len(CONDITION_FEATURES)],
                    "dtype": "float64",
                },
                "trajectories": {
                    "shape": [2, 3, len(OUTPUT_FEATURES)],
                    "dtype": "float64",
                },
                "time_axis": {"shape": [3], "dtype": "float64"},
            },
            "artifacts": {
                "bundle.npz": {
                    "size_bytes": bundle_path.stat().st_size,
                    "sha256": sha256,
                }
            },
        }


if __name__ == "__main__":
    unittest.main()
