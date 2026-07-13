from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from conditional_model_v1.preprocessing import (
    NORMALIZED_TARGET_DISK_HEADROOM,
    NPY_HEADER_ALLOWANCE_BYTES,
    ConditionScaler,
    OutputScaler,
    estimate_normalized_target_bytes,
    materialize_normalized_targets,
)


class GuardedRunArray:
    """Array-like that rejects full materialization and oversized run reads."""

    def __init__(self, values: np.ndarray, *, max_runs: int) -> None:
        self._values = values
        self.max_runs = max_runs
        self.shape = values.shape
        self.ndim = values.ndim
        self.dtype = values.dtype

    def __getitem__(self, key):
        result = self._values[key]
        if result.ndim == self.ndim and result.shape[0] > self.max_runs:
            raise AssertionError("source read exceeded configured run chunk")
        return result

    def __array__(self, *args, **kwargs):
        raise AssertionError("full source array must not be materialized")


class MemoryBoundedPreprocessingTests(unittest.TestCase):
    def test_indexed_scalers_match_one_shot_train_only_statistics(self) -> None:
        rng = np.random.RandomState(7)
        conditions = rng.normal(size=(7, 4)).astype(np.float64)
        trajectories = rng.normal(size=(7, 5, 3)).astype(np.float64)
        trajectories[:, :, 1] -= 1.5
        trajectories[:, :, 2] = 5.0
        train_indices = np.array([6, 1, 4, 0], dtype=np.int64)

        expected_conditions = ConditionScaler().fit(conditions[train_indices])
        actual_conditions = ConditionScaler().fit_indexed(
            conditions,
            train_indices,
            chunk_runs=2,
        )
        expected_outputs = OutputScaler(log_feature_indices=(1,)).fit(
            trajectories[train_indices]
        )
        actual_outputs = OutputScaler(log_feature_indices=(1,)).fit_indexed(
            trajectories,
            train_indices,
            chunk_runs=2,
        )

        np.testing.assert_allclose(
            actual_conditions.scaler.mean_,
            expected_conditions.scaler.mean_,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual_conditions.scaler.var_,
            expected_conditions.scaler.var_,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual_outputs.scaler.mean_,
            expected_outputs.scaler.mean_,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual_outputs.scaler.var_,
            expected_outputs.scaler.var_,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual_outputs.transform(trajectories),
            expected_outputs.transform(trajectories),
            rtol=1e-6,
            atol=1e-6,
        )
        normalized = actual_outputs.transform(trajectories)
        np.testing.assert_allclose(
            actual_outputs.inverse_transform(normalized),
            expected_outputs.inverse_transform(normalized),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(actual_outputs.scaler.scale_[2], 1.0)

        changed_non_train = trajectories.copy()
        changed_non_train[[2, 3, 5]] = 1e12
        no_leakage = OutputScaler(log_feature_indices=(1,)).fit_indexed(
            changed_non_train,
            train_indices,
            chunk_runs=1,
        )
        np.testing.assert_allclose(
            no_leakage.scaler.mean_,
            actual_outputs.scaler.mean_,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_indexed_scaler_partial_fit_calls_are_bounded_by_run_chunks(self) -> None:
        conditions = np.arange(9 * 4, dtype=np.float64).reshape(9, 4)
        trajectories = np.arange(9 * 5 * 3, dtype=np.float64).reshape(9, 5, 3)
        train_indices = np.array([8, 0, 6, 2, 4], dtype=np.int64)
        condition_scaler = ConditionScaler()
        output_scaler = OutputScaler(log_feature_indices=())
        scaler_class = type(condition_scaler.scaler)
        original_partial_fit = scaler_class.partial_fit

        def run_partial_fit(candidate, values=None, *args, **kwargs):
            if values is None:
                values = kwargs.pop("X")
            return original_partial_fit(candidate, values, *args, **kwargs)

        def values_from_call(call):
            if len(call.args) > 1:
                return call.args[1]
            return call.kwargs["X"]

        with mock.patch.object(
            scaler_class,
            "partial_fit",
            autospec=True,
            side_effect=run_partial_fit,
        ) as condition_partial_fit:
            condition_scaler.fit_indexed(
                GuardedRunArray(conditions, max_runs=2),
                train_indices,
                chunk_runs=2,
            )

        with mock.patch.object(
            scaler_class,
            "partial_fit",
            autospec=True,
            side_effect=run_partial_fit,
        ) as output_partial_fit:
            output_scaler.fit_indexed(
                GuardedRunArray(trajectories, max_runs=2),
                train_indices,
                chunk_runs=2,
            )

        self.assertGreater(condition_partial_fit.call_count, 0)
        self.assertGreater(output_partial_fit.call_count, 0)
        self.assertEqual(
            sum(
                values_from_call(call).shape[0]
                for call in condition_partial_fit.call_args_list
            ),
            len(train_indices),
        )
        self.assertEqual(
            sum(
                values_from_call(call).shape[0]
                for call in output_partial_fit.call_args_list
            ),
            len(train_indices) * trajectories.shape[1],
        )
        self.assertTrue(
            all(
                values_from_call(call).shape[0] <= 2
                for call in condition_partial_fit.call_args_list
            )
        )
        self.assertTrue(
            all(
                values_from_call(call).shape[0] <= 2 * trajectories.shape[1]
                for call in output_partial_fit.call_args_list
            )
        )

    def test_indexed_fit_replaces_old_statistics_and_failure_preserves_last_good_fit(self) -> None:
        first_conditions = np.zeros((4, 2), dtype=np.float64)
        second_conditions = np.full((4, 2), 10.0, dtype=np.float64)
        condition_scaler = ConditionScaler().fit(first_conditions)
        condition_scaler.fit_indexed(
            second_conditions,
            np.array([0, 1, 2, 3]),
            chunk_runs=2,
        )
        np.testing.assert_allclose(condition_scaler.scaler.mean_, [10.0, 10.0])

        last_good_condition_scaler = condition_scaler.scaler
        last_good_condition_mean = condition_scaler.scaler.mean_.copy()
        scaler_class = type(condition_scaler.scaler)
        original_partial_fit = scaler_class.partial_fit
        condition_partial_fit_calls = 0

        def fail_second_condition_partial_fit(candidate, values, *args, **kwargs):
            nonlocal condition_partial_fit_calls
            condition_partial_fit_calls += 1
            if condition_partial_fit_calls == 2:
                raise RuntimeError("injected condition partial-fit failure")
            return original_partial_fit(candidate, values, *args, **kwargs)

        with mock.patch.object(
            scaler_class,
            "partial_fit",
            autospec=True,
            side_effect=fail_second_condition_partial_fit,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "injected condition partial-fit failure",
            ):
                condition_scaler.fit_indexed(
                    np.full((3, 2), 99.0, dtype=np.float64),
                    np.array([0, 1, 2]),
                    chunk_runs=1,
                )

        self.assertIs(condition_scaler.scaler, last_good_condition_scaler)
        np.testing.assert_array_equal(
            condition_scaler.scaler.mean_,
            last_good_condition_mean,
        )
        self.assertTrue(condition_scaler.fitted)

        first_targets = np.zeros((4, 3, 2), dtype=np.float64)
        second_targets = np.full((4, 3, 2), 20.0, dtype=np.float64)
        output_scaler = OutputScaler(log_feature_indices=()).fit(first_targets)
        output_scaler.fit_indexed(
            second_targets,
            np.array([0, 1, 2, 3]),
            chunk_runs=2,
        )
        np.testing.assert_allclose(output_scaler.scaler.mean_, [20.0, 20.0])

        last_good_scaler = output_scaler.scaler
        last_good_mean = output_scaler.scaler.mean_.copy()
        scaler_class = type(output_scaler.scaler)
        original_partial_fit = scaler_class.partial_fit
        partial_fit_calls = 0

        def fail_second_partial_fit(candidate, values, *args, **kwargs):
            nonlocal partial_fit_calls
            partial_fit_calls += 1
            if partial_fit_calls == 2:
                raise RuntimeError("injected partial-fit failure")
            return original_partial_fit(candidate, values, *args, **kwargs)

        with mock.patch.object(
            scaler_class,
            "partial_fit",
            autospec=True,
            side_effect=fail_second_partial_fit,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected partial-fit failure"):
                output_scaler.fit_indexed(
                    np.full((3, 3, 2), 99.0, dtype=np.float64),
                    np.array([0, 1, 2]),
                    chunk_runs=1,
                )

        self.assertIs(output_scaler.scaler, last_good_scaler)
        np.testing.assert_array_equal(output_scaler.scaler.mean_, last_good_mean)
        self.assertTrue(output_scaler.fitted)

    def test_indexed_scalers_reject_invalid_indices_and_chunk_sizes(self) -> None:
        conditions = np.zeros((3, 2), dtype=np.float64)
        trajectories = np.zeros((3, 4, 2), dtype=np.float64)
        cases = (
            (np.array([], dtype=np.int64), 1, "non-empty"),
            (np.array([0, 0], dtype=np.int64), 1, "unique"),
            (np.array([3], dtype=np.int64), 1, "range"),
            (np.array([-1], dtype=np.int64), 1, "range"),
            (np.array([[0]], dtype=np.int64), 1, "one-dimensional"),
            (np.array([0.0]), 1, "integer"),
            (np.array([True]), 1, "integer"),
            (np.array([0], dtype=np.int64), 0, "chunk_runs"),
        )
        for indices, chunk_runs, expected_error in cases:
            with self.subTest(indices=indices.tolist(), chunk_runs=chunk_runs):
                with self.assertRaisesRegex(ValueError, expected_error):
                    ConditionScaler().fit_indexed(
                        conditions,
                        indices,
                        chunk_runs=chunk_runs,
                    )
                with self.assertRaisesRegex(ValueError, expected_error):
                    OutputScaler(log_feature_indices=()).fit_indexed(
                        trajectories,
                        indices,
                        chunk_runs=chunk_runs,
                    )

    def test_normalized_targets_are_written_once_in_bounded_float32_chunks(self) -> None:
        rng = np.random.RandomState(11)
        trajectories = np.abs(rng.normal(size=(7, 5, 3))).astype(np.float64)
        original = trajectories.copy()
        scaler = OutputScaler(log_feature_indices=(1,)).fit_indexed(
            trajectories,
            np.array([0, 2, 4, 6]),
            chunk_runs=2,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "targets_norm.npy"
            with mock.patch.object(
                scaler,
                "transform",
                wraps=scaler.transform,
            ) as transform_mock:
                targets = materialize_normalized_targets(
                    trajectories=GuardedRunArray(trajectories, max_runs=2),
                    scaler=scaler,
                    output_path=output_path,
                    chunk_runs=2,
                )

            self.assertIsInstance(targets, np.memmap)
            self.assertFalse(targets.flags.writeable)
            self.assertEqual(targets.dtype, np.float32)
            self.assertEqual(targets.shape, trajectories.shape)
            self.assertEqual(Path(targets.filename).resolve(), output_path.resolve())
            self.assertTrue(Path(targets.filename).is_file())
            np.testing.assert_allclose(
                targets,
                scaler.transform(trajectories),
                rtol=1e-6,
                atol=1e-6,
            )
            self.assertTrue(np.isfinite(targets).all())
            self.assertEqual(transform_mock.call_count, 4)
            self.assertTrue(
                all(call.args[0].shape[0] <= 2 for call in transform_mock.call_args_list)
            )
            np.testing.assert_array_equal(trajectories, original)
            del targets

    def test_target_disk_preflight_uses_float32_shape_with_five_percent_headroom(self) -> None:
        shape = (7, 5, 3)
        raw_bytes = math.prod(shape) * np.dtype(np.float32).itemsize
        required_bytes = estimate_normalized_target_bytes(shape)
        self.assertEqual(NORMALIZED_TARGET_DISK_HEADROOM, 1.05)
        self.assertEqual(NPY_HEADER_ALLOWANCE_BYTES, 4096)
        self.assertEqual(
            required_bytes,
            math.ceil(raw_bytes * 1.05) + 4096,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            actual_path = Path(tmp_dir) / "actual.npy"
            actual = np.lib.format.open_memmap(
                actual_path,
                mode="w+",
                dtype=np.float32,
                shape=shape,
            )
            actual.flush()
            actual._mmap.close()
            self.assertGreaterEqual(required_bytes, actual_path.stat().st_size)

        trajectories = np.ones(shape, dtype=np.float64)
        scaler = OutputScaler(log_feature_indices=()).fit(trajectories[:2])
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            usage_type = type(shutil.disk_usage(root))
            with mock.patch(
                "conditional_model_v1.preprocessing.shutil.disk_usage",
                return_value=usage_type(
                    total=required_bytes * 2,
                    used=required_bytes + 1,
                    free=required_bytes - 1,
                ),
            ):
                with self.assertRaisesRegex(OSError, "disk space"):
                    materialize_normalized_targets(
                        trajectories=trajectories,
                        scaler=scaler,
                        output_path=root / "targets_norm.npy",
                        chunk_runs=2,
                    )
            self.assertEqual(list(root.iterdir()), [])

    def test_failed_target_materialization_leaves_no_final_or_staging_file(self) -> None:
        trajectories = np.ones((3, 4, 2), dtype=np.float64)
        scaler = OutputScaler(log_feature_indices=()).fit(trajectories[:2])
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_path = root / "targets_norm.npy"
            calls = 0
            original_transform = scaler.transform

            def fail_second_chunk(values):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected transform failure")
                return original_transform(values)

            with mock.patch.object(scaler, "transform", side_effect=fail_second_chunk):
                with self.assertRaisesRegex(RuntimeError, "injected transform failure"):
                    materialize_normalized_targets(
                        trajectories=trajectories,
                        scaler=scaler,
                        output_path=output_path,
                        chunk_runs=1,
                    )

            self.assertFalse(output_path.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_close_cleanup_failure_is_attached_to_primary_transform_error(self) -> None:
        trajectories = np.ones((3, 4, 2), dtype=np.float64)
        scaler = OutputScaler(log_feature_indices=()).fit(trajectories[:2])
        real_open_memmap = np.lib.format.open_memmap
        opened_maps = []

        class FailingCloseProxy:
            def __init__(self, real_mmap):
                self.real_mmap = real_mmap

            def close(self):
                raise OSError("injected close-cleanup failure")

            def __getattr__(self, name):
                return getattr(self.real_mmap, name)

        def open_with_failing_close(*args, **kwargs):
            target = real_open_memmap(*args, **kwargs)
            real_mmap = target._mmap
            target._mmap = FailingCloseProxy(real_mmap)
            opened_maps.append((target, real_mmap))
            return target

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_path = root / "targets_norm.npy"
            try:
                with (
                    mock.patch(
                        "conditional_model_v1.preprocessing.np.lib.format.open_memmap",
                        side_effect=open_with_failing_close,
                    ),
                    mock.patch.object(
                        scaler,
                        "transform",
                        side_effect=RuntimeError("injected primary transform failure"),
                    ),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "injected primary transform failure",
                    ) as raised:
                        materialize_normalized_targets(
                            trajectories=trajectories,
                            scaler=scaler,
                            output_path=output_path,
                            chunk_runs=1,
                        )

                self.assertTrue(
                    any(
                        "injected close-cleanup failure" in note
                        for note in getattr(raised.exception, "__notes__", ())
                    )
                )
                self.assertFalse(output_path.exists())
            finally:
                for target, real_mmap in opened_maps:
                    target._mmap = real_mmap
                    real_mmap.close()
                for leftover in root.iterdir():
                    leftover.unlink()

    def test_nonfinite_or_flush_failure_does_not_publish_normalized_targets(self) -> None:
        trajectories = np.ones((3, 4, 2), dtype=np.float64)
        scaler = OutputScaler(log_feature_indices=()).fit(trajectories[:2])

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_path = root / "targets_norm.npy"
            nonfinite = np.full((1, 4, 2), np.nan, dtype=np.float32)
            with mock.patch.object(scaler, "transform", return_value=nonfinite):
                with self.assertRaisesRegex(ValueError, "finite"):
                    materialize_normalized_targets(
                        trajectories=trajectories,
                        scaler=scaler,
                        output_path=output_path,
                        chunk_runs=1,
                    )
            self.assertEqual(list(root.iterdir()), [])

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_path = root / "targets_norm.npy"
            with mock.patch.object(
                np.memmap,
                "flush",
                side_effect=OSError("injected flush failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected flush failure"):
                    materialize_normalized_targets(
                        trajectories=trajectories,
                        scaler=scaler,
                        output_path=output_path,
                        chunk_runs=1,
                    )
            self.assertEqual(list(root.iterdir()), [])

    def test_staging_load_failure_does_not_publish_final_target(self) -> None:
        trajectories = np.ones((3, 4, 2), dtype=np.float64)
        scaler = OutputScaler(log_feature_indices=()).fit(trajectories[:2])

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_path = root / "targets_norm.npy"

            def fail_staging_load(path, *args, **kwargs):
                self.assertIn(".building-", Path(path).name)
                self.assertTrue(Path(path).is_file())
                self.assertFalse(output_path.exists())
                raise OSError("injected staging-load failure")

            with mock.patch(
                "conditional_model_v1.preprocessing.np.load",
                side_effect=fail_staging_load,
            ):
                with self.assertRaisesRegex(OSError, "injected staging-load failure"):
                    materialize_normalized_targets(
                        trajectories=trajectories,
                        scaler=scaler,
                        output_path=output_path,
                        chunk_runs=1,
                    )

            self.assertFalse(output_path.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_post_publish_staging_cleanup_failure_warns_but_returns_final_target(
        self,
    ) -> None:
        trajectories = np.ones((3, 4, 2), dtype=np.float64)
        scaler = OutputScaler(log_feature_indices=()).fit(trajectories[:2])
        original_unlink = Path.unlink

        def fail_staging_unlink(path, *args, **kwargs):
            if ".building-" in path.name:
                raise OSError("injected staging-cleanup failure")
            return original_unlink(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_path = root / "targets_norm.npy"
            with self.assertLogs(
                "conditional_model_v1.preprocessing",
                level="WARNING",
            ) as captured_logs:
                with mock.patch.object(
                    Path,
                    "unlink",
                    autospec=True,
                    side_effect=fail_staging_unlink,
                ) as unlink_mock:
                    targets = materialize_normalized_targets(
                        trajectories=trajectories,
                        scaler=scaler,
                        output_path=output_path,
                        chunk_runs=1,
                    )

            self.assertTrue(any("staging" in entry for entry in captured_logs.output))
            staging_unlinks = [
                call
                for call in unlink_mock.call_args_list
                if ".building-" in call.args[0].name
            ]
            self.assertEqual(len(staging_unlinks), 1)
            self.assertTrue(output_path.exists())
            np.testing.assert_allclose(targets, scaler.transform(trajectories))
            del targets
            staging_paths = list(root.glob(".*.building-*.npy"))
            self.assertEqual(len(staging_paths), 1)
            for staging_path in staging_paths:
                original_unlink(staging_path)

    def test_destination_created_during_publish_is_preserved(self) -> None:
        trajectories = np.ones((3, 4, 2), dtype=np.float64)
        scaler = OutputScaler(log_feature_indices=()).fit(trajectories[:2])

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_path = root / "targets_norm.npy"

            def competing_publish(src=None, dst=None, **kwargs):
                source = src if src is not None else kwargs["src"]
                destination = dst if dst is not None else kwargs["dst"]
                self.assertIn(".building-", Path(source).name)
                Path(destination).write_bytes(b"competing writer")
                raise FileExistsError("injected publish race")

            with mock.patch(
                "conditional_model_v1.preprocessing.os.link",
                side_effect=competing_publish,
            ):
                with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                    materialize_normalized_targets(
                        trajectories=trajectories,
                        scaler=scaler,
                        output_path=output_path,
                        chunk_runs=1,
                    )

            self.assertEqual(output_path.read_bytes(), b"competing writer")
            self.assertEqual(list(root.iterdir()), [output_path])


if __name__ == "__main__":
    unittest.main()
