from __future__ import annotations

import os
import pickle
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader

from conditional_model_v1.data import (
    IndexedTrajectoryDataset,
    build_condition_time_tensor,
)


class IndexedTrajectoryDatasetTests(unittest.TestCase):
    def test_noncontiguous_indices_match_eager_reference_and_exclude_metadata(self) -> None:
        conditions = np.array(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
            dtype=np.float32,
        )
        original_conditions = conditions.copy()
        time_axis = np.array([0.0, 1.5, 3.0], dtype=np.float64)
        time_mean = float(time_axis.mean())
        time_std = float(time_axis.std())
        time_norm = ((time_axis - time_mean) / time_std).astype(np.float32)
        targets = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
        original_targets = targets.copy()
        indices = np.array([3, 0, 2], dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmp_dir:
            target_path = Path(tmp_dir) / "targets.npy"
            np.save(target_path, targets, allow_pickle=False)
            with closing(
                IndexedTrajectoryDataset(
                    conditions_norm=conditions,
                    time_norm=time_norm,
                    targets_path=target_path,
                    indices=indices,
                )
            ) as dataset:
                eager_x = build_condition_time_tensor(
                    conditions,
                    time_axis,
                    time_mean=time_mean,
                    time_std=time_std,
                )

                self.assertEqual(len(dataset), 3)
                for attribute_name in vars(dataset):
                    self.assertNotIn("rock", attribute_name.lower())
                    self.assertNotIn("run_id", attribute_name.lower())
                self.assertNotIn("x", vars(dataset))
                for local_index, global_index in enumerate(indices):
                    sample_x, sample_y = dataset[local_index]
                    self.assertEqual(tuple(sample_x.shape), (3, 3))
                    self.assertEqual(tuple(sample_y.shape), (3, 2))
                    self.assertEqual(sample_x.dtype, torch.float32)
                    self.assertEqual(sample_y.dtype, torch.float32)
                    np.testing.assert_allclose(sample_x.numpy(), eager_x[global_index])
                    np.testing.assert_allclose(sample_y.numpy(), targets[global_index])

                _sample_x, independent_y = dataset[0]
                independent_y[0, 0] = -999.0
                reopened = np.load(target_path, mmap_mode="r", allow_pickle=False)
                try:
                    self.assertEqual(
                        reopened[indices[0], 0, 0],
                        targets[indices[0], 0, 0],
                    )
                finally:
                    reopened._mmap.close()

        np.testing.assert_array_equal(conditions, original_conditions)
        np.testing.assert_array_equal(targets, original_targets)

    def test_target_map_is_lazy_per_process_and_removed_from_pickle_state(self) -> None:
        conditions = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
        time_norm = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        targets = np.arange(5 * 3 * 2, dtype=np.float32).reshape(5, 3, 2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            target_path = Path(tmp_dir) / "targets.npy"
            np.save(target_path, targets, allow_pickle=False)
            real_load = np.load
            with mock.patch(
                "conditional_model_v1.data.np.load",
                wraps=real_load,
            ) as load_mock:
                with closing(
                    IndexedTrajectoryDataset(
                        conditions_norm=conditions,
                        time_norm=time_norm,
                        targets_path=target_path,
                        indices=np.array([4, 1, 3]),
                    )
                ) as dataset:
                    self.assertIsNone(dataset._targets)
                    constructor_loads = load_mock.call_count

                    dataset[0]
                    self.assertEqual(load_mock.call_count, constructor_loads + 1)
                    first_map = dataset._targets
                    dataset[1]
                    self.assertEqual(load_mock.call_count, constructor_loads + 1)

                    next_pid = os.getpid() + 1000
                    with mock.patch(
                        "conditional_model_v1.data.os.getpid",
                        return_value=next_pid,
                    ):
                        dataset[2]
                    self.assertEqual(load_mock.call_count, constructor_loads + 2)
                    self.assertTrue(first_map._mmap.closed)

                    with closing(pickle.loads(pickle.dumps(dataset))) as restored:
                        self.assertIsNone(restored._targets)
                        self.assertIsNone(restored._target_pid)
                        restored[0]
                        self.assertEqual(load_mock.call_count, constructor_loads + 3)

                    dataset.close()
                    dataset.close()
                    dataset[0]
                    self.assertEqual(load_mock.call_count, constructor_loads + 4)

                for call in load_mock.call_args_list:
                    load_path = call.args[0] if call.args else call.kwargs["file"]
                    self.assertEqual(Path(load_path), target_path)
                    self.assertEqual(call.kwargs["mmap_mode"], "r")
                    self.assertFalse(call.kwargs["allow_pickle"])

    def test_constructor_rejects_invalid_indices_arrays_and_target_contracts(self) -> None:
        conditions = np.zeros((4, 2), dtype=np.float32)
        time_norm = np.zeros(3, dtype=np.float32)
        targets = np.zeros((4, 3, 2), dtype=np.float32)
        invalid_indices = (
            (np.array([], dtype=np.int64), "non-empty"),
            (np.array([0, 0], dtype=np.int64), "unique"),
            (np.array([-1], dtype=np.int64), "range"),
            (np.array([4], dtype=np.int64), "range"),
            (np.array([[0]], dtype=np.int64), "one-dimensional"),
            (np.array([0.0]), "integer"),
            (np.array([True]), "integer"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            target_path = root / "targets.npy"
            np.save(target_path, targets, allow_pickle=False)

            for indices, expected_error in invalid_indices:
                with self.subTest(indices=indices.tolist()):
                    with self.assertRaisesRegex(ValueError, expected_error):
                        IndexedTrajectoryDataset(
                            conditions_norm=conditions,
                            time_norm=time_norm,
                            targets_path=target_path,
                            indices=indices,
                        )

            invalid_arrays = (
                (conditions[:, None, :], time_norm, "conditions_norm must be 2D"),
                (conditions.astype(np.float64), time_norm, "conditions_norm must be float32"),
                (
                    np.empty((4, 0), dtype=np.float32),
                    time_norm,
                    "at least one condition feature",
                ),
                (conditions, time_norm[:, None], "time_norm must be 1D"),
                (conditions, time_norm.astype(np.float64), "time_norm must be float32"),
                (
                    conditions,
                    np.empty((0,), dtype=np.float32),
                    "at least one timestep",
                ),
                (
                    np.array([[np.nan, 0.0]] * 4, dtype=np.float32),
                    time_norm,
                    "conditions_norm must be finite",
                ),
                (
                    conditions,
                    np.array([0.0, np.inf, 1.0], dtype=np.float32),
                    "time_norm must be finite",
                ),
            )
            for invalid_conditions, invalid_time, expected_error in invalid_arrays:
                with self.subTest(expected_error=expected_error):
                    with self.assertRaisesRegex(ValueError, expected_error):
                        IndexedTrajectoryDataset(
                            conditions_norm=invalid_conditions,
                            time_norm=invalid_time,
                            targets_path=target_path,
                            indices=np.array([0, 1]),
                        )

            bad_targets = (
                (np.zeros((4, 3), dtype=np.float32), "targets must be 3D"),
                (np.zeros((4, 3, 2), dtype=np.float64), "targets must be float32"),
                (np.zeros((5, 3, 2), dtype=np.float32), "run count"),
                (np.zeros((4, 4, 2), dtype=np.float32), "timestep count"),
                (np.zeros((4, 3, 0), dtype=np.float32), "output feature"),
            )
            for position, (bad_target, expected_error) in enumerate(bad_targets):
                bad_path = root / f"bad_{position}.npy"
                np.save(bad_path, bad_target, allow_pickle=False)
                with self.subTest(expected_error=expected_error):
                    with self.assertRaisesRegex(ValueError, expected_error):
                        IndexedTrajectoryDataset(
                            conditions_norm=conditions,
                            time_norm=time_norm,
                            targets_path=bad_path,
                            indices=np.array([0, 1]),
                        )

            for metadata_name in ("rocks", "run_ids", "rock_labels"):
                with self.subTest(metadata_name=metadata_name):
                    with self.assertRaises(TypeError):
                        IndexedTrajectoryDataset(
                            conditions_norm=conditions,
                            time_norm=time_norm,
                            targets_path=target_path,
                            indices=np.array([0, 1]),
                            **{metadata_name: np.array(["metadata"] * 4)},
                        )

            with self.assertRaises(FileNotFoundError):
                IndexedTrajectoryDataset(
                    conditions_norm=conditions,
                    time_norm=time_norm,
                    targets_path=root / "missing.npy",
                    indices=np.array([0, 1]),
                )

            with closing(
                IndexedTrajectoryDataset(
                    conditions_norm=conditions,
                    time_norm=time_norm,
                    targets_path=target_path,
                    indices=np.array([0, 1]),
                )
            ) as dataset:
                invalid_items = (
                    (-1, IndexError),
                    (2, IndexError),
                    (0.5, TypeError),
                    (True, TypeError),
                    (np.bool_(True), TypeError),
                )
                for invalid_item, expected_exception in invalid_items:
                    with self.subTest(invalid_item=invalid_item):
                        with self.assertRaises(expected_exception):
                            dataset[invalid_item]
                sample_x, sample_y = dataset[np.int64(1)]
                self.assertEqual(tuple(sample_x.shape), (3, 3))
                self.assertEqual(tuple(sample_y.shape), (3, 2))

    def test_getitem_never_materializes_all_inputs_or_targets(self) -> None:
        conditions = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
        time_norm = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        targets = np.arange(5 * 3 * 2, dtype=np.float32).reshape(5, 3, 2)

        class GuardedTargetMap(np.memmap):
            def __getitem__(self, key):
                run_key = key[0] if isinstance(key, tuple) else key
                remaining_keys = key[1:] if isinstance(key, tuple) else ()
                full_slices = all(
                    isinstance(value, slice) and value == slice(None)
                    for value in remaining_keys
                )
                one_run_slice = (
                    isinstance(run_key, slice)
                    and run_key.step in (None, 1)
                    and run_key.start is not None
                    and run_key.stop == run_key.start + 1
                )
                if not full_slices or not (
                    isinstance(run_key, (int, np.integer)) or one_run_slice
                ):
                    raise AssertionError("target access must select exactly one run")
                return super().__getitem__(key)

            def copy(self, *args, **kwargs):
                if self.ndim == 3 and self.shape[0] > 1:
                    raise AssertionError("full target array must not be copied")
                return super().copy(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            target_path = Path(tmp_dir) / "targets.npy"
            np.save(target_path, targets, allow_pickle=False)
            real_load = np.load
            real_array = np.array
            real_asarray = np.asarray
            opened_maps = []

            def guarded_load(*args, **kwargs):
                loaded = real_load(*args, **kwargs)
                guarded = loaded.view(GuardedTargetMap)
                opened_maps.append(guarded)
                return guarded

            def array_argument(args, kwargs, keyword):
                return args[0] if args else kwargs[keyword]

            def guarded_array(*args, **kwargs):
                values = array_argument(args, kwargs, "object")
                if (
                    isinstance(values, GuardedTargetMap)
                    and values.ndim == 3
                    and values.shape[0] > 1
                ):
                    raise AssertionError("full target array must not be copied")
                return real_array(*args, **kwargs)

            def guarded_asarray(*args, **kwargs):
                values = array_argument(args, kwargs, "a")
                if (
                    isinstance(values, GuardedTargetMap)
                    and values.ndim == 3
                    and values.shape[0] > 1
                ):
                    raise AssertionError("full target array must not be materialized")
                return real_asarray(*args, **kwargs)

            real_torch_tensor = torch.tensor

            def guarded_torch_tensor(data, *args, **kwargs):
                if (
                    isinstance(data, GuardedTargetMap)
                    and data.ndim == 3
                    and data.shape[0] > 1
                ):
                    raise AssertionError("full target tensor must not be materialized")
                return real_torch_tensor(data, *args, **kwargs)

            try:
                with (
                    mock.patch(
                        "conditional_model_v1.data.np.load",
                        side_effect=guarded_load,
                    ),
                    mock.patch(
                        "conditional_model_v1.data.np.array",
                        side_effect=guarded_array,
                    ),
                    mock.patch(
                        "conditional_model_v1.data.np.asarray",
                        side_effect=guarded_asarray,
                    ),
                    mock.patch(
                        "conditional_model_v1.data.torch.tensor",
                        side_effect=guarded_torch_tensor,
                    ),
                    mock.patch(
                        "conditional_model_v1.data.build_condition_time_tensor",
                        side_effect=AssertionError("full X builder must not be called"),
                    ),
                ):
                    with closing(
                        IndexedTrajectoryDataset(
                            conditions_norm=conditions,
                            time_norm=time_norm,
                            targets_path=target_path,
                            indices=np.array([4, 1, 3]),
                        )
                    ) as dataset:
                        sample_x, sample_y = dataset[0]
                        self.assertEqual(tuple(sample_x.shape), (3, 3))
                        self.assertEqual(tuple(sample_y.shape), (3, 2))
                        self.assertFalse(
                            any(
                                (
                                    isinstance(value, np.ndarray)
                                    and not isinstance(value, np.memmap)
                                    and value.ndim == 3
                                )
                                or (
                                    isinstance(value, torch.Tensor)
                                    and value.ndim == 3
                                )
                                for value in vars(dataset).values()
                            )
                        )
            finally:
                for opened_map in opened_maps:
                    if not opened_map._mmap.closed:
                        opened_map._mmap.close()

    def test_dataloader_preserves_eval_order_and_seeded_shuffle_is_repeatable(self) -> None:
        conditions = np.column_stack(
            [np.arange(6, dtype=np.float32), np.ones(6, dtype=np.float32)]
        )
        time_norm = np.array([-1.0, 1.0], dtype=np.float32)
        targets = np.zeros((6, 2, 3), dtype=np.float32)
        indices = np.array([5, 1, 4, 0, 3], dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmp_dir:
            target_path = Path(tmp_dir) / "targets.npy"
            np.save(target_path, targets, allow_pickle=False)
            with closing(
                IndexedTrajectoryDataset(
                    conditions_norm=conditions,
                    time_norm=time_norm,
                    targets_path=target_path,
                    indices=indices,
                )
            ) as dataset:
                eval_loader = DataLoader(
                    dataset,
                    batch_size=2,
                    shuffle=False,
                    num_workers=0,
                )
                eval_order = []
                batch_sizes = []
                for x_batch, y_batch in eval_loader:
                    batch_sizes.append(x_batch.shape[0])
                    self.assertEqual(tuple(x_batch.shape[1:]), (2, 3))
                    self.assertEqual(tuple(y_batch.shape[1:]), (2, 3))
                    self.assertEqual(x_batch.dtype, torch.float32)
                    self.assertEqual(y_batch.dtype, torch.float32)
                    eval_order.extend(x_batch[:, 0, 0].to(torch.int64).tolist())
                self.assertEqual(batch_sizes, [2, 2, 1])
                self.assertEqual(eval_order, indices.tolist())

                generator_a = torch.Generator().manual_seed(17)
                generator_b = torch.Generator().manual_seed(17)
                loader_a = DataLoader(
                    dataset,
                    batch_size=2,
                    shuffle=True,
                    num_workers=0,
                    generator=generator_a,
                )
                loader_b = DataLoader(
                    dataset,
                    batch_size=2,
                    shuffle=True,
                    num_workers=0,
                    generator=generator_b,
                )

                order_a = [
                    int(value)
                    for x_batch, _ in loader_a
                    for value in x_batch[:, 0, 0].tolist()
                ]
                order_b = [
                    int(value)
                    for x_batch, _ in loader_b
                    for value in x_batch[:, 0, 0].tolist()
                ]
                self.assertEqual(order_a, order_b)
                self.assertCountEqual(order_a, indices.tolist())

    def test_spawn_workers_reuse_lazy_dataset_across_two_epochs(self) -> None:
        conditions = np.column_stack(
            [np.arange(5, dtype=np.float32), np.ones(5, dtype=np.float32)]
        )
        time_norm = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        targets = np.broadcast_to(
            np.arange(5, dtype=np.float32)[:, None, None],
            (5, 3, 2),
        ).copy()

        with tempfile.TemporaryDirectory() as tmp_dir:
            target_path = Path(tmp_dir) / "targets.npy"
            np.save(target_path, targets, allow_pickle=False)
            with closing(
                IndexedTrajectoryDataset(
                    conditions_norm=conditions,
                    time_norm=time_norm,
                    targets_path=target_path,
                    indices=np.array([4, 1, 3, 0, 2]),
                )
            ) as dataset:
                loader = DataLoader(
                    dataset,
                    batch_size=2,
                    shuffle=False,
                    num_workers=2,
                    persistent_workers=True,
                    prefetch_factor=1,
                    multiprocessing_context="spawn",
                    timeout=30,
                )
                primary_error = None
                try:
                    for _epoch in range(2):
                        batches = list(loader)
                        self.assertEqual(
                            [x_batch.shape[0] for x_batch, _ in batches],
                            [2, 2, 1],
                        )
                        self.assertEqual(
                            [
                                int(value)
                                for x_batch, _ in batches
                                for value in x_batch[:, 0, 0].tolist()
                            ],
                            [4, 1, 3, 0, 2],
                        )
                        self.assertEqual(
                            [
                                int(value)
                                for _, y_batch in batches
                                for value in y_batch[:, 0, 0].tolist()
                            ],
                            [4, 1, 3, 0, 2],
                        )
                        self.assertTrue(
                            all(
                                x_batch.dtype == torch.float32
                                and y_batch.dtype == torch.float32
                                for x_batch, y_batch in batches
                            )
                        )
                except BaseException as exc:
                    primary_error = exc
                    raise
                finally:
                    iterator = getattr(loader, "_iterator", None)
                    if iterator is not None:
                        try:
                            iterator._shutdown_workers()
                        except BaseException:
                            if primary_error is None:
                                raise


if __name__ == "__main__":
    unittest.main()
