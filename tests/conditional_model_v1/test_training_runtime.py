from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from conditional_model_v1.cli.train import _make_loader
from conditional_model_v1.config import parse_config
from conditional_model_v1.data import IndexedTrajectoryDataset
from conditional_model_v1.training import _run_epoch, predict_with_targets


class TrainingRuntimeTests(unittest.TestCase):
    def test_loader_factory_is_lazy_deterministic_and_worker_aware(self) -> None:
        conditions = np.column_stack(
            [np.arange(6, dtype=np.float32), np.ones(6, dtype=np.float32)]
        )
        time_norm = np.array([-1.0, 1.0], dtype=np.float32)
        targets = np.zeros((6, 2, 3), dtype=np.float32)
        indices = np.array([5, 1, 4, 0, 3], dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmp_dir:
            target_path = Path(tmp_dir) / "targets.npy"
            np.save(target_path, targets, allow_pickle=False)
            loader_a = _make_loader(
                conditions_norm=conditions,
                time_norm=time_norm,
                targets_path=target_path,
                indices=indices,
                batch_size=2,
                shuffle=True,
                num_workers=0,
                seed=17,
                pin_memory=False,
                persistent_workers=True,
            )
            loader_b = _make_loader(
                conditions_norm=conditions,
                time_norm=time_norm,
                targets_path=target_path,
                indices=indices,
                batch_size=2,
                shuffle=True,
                num_workers=0,
                seed=17,
                pin_memory=False,
                persistent_workers=True,
            )
            self.assertIsInstance(loader_a.dataset, IndexedTrajectoryDataset)
            self.assertFalse(loader_a.persistent_workers)
            self.assertIsNone(loader_a.prefetch_factor)
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

            eval_loader = _make_loader(
                conditions_norm=conditions,
                time_norm=time_norm,
                targets_path=target_path,
                indices=indices,
                batch_size=2,
                shuffle=False,
                num_workers=0,
                seed=18,
                pin_memory=False,
                persistent_workers=False,
            )
            eval_order = [
                int(value)
                for x_batch, _ in eval_loader
                for value in x_batch[:, 0, 0].tolist()
            ]
            self.assertEqual(eval_order, indices.tolist())

            worker_loader = _make_loader(
                conditions_norm=conditions,
                time_norm=time_norm,
                targets_path=target_path,
                indices=indices,
                batch_size=2,
                shuffle=False,
                num_workers=2,
                seed=19,
                pin_memory=True,
                persistent_workers=True,
            )
            self.assertTrue(worker_loader.persistent_workers)
            self.assertEqual(worker_loader.prefetch_factor, 1)
            self.assertTrue(worker_loader.pin_memory)

            loader_a.dataset.close()
            loader_b.dataset.close()
            eval_loader.dataset.close()
            worker_loader.dataset.close()

    def test_validation_uses_eval_mode_and_prediction_preserves_loader_order(self) -> None:
        x = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)
        y = x * 2.0
        loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
        model = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(3, 3, bias=False))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        _run_epoch(
            model=model,
            loader=loader,
            loss_fn=loss_fn,
            device=torch.device("cpu"),
            optimizer=optimizer,
        )
        self.assertTrue(model.training)
        _run_epoch(
            model=model,
            loader=loader,
            loss_fn=loss_fn,
            device=torch.device("cpu"),
        )
        self.assertFalse(model.training)

        identity = nn.Identity()
        predictions, collected_targets = predict_with_targets(
            identity,
            loader,
            torch.device("cpu"),
        )
        np.testing.assert_array_equal(predictions, x.numpy())
        np.testing.assert_array_equal(collected_targets, y.numpy())

    def test_preprocessing_chunk_size_is_config_driven(self) -> None:
        config = parse_config(
            {
                "experiment": {"name": "chunk_config", "run_root": "/tmp/runs"},
                "data": {
                    "processed_root": "/tmp/processed",
                    "datasets": [
                        {"name": "calcite", "rock": "Calcite", "path": "/tmp/data"}
                    ],
                },
                "training": {"preprocessing_chunk_runs": 37},
            }
        )
        self.assertEqual(config.training.preprocessing_chunk_runs, 37)


if __name__ == "__main__":
    unittest.main()
