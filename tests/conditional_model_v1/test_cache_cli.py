from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from conditional_model_v1.cli.cache import main
from conditional_model_v1.config import load_config, parse_config


class CacheCliTests(unittest.TestCase):
    def test_committed_four_rock_config_uses_full_versioned_cache_contract(self) -> None:
        config_path = (
            Path(__file__).parents[2]
            / "configs"
            / "conditional_model_v1"
            / "full_colab_four_rocks.yaml"
        )
        with mock.patch.dict(
            os.environ,
            {
                "DATA_ROOT": "/content/data",
                "PROCESSED_ROOT": "/content/processed",
                "RUN_ROOT": "/content/runs",
            },
        ):
            config = load_config(config_path)

        self.assertEqual(config.data.cache_name, "four_rocks_v2")
        self.assertTrue(config.data.require_cache)
        self.assertTrue(config.data.use_cache)
        self.assertFalse(config.data.rebuild_cache)
        self.assertFalse(config.data.write_outputs_csv)
        self.assertEqual(config.run_root, "/content/runs")
        self.assertEqual(config.data.split.strategy, "rock_aware_run_level")
        self.assertEqual(
            (
                config.data.split.train,
                config.data.split.val,
                config.data.split.test,
                config.data.split.seed,
            ),
            (0.8, 0.1, 0.1, 42),
        )
        self.assertEqual(
            [
                (dataset.name, dataset.rock, dataset.path, dataset.max_runs)
                for dataset in config.data.datasets
            ],
            [
                (
                    "Calcite_wat_sat_data_3",
                    "Calcite",
                    "/content/data/wat_sat/Calcite/Calcite_wat_sat_data_3",
                    None,
                ),
                (
                    "Dolomite_wat_sat_data_2",
                    "Dolomite",
                    "/content/data/wat_sat/Dolomite/Dolomite_wat_sat_data_2",
                    None,
                ),
                (
                    "Halite_wat_sat_data_2",
                    "Halite",
                    "/content/data/wat_sat/Halite/Halite_wat_sat_data_2",
                    None,
                ),
                (
                    "Trona_par_sat_data_3",
                    "Trona",
                    "/content/data/wat_sat/Trona/Trona_par_sat_data_3",
                    None,
                ),
            ],
        )
        self.assertEqual(config.cache_dir, Path("/content/processed/four_rocks_v2"))
        self.assertEqual(config.training.epochs, 100)
        self.assertEqual(config.training.batch_size, 64)

    def test_prepare_command_uses_configured_shared_cache_and_all_datasets(self) -> None:
        config = self._config(cache_name="four_rocks_v2")
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "four_rocks.yaml"
            config_path.write_text("placeholder: patched loader\n")
            expected_cache = config.cache_dir / "bundle.npz"
            with (
                mock.patch(
                    "conditional_model_v1.cli.cache.load_config",
                    return_value=config,
                ),
                mock.patch(
                    "conditional_model_v1.cli.cache.prepare_cache",
                    return_value=expected_cache,
                ) as prepare_mock,
                redirect_stdout(StringIO()) as stdout,
            ):
                main(["prepare", "--config", str(config_path)])

        prepare_mock.assert_called_once()
        specs = self._call_argument(prepare_mock.call_args, 0, "specs")
        cache_dir = self._call_argument(prepare_mock.call_args, 1, "cache_dir")
        self.assertEqual(
            [(spec.name, spec.rock, str(spec.path), spec.max_runs) for spec in specs],
            [
                (
                    "Calcite_wat_sat_data_3",
                    "Calcite",
                    "/content/data/Calcite_wat_sat_data_3",
                    123,
                ),
                (
                    "Dolomite_wat_sat_data_2",
                    "Dolomite",
                    "/content/data/Dolomite_wat_sat_data_2",
                    None,
                ),
                (
                    "Halite_wat_sat_data_2",
                    "Halite",
                    "/content/data/Halite_wat_sat_data_2",
                    None,
                ),
                (
                    "Trona_par_sat_data_3",
                    "Trona",
                    "/content/data/Trona_par_sat_data_3",
                    None,
                ),
            ],
        )
        self.assertEqual(cache_dir, config.cache_dir)
        self.assertIn(f"cache_path={expected_cache}", stdout.getvalue())

    def test_validate_command_enables_hash_verification_only_with_deep_flag(self) -> None:
        config = self._config(cache_name="four_rocks_v2")
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "four_rocks.yaml"
            config_path.write_text("placeholder: patched loader\n")
            with (
                mock.patch(
                    "conditional_model_v1.cli.cache.load_config",
                    return_value=config,
                ),
                mock.patch(
                    "conditional_model_v1.cli.cache.validate_cache_manifest",
                    return_value={"n_runs": 39081, "n_timesteps": 301},
                ) as validate_mock,
                redirect_stdout(StringIO()) as stdout,
            ):
                main(["validate", "--config", str(config_path)])
                main(["validate", "--config", str(config_path), "--deep"])

        self.assertEqual(validate_mock.call_count, 2)
        for validation_call in validate_mock.call_args_list:
            cache_dir = self._call_argument(validation_call, 0, "cache_dir")
            specs = self._call_argument(validation_call, 1, "specs")
            self.assertEqual(cache_dir, config.cache_dir)
            self.assertEqual(
                [
                    (spec.name, spec.rock, str(spec.path), spec.max_runs)
                    for spec in specs
                ],
                [
                    (
                        "Calcite_wat_sat_data_3",
                        "Calcite",
                        "/content/data/Calcite_wat_sat_data_3",
                        123,
                    ),
                    (
                        "Dolomite_wat_sat_data_2",
                        "Dolomite",
                        "/content/data/Dolomite_wat_sat_data_2",
                        None,
                    ),
                    (
                        "Halite_wat_sat_data_2",
                        "Halite",
                        "/content/data/Halite_wat_sat_data_2",
                        None,
                    ),
                    (
                        "Trona_par_sat_data_3",
                        "Trona",
                        "/content/data/Trona_par_sat_data_3",
                        None,
                    ),
                ],
            )
        self.assertFalse(
            validate_mock.call_args_list[0].kwargs.get("verify_artifact_hashes", False)
        )
        self.assertTrue(validate_mock.call_args_list[1].kwargs["verify_artifact_hashes"])
        self.assertIn("n_runs=39081 n_timesteps=301", stdout.getvalue())

    def test_commands_require_an_explicit_versioned_cache_name(self) -> None:
        config = self._config(cache_name=None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "legacy.yaml"
            config_path.write_text("placeholder: patched loader\n")
            with (
                mock.patch(
                    "conditional_model_v1.cli.cache.load_config",
                    return_value=config,
                ),
                mock.patch(
                    "conditional_model_v1.cli.cache.prepare_cache"
                ) as prepare_mock,
                mock.patch(
                    "conditional_model_v1.cli.cache.validate_cache_manifest"
                ) as validate_mock,
            ):
                for command in ("prepare", "validate"):
                    with self.subTest(command=command):
                        with self.assertRaisesRegex(ValueError, "data.cache_name"):
                            main([command, "--config", str(config_path)])
            prepare_mock.assert_not_called()
            validate_mock.assert_not_called()

    @staticmethod
    def _call_argument(call, position: int, name: str):
        if name in call.kwargs:
            return call.kwargs[name]
        return call.args[position]

    @staticmethod
    def _config(*, cache_name: str | None):
        return parse_config(
            {
                "experiment": {"name": "four_rock_train", "run_root": "/content/runs"},
                "data": {
                    "processed_root": "/content/processed",
                    "cache_name": cache_name,
                    "require_cache": True,
                    "datasets": [
                        {
                            "name": "Calcite_wat_sat_data_3",
                            "rock": "Calcite",
                            "path": "/content/data/Calcite_wat_sat_data_3",
                            "max_runs": 123,
                        },
                        {
                            "name": "Dolomite_wat_sat_data_2",
                            "rock": "Dolomite",
                            "path": "/content/data/Dolomite_wat_sat_data_2",
                        },
                        {
                            "name": "Halite_wat_sat_data_2",
                            "rock": "Halite",
                            "path": "/content/data/Halite_wat_sat_data_2",
                        },
                        {
                            "name": "Trona_par_sat_data_3",
                            "rock": "Trona",
                            "path": "/content/data/Trona_par_sat_data_3",
                        },
                    ],
                },
            }
        )


if __name__ == "__main__":
    unittest.main()
