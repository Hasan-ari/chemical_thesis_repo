from __future__ import annotations

import unittest

import numpy as np

from conditional_model_v1.config import DataConfig, DatasetConfig, SplitConfig, parse_config
from conditional_model_v1.splitting import (
    build_split,
    leave_one_rock_out_split,
    rock_aware_split,
)

ROCKS = np.array(["Calcite"] * 10 + ["Dolomite"] * 10 + ["Halite"] * 7, dtype=np.str_)


def _config_payload(strategy: str, held_out_rocks=None, datasets=None) -> dict:
    split = {"strategy": strategy, "train": 0.8, "val": 0.1, "test": 0.1, "seed": 42}
    if held_out_rocks is not None:
        split["held_out_rocks"] = held_out_rocks
    return {
        "experiment": {"name": "loro_test", "run_root": "/tmp/runs"},
        "data": {
            "processed_root": "/tmp/processed",
            "datasets": datasets
            or [
                {"name": "c", "rock": "Calcite", "path": "/tmp/c"},
                {"name": "d", "rock": "Dolomite", "path": "/tmp/d"},
            ],
            "split": split,
        },
    }


class LeaveOneRockOutSplitTests(unittest.TestCase):
    def test_held_out_rock_is_entirely_in_test_and_absent_from_train_and_val(self) -> None:
        split = leave_one_rock_out_split(
            rocks=ROCKS, held_out_rock="Dolomite",
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42,
        )

        train, val, test = (set(part.tolist()) for part in (split.train, split.val, split.test))
        dolomite = set(np.flatnonzero(ROCKS == "Dolomite").tolist())
        self.assertTrue(dolomite <= test)
        self.assertFalse(dolomite & train)
        self.assertFalse(dolomite & val)
        # Disjoint and complete.
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertEqual(train | val | test, set(range(len(ROCKS))))
        # Remaining rocks keep their own train/val/test shares.
        self.assertEqual(set(ROCKS[split.train]), {"Calcite", "Halite"})
        self.assertEqual(set(ROCKS[split.val]), {"Calcite", "Halite"})
        self.assertEqual(set(ROCKS[split.test]), {"Calcite", "Dolomite", "Halite"})
        for part in (split.train, split.val, split.test):
            self.assertTrue(np.all(np.diff(part) > 0))

    def test_zero_test_ratio_makes_test_split_exactly_the_held_out_rock(self) -> None:
        split = leave_one_rock_out_split(
            rocks=ROCKS, held_out_rock="Halite",
            train_ratio=0.9, val_ratio=0.1, test_ratio=0.0, seed=1,
        )

        np.testing.assert_array_equal(split.test, np.flatnonzero(ROCKS == "Halite"))
        self.assertEqual(len(split.train) + len(split.val), 20)

    def test_seed_determinism_and_unknown_or_only_rock_rejected(self) -> None:
        kwargs = dict(rocks=ROCKS, held_out_rock="Calcite", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        first = leave_one_rock_out_split(seed=7, **kwargs)
        second = leave_one_rock_out_split(seed=7, **kwargs)
        third = leave_one_rock_out_split(seed=8, **kwargs)
        np.testing.assert_array_equal(first.train, second.train)
        np.testing.assert_array_equal(first.val, second.val)
        self.assertFalse(np.array_equal(first.train, third.train))

        with self.assertRaisesRegex(ValueError, "has no runs"):
            leave_one_rock_out_split(rocks=ROCKS, held_out_rock="Trona",
                                     train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0)
        with self.assertRaisesRegex(ValueError, "at least one rock besides"):
            leave_one_rock_out_split(rocks=ROCKS[:10], held_out_rock="Calcite",
                                     train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0)

    def test_build_split_rock_aware_path_is_byte_identical_to_legacy_call(self) -> None:
        split_config = SplitConfig(train=0.8, val=0.1, test=0.1, seed=42)
        expected = rock_aware_split(rocks=ROCKS, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

        actual = build_split(split_config, ROCKS)

        np.testing.assert_array_equal(actual.train, expected.train)
        np.testing.assert_array_equal(actual.val, expected.val)
        np.testing.assert_array_equal(actual.test, expected.test)
        with self.assertRaisesRegex(ValueError, "only valid with"):
            build_split(split_config, ROCKS, held_out_rock="Calcite")

    def test_build_split_loro_path_requires_and_uses_held_out_rock(self) -> None:
        split_config = SplitConfig(strategy="leave_one_rock_out", held_out_rocks=("Calcite",))
        with self.assertRaisesRegex(ValueError, "requires a held_out_rock"):
            build_split(split_config, ROCKS)

        actual = build_split(split_config, ROCKS, held_out_rock="Calcite")
        expected = leave_one_rock_out_split(
            rocks=ROCKS, held_out_rock="Calcite",
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42,
        )
        np.testing.assert_array_equal(actual.test, expected.test)


class SplitConfigValidationTests(unittest.TestCase):
    def test_bad_strategy_ratio_sum_and_empty_list_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported split.strategy"):
            SplitConfig(strategy="random")
        with self.assertRaisesRegex(ValueError, "must equal 1.0"):
            SplitConfig(train=0.8, val=0.1, test=0.2)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            SplitConfig(train=0.0, val=0.5, test=0.5)
        with self.assertRaisesRegex(ValueError, "at least one rock"):
            SplitConfig(strategy="leave_one_rock_out", held_out_rocks=())

    def test_held_out_rocks_needs_loro_strategy_and_is_normalized_to_tuple(self) -> None:
        with self.assertRaisesRegex(ValueError, "only valid with split.strategy=leave_one_rock_out"):
            SplitConfig(held_out_rocks=["Calcite"])

        config = SplitConfig(strategy="leave_one_rock_out", held_out_rocks=["Calcite", "Trona"])
        self.assertEqual(config.held_out_rocks, ("Calcite", "Trona"))
        self.assertEqual(SplitConfig(strategy="leave_one_rock_out").held_out_rocks, "all")

    def test_parse_config_rejects_unknown_held_out_rock_and_single_rock_loro(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing from data.datasets.*Trona"):
            parse_config(_config_payload("leave_one_rock_out", ["Trona"]))
        with self.assertRaisesRegex(ValueError, "at least two distinct rocks"):
            parse_config(
                _config_payload(
                    "leave_one_rock_out",
                    datasets=[{"name": "c", "rock": "Calcite", "path": "/tmp/c"}],
                )
            )

        config = parse_config(_config_payload("leave_one_rock_out", ["Dolomite"]))
        self.assertEqual(config.data.split.strategy, "leave_one_rock_out")
        self.assertEqual(config.data.split.held_out_rocks, ("Dolomite",))
        legacy = parse_config(_config_payload("rock_aware_run_level"))
        self.assertEqual(legacy.data.split.held_out_rocks, "all")

    def test_data_config_direct_construction_checks_rocks_too(self) -> None:
        datasets = (DatasetConfig(name="c", rock="Calcite", path="/tmp/c"),)
        with self.assertRaisesRegex(ValueError, "at least two distinct rocks"):
            DataConfig(
                datasets=datasets,
                processed_root="/tmp/p",
                split=SplitConfig(strategy="leave_one_rock_out"),
            )


if __name__ == "__main__":
    unittest.main()
