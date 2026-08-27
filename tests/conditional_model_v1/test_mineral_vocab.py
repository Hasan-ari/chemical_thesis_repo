from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from conditional_model_v1.data import (
    CONDITION_FEATURES,
    MINERAL_CONDITION_FEATURES,
    MINERAL_VOCAB,
    OUTPUT_FEATURES,
    SCALAR_CONDITION_FEATURES,
    DatasetSpec,
    load_cached_bundle,
    load_input_parameters,
    load_output_trajectory,
)
from conditional_model_v1.preparation import prepare_cache
from conditional_model_v1.preprocessing import ConditionScaler

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "five_rocks"

# The fixture files are real professor-generated PHREEQC runs and stay local-only
# (repo rule: raw data never goes to GitHub). Tests that need them skip cleanly
# on machines where the fixtures are absent.
requires_fixtures = unittest.skipUnless(
    FIXTURE_ROOT.exists(), "local-only raw-data fixtures not present"
)

# Two real input/output pairs per dataset, copied from the raw PHREEQC folders.
FIVE_ROCKS: tuple[tuple[str, str], ...] = (
    ("Calcite_wat_sat_data_3", "Calcite"),
    ("Dolomite_wat_sat_data_2", "Dolomite"),
    ("Halite_wat_sat_data_2", "Halite"),
    ("Trona_par_sat_data_3", "Trona"),
    ("Sandstone_data_2", "Sandstone"),
)

SANDSTONE_MINERALS: frozenset[str] = frozenset(
    {
        "ALBITE",
        "BARITE",
        "CALCITE",
        "HEMATITE",
        "ILLITE",
        "KAOLINITE",
        "PYRITE",
        "QUARTZ",
    }
)


def _specs() -> tuple[DatasetSpec, ...]:
    return tuple(
        DatasetSpec(name=name, rock=rock, path=FIXTURE_ROOT / name)
        for name, rock in FIVE_ROCKS
    )


def _first_input(dataset_name: str) -> Path:
    return sorted((FIXTURE_ROOT / dataset_name / "input").glob("*_Input.txt"))[0]


def _first_output(dataset_name: str) -> Path:
    return sorted((FIXTURE_ROOT / dataset_name / "output").glob("*_Output.txt"))[0]


class MineralVocabContractTests(unittest.TestCase):
    def test_condition_features_expand_the_mineral_dictionary(self) -> None:
        self.assertEqual(len(SCALAR_CONDITION_FEATURES), 23)
        self.assertEqual(len(MINERAL_VOCAB), 17)
        self.assertEqual(len(MINERAL_CONDITION_FEATURES), 34)
        self.assertEqual(len(CONDITION_FEATURES), 57)
        self.assertEqual(len(set(CONDITION_FEATURES)), len(CONDITION_FEATURES))
        self.assertEqual(MINERAL_VOCAB, tuple(sorted(MINERAL_VOCAB)))
        # All MOLES slots first, then all AREA slots, both in MINERAL_VOCAB order.
        self.assertEqual(
            MINERAL_CONDITION_FEATURES[: len(MINERAL_VOCAB)],
            tuple(f"{mineral}_MOLES" for mineral in MINERAL_VOCAB),
        )
        self.assertEqual(
            MINERAL_CONDITION_FEATURES[len(MINERAL_VOCAB) :],
            tuple(f"{mineral}_AREA" for mineral in MINERAL_VOCAB),
        )
        self.assertNotIn("mineral_moles", CONDITION_FEATURES)
        self.assertNotIn("mineral_area", CONDITION_FEATURES)

    @requires_fixtures
    def test_single_mineral_row_fills_one_slot_and_zeroes_the_rest(self) -> None:
        spec = DatasetSpec(
            name="Calcite_wat_sat_data_3",
            rock="Calcite",
            path=FIXTURE_ROOT / "Calcite_wat_sat_data_3",
        )

        row = load_input_parameters(_first_input(spec.name), spec)

        self.assertGreater(row["CALCITE_MOLES"], 0.0)
        self.assertGreater(row["CALCITE_AREA"], 0.0)
        zero_features = [
            feature for feature in MINERAL_CONDITION_FEATURES if row[feature] == 0.0
        ]
        self.assertEqual(len(zero_features), 32)
        self.assertEqual(
            set(MINERAL_CONDITION_FEATURES) - set(zero_features),
            {"CALCITE_MOLES", "CALCITE_AREA"},
        )

    @requires_fixtures
    def test_multi_mineral_row_fills_every_declared_mineral(self) -> None:
        spec = DatasetSpec(
            name="Sandstone_data_2",
            rock="Sandstone",
            path=FIXTURE_ROOT / "Sandstone_data_2",
        )

        row = load_input_parameters(_first_input(spec.name), spec)

        nonzero_moles = {
            mineral for mineral in MINERAL_VOCAB if row[f"{mineral}_MOLES"] > 0.0
        }
        nonzero_area = {
            mineral for mineral in MINERAL_VOCAB if row[f"{mineral}_AREA"] > 0.0
        }
        self.assertEqual(nonzero_moles, SANDSTONE_MINERALS)
        self.assertEqual(nonzero_area, SANDSTONE_MINERALS)
        self.assertEqual(len(nonzero_moles), 8)
        self.assertEqual(row["TRONA_MOLES"], 0.0)
        self.assertEqual(row["HALITE_MOLES"], 0.0)

    def test_input_without_any_mineral_is_rejected(self) -> None:
        spec = DatasetSpec(name="synthetic", rock="Calcite", path=FIXTURE_ROOT)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "1_Input.txt"
            path.write_text(
                "\n".join(
                    f"{{{feature}}} {index + 1.0}"
                    for index, feature in enumerate(SCALAR_CONDITION_FEATURES)
                )
                + "\n"
            )

            with self.assertRaisesRegex(ValueError, "no mineral with positive moles"):
                load_input_parameters(path, spec)

    def test_missing_scalar_condition_feature_still_raises(self) -> None:
        spec = DatasetSpec(name="synthetic", rock="Calcite", path=FIXTURE_ROOT)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "1_Input.txt"
            lines = [
                f"{{{feature}}} {index + 1.0}"
                for index, feature in enumerate(SCALAR_CONDITION_FEATURES[1:])  # drop TEMPERATURE (mandatory)
            ]
            lines.extend(["{CALCITE_MOLES} 12.0", "{CALCITE_AREA} 34.0"])
            path.write_text("\n".join(lines) + "\n")

            with self.assertRaisesRegex(ValueError, "missing condition features"):
                load_input_parameters(path, spec)


@requires_fixtures
class SandstoneOutputContractTests(unittest.TestCase):
    def test_raw_sandstone_output_carries_extra_mineral_columns(self) -> None:
        frame = load_output_trajectory(_first_output("Sandstone_data_2"))

        self.assertIn("Barite", frame.columns)
        self.assertIn("Calcite", frame.columns)
        self.assertEqual(len(frame), 301)

    def test_prepare_cache_keeps_only_the_shared_output_features(self) -> None:
        specs = _specs()
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "five_rocks_pilot_v1"

            cache_path = prepare_cache(specs, cache_dir, progress_every=0)

            bundle = load_cached_bundle(cache_path)

        self.assertEqual(bundle.output_features, OUTPUT_FEATURES)
        self.assertEqual(len(bundle.output_features), 26)
        self.assertEqual(bundle.condition_features, CONDITION_FEATURES)
        self.assertEqual(bundle.conditions.shape, (10, 57))
        self.assertEqual(bundle.trajectories.shape, (10, 301, 26))
        self.assertEqual(len(bundle.time_axis), 301)
        self.assertEqual(sorted(set(bundle.rocks.tolist())), sorted({rock for _, rock in FIVE_ROCKS}))

        moles_offset = len(SCALAR_CONDITION_FEATURES)
        moles_block = bundle.conditions[:, moles_offset : moles_offset + len(MINERAL_VOCAB)]
        sandstone_rows = np.flatnonzero(bundle.rocks == "Sandstone")
        calcite_rows = np.flatnonzero(bundle.rocks == "Calcite")
        self.assertTrue((np.count_nonzero(moles_block[sandstone_rows], axis=1) == 8).all())
        self.assertTrue((np.count_nonzero(moles_block[calcite_rows], axis=1) == 1).all())


class ZeroVarianceNormalizationTests(unittest.TestCase):
    def test_all_zero_vocab_column_passes_through_as_zeros(self) -> None:
        conditions = np.tile(np.arange(1.0, 4.0), (5, 1))
        conditions[:, 0] = np.arange(5.0)
        conditions[:, 1] = 0.0  # e.g. EPIDOTE_MOLES: absent from every rock in the cache
        conditions[:, 2] = 7.5  # constant but non-zero

        scaler = ConditionScaler().fit(conditions)
        normalized = scaler.transform(conditions)

        self.assertEqual(scaler.scaler.scale_[1], 1.0)
        self.assertEqual(scaler.scaler.scale_[2], 1.0)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertTrue((normalized[:, 1] == 0.0).all())
        self.assertTrue((normalized[:, 2] == 0.0).all())
        self.assertFalse(np.allclose(normalized[:, 0], 0.0))

    def test_indexed_fit_uses_the_same_zero_variance_guard(self) -> None:
        conditions = np.zeros((6, len(CONDITION_FEATURES)), dtype=np.float64)
        conditions[:, 0] = np.arange(6.0)

        scaler = ConditionScaler().fit_indexed(
            conditions,
            np.arange(6),
            chunk_runs=2,
        )
        normalized = scaler.transform(conditions)

        self.assertTrue(np.isfinite(normalized).all())
        self.assertTrue((normalized[:, 1:] == 0.0).all())


if __name__ == "__main__":
    unittest.main()
