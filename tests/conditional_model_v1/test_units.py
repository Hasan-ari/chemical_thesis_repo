from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from conditional_model_v1.data import (
    OPTIONAL_SCALAR_CONDITION_FEATURES,
    SCALAR_CONDITION_FEATURES,
    DatasetSpec,
    load_input_parameters,
)
from conditional_model_v1.units import (
    SOLUTION_SPECIES_FEATURES,
    SOLUTION_SPECIES_GFW,
    SUPPORTED_INPUT_UNITS,
    conversion_table,
    convert_solution_units,
    mg_per_l_to_mol_per_kgw,
)

# Real first Quartzite-schist input (mg/L template), minus its mineral lines.
SCHIST_SCALARS = {
    "TEMPERATURE": 35.16,
    "POROSITY": 0.25,
    "WATER_VOLUME": 29.42,
    "GAS_VOLUME": 70.20,
    "SOLID_MASS": 2003.60,
    "ALKALINITY": 1049.99721,
    "NA": 970.59259,
    "K": 164.79921,
    "CA": 3.24828,
    "MG": 0.02120,
    "CL": 102.42744,
    "MN": 0.04989,
    "FE(2)": 0.09024,
    "SI": 227.80827,
    "AL": 0.58568,
    "GAS_PRESSURE": 60.00,
    "H2": 1.1843,
    "CH4": 53.2877,
    "CO2": 4.58860,
    "N2": 0.93569,
    "H2S": 0.003732,
}


class ConversionRuleTests(unittest.TestCase):
    def test_one_mg_per_litre_of_sodium_is_one_over_gfw_millimoles(self) -> None:
        # 22.9898 mg/L Na is exactly 1 mmol/kgw under PHREEQC's rule.
        self.assertAlmostEqual(mg_per_l_to_mol_per_kgw(22.9898, 22.9898), 1e-3)

    def test_ca_factor_matches_the_ratio_measured_on_quartzite_outputs(self) -> None:
        # Quartzite t=0 outputs gave Ca_input(mg/L) / Ca_mol(mol/kgw) = 39948
        # (60 runs, CV 0.05%); the pure-gfw rule differs by PHREEQC's ~0.3%
        # solution-mass correction, which is deliberately ignored.
        divisor = SOLUTION_SPECIES_GFW["CA"].gfw_g_per_mol * 1000.0
        self.assertAlmostEqual(divisor / 39948.0, 1.0, delta=0.005)

    def test_mol_kgw_input_is_a_no_op(self) -> None:
        converted = convert_solution_units(SCHIST_SCALARS, input_units="mol_kgw")
        self.assertEqual(converted, SCHIST_SCALARS)
        self.assertIsNot(converted, SCHIST_SCALARS)

    def test_mg_l_input_only_touches_solution_species(self) -> None:
        converted = convert_solution_units(SCHIST_SCALARS, input_units="mg_L")

        for key, value in SCHIST_SCALARS.items():
            if key in SOLUTION_SPECIES_FEATURES:
                gfw = SOLUTION_SPECIES_GFW[key].gfw_g_per_mol
                self.assertAlmostEqual(converted[key], value / 1000.0 / gfw, places=12)
            else:
                self.assertEqual(converted[key], value)
        # Sodium lands in the same range as the legacy mol/kgw inputs (~0.04-1.7).
        self.assertAlmostEqual(converted["NA"], 0.04222, places=4)

    def test_unknown_units_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported input_units"):
            convert_solution_units(SCHIST_SCALARS, input_units="ppm")

    def test_non_numeric_concentration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            convert_solution_units({"NA": "abc"}, input_units="mg_L")

    def test_table_flags_the_two_unverified_factors(self) -> None:
        unverified = {row["feature"] for row in conversion_table() if not row["verified"]}
        self.assertEqual(unverified, {"SI", "ALKALINITY"})
        self.assertEqual(SUPPORTED_INPUT_UNITS, ("mol_kgw", "mg_L"))


class LoaderIntegrationTests(unittest.TestCase):
    def _write_schist_input(self, directory: Path) -> Path:
        lines = [f"{{{key}}} {value}" for key, value in SCHIST_SCALARS.items()]
        lines.extend(["{QUARTZ_MOLES} 31.3942", "{QUARTZ_AREA} 42.7084"])
        path = directory / "1_Input.txt"
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_schist_row_is_converted_and_zero_filled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = self._write_schist_input(Path(tmp_dir))
            spec = DatasetSpec(name="qs", rock="Quartzite_schist", path=tmp_dir, input_units="mg_L")

            row = load_input_parameters(path, spec)

        self.assertAlmostEqual(row["NA"], 970.59259 / 1000.0 / 22.9898, places=12)
        self.assertAlmostEqual(row["SI"], 227.80827 / 1000.0 / 60.08, places=12)
        self.assertEqual(row["TEMPERATURE"], 35.16)
        self.assertEqual(row["GAS_PRESSURE"], 60.0)
        # Fields the schist template never defines are zero, like absent minerals.
        self.assertEqual(row["PORE_VOLUME"], 0.0)
        self.assertEqual(row["S6"], 0.0)
        for feature in SCALAR_CONDITION_FEATURES:
            self.assertIn(feature, row)

    def test_legacy_row_zero_fills_schist_only_scalars_without_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "1_Input.txt"
            legacy = {
                "TEMPERATURE": 43.24, "POROSITY": 0.14, "WATER_VOLUME": 15.34,
                "GAS_VOLUME": 48.15, "SOLID_MASS": 2339.39, "PORE_VOLUME": 0.14,
                "ALKALINITY": 0.0746, "NA": 1.7376, "MG": 0.000622, "CL": 1.7376,
                "CA": 0.00029, "S6": 0.01197, "H2": 1.184, "CH4": 47.976,
                "CO2": 6.248, "N2": 4.583, "H2S": 0.0095,
            }
            lines = [f"{{{key}}} {value}" for key, value in legacy.items()]
            lines.extend(["{CALCITE_MOLES} 23.37", "{CALCITE_AREA} 51.79"])
            path.write_text("\n".join(lines) + "\n")

            row = load_input_parameters(path, DatasetSpec(name="c", rock="Calcite", path=tmp_dir))

        self.assertEqual(row["NA"], 1.7376)
        self.assertIn("SOLID_MASS", OPTIONAL_SCALAR_CONDITION_FEATURES)  # mica schists omit it
        for feature in ("K", "MN", "FE(2)", "SI", "AL", "GAS_PRESSURE"):
            self.assertIn(feature, OPTIONAL_SCALAR_CONDITION_FEATURES)
            self.assertEqual(row[feature], 0.0)

    def test_required_scalar_is_still_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "1_Input.txt"
            scalars = dict(SCHIST_SCALARS)
            del scalars["TEMPERATURE"]
            lines = [f"{{{key}}} {value}" for key, value in scalars.items()]
            lines.extend(["{QUARTZ_MOLES} 1.0", "{QUARTZ_AREA} 1.0"])
            path.write_text("\n".join(lines) + "\n")
            spec = DatasetSpec(name="qs", rock="Quartzite_schist", path=tmp_dir, input_units="mg_L")

            with self.assertRaisesRegex(ValueError, "missing condition features.*TEMPERATURE"):
                load_input_parameters(path, spec)


if __name__ == "__main__":
    unittest.main()
