"""Tests of conditional_input_optimization.optimization.file_units."""

from __future__ import annotations

import numpy as np
import pytest

from conditional_input_optimization.optimization import file_units
from conditional_input_optimization.tests import optimization_fakes as fakes
from conditional_model_v1 import units


def test_mg_l_round_trip_with_the_v2_conversion():
    names = list(fakes.FEATURE_NAMES)
    file_values = {name: 10.0 + i for i, name in enumerate(names)}
    model_values = units.convert_solution_units(file_values, input_units="mg_L")
    row = np.array([[model_values[name] for name in names]])
    back = file_units.to_file_units(row, names, file_units.UNITS_MG_L)
    expected = np.array([[file_values[name] for name in names]])
    np.testing.assert_allclose(back, expected, rtol=1e-12)


def test_mg_l_changes_only_the_solution_inputs():
    names = list(fakes.FEATURE_NAMES)
    conditions = fakes.fake_conditions(3, seed=0)
    converted = file_units.to_file_units(
        conditions, names, file_units.UNITS_MG_L
    )
    solution = [names.index(name) for name in units.SOLUTION_SPECIES_GFW]
    other = np.setdiff1d(np.arange(len(names)), solution)
    np.testing.assert_array_equal(converted[:, other], conditions[:, other])
    assert np.all(converted[:, solution] > conditions[:, solution])


def test_mol_kgw_returns_an_equal_copy():
    conditions = fakes.fake_conditions(3, seed=1)
    converted = file_units.to_file_units(
        conditions, fakes.FEATURE_NAMES, file_units.UNITS_MOL_KGW
    )
    np.testing.assert_array_equal(converted, conditions)
    converted[0, 0] = -1.0
    assert conditions[0, 0] != -1.0


def test_to_file_units_rejects_bad_inputs():
    conditions = fakes.fake_conditions(2, seed=2)
    with pytest.raises(ValueError, match="Unsupported"):
        file_units.to_file_units(conditions, fakes.FEATURE_NAMES, "ppm")
    with pytest.raises(ValueError, match="one column"):
        file_units.to_file_units(conditions, ["NA"], file_units.UNITS_MG_L)
