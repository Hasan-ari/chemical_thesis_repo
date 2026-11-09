"""Convert the solution inputs back to the units of the rock input files.

The run table and the model use mol/kgw for all solution inputs. The input
files of some rocks use mg/L. The chemistry team writes new input files, so
the report also gives the values in the file units.

This module imports conditional_model_v1.units. Reason: the gram formula
weights must be the same values that converted the files to mol/kgw.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from conditional_model_v1 import units

UNITS_MOL_KGW = "mol_kgw"
UNITS_MG_L = "mg_L"
_MG_PER_G = 1000.0
_MATRIX_NDIM = 2


def to_file_units(
    conditions: np.ndarray, feature_names: Sequence[str], input_units: str
) -> np.ndarray:
    """Convert the solution inputs from mol/kgw to the file units.

    This is the inverse of units.mg_per_l_to_mol_per_kgw:
    mg/L = mol/kgw * gfw * 1000. Like units.py, it ignores the PHREEQC
    density correction (about 0.3%).

    Time and memory: O(N * 57).

    Args:
        conditions: Inputs, shape (N, D), solution inputs in mol/kgw.
        feature_names: The D input names, in column order.
        input_units: "mol_kgw" or "mg_L".

    Returns:
        A new array, shape (N, D). Only the solution inputs change.

    Raises:
        ValueError: If input_units is not supported or the names do not
            match the columns.
    """
    if conditions.ndim != _MATRIX_NDIM or conditions.shape[1] != len(
        feature_names
    ):
        raise ValueError("conditions must have one column for each name.")
    converted = np.array(conditions, dtype=np.float64, copy=True)
    if input_units == UNITS_MOL_KGW:
        return converted
    if input_units != UNITS_MG_L:
        raise ValueError(
            f"Unsupported input_units {input_units!r}. Expected "
            f"{UNITS_MOL_KGW!r} or {UNITS_MG_L!r}."
        )
    for column, name in enumerate(feature_names):
        species = units.SOLUTION_SPECIES_GFW.get(name)
        if species is not None:
            converted[:, column] *= species.gfw_g_per_mol * _MG_PER_G
    return converted
