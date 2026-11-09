"""Write small synthetic run files for the tests.

The files copy the format of the PHREEQC run files, but they are very small.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence

from conditional_model_v1 import data as v2_data

MINERAL = "CALCITE"


def required_input_values() -> dict[str, float]:
    """Return one distinct value for each required input key.

    Returns:
        The required scalar keys and the CALCITE mineral keys. The value of
        the key at index i is i + 1, so each key has a different value.
    """
    keys = [
        name
        for name in v2_data.SCALAR_CONDITION_FEATURES
        if name not in v2_data.OPTIONAL_SCALAR_CONDITION_FEATURES
    ]
    keys += [f"{MINERAL}_MOLES", f"{MINERAL}_AREA"]
    return {key: float(index + 1) for index, key in enumerate(keys)}


def write_input(
    folder: pathlib.Path, run_id: str, values: Mapping[str, float]
) -> pathlib.Path:
    """Write <folder>/input/<run_id>_Input.txt and return its path."""
    path = folder / "input" / f"{run_id}_Input.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["{DATABASE} phreeqc.dat"]
    lines += [f"{{{key}}} {value}" for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_output(
    folder: pathlib.Path,
    run_id: str,
    columns: Mapping[str, Sequence[float]],
) -> pathlib.Path:
    """Write <folder>/output/<run_id>_Output.txt and return its path.

    Args:
        folder: The rock folder.
        run_id: The run id.
        columns: The column names and values. All columns have equal length.

    Returns:
        The path of the written file.
    """
    path = folder / "output" / f"{run_id}_Output.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = zip(*columns.values(), strict=True)
    lines = [" ".join(columns)]
    lines += [" ".join(f"{value:.12e}" for value in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def output_columns(
    n_steps: int = 3, scale: float = 1.0
) -> dict[str, list[float]]:
    """Return a time column and all v2 output columns with distinct values.

    Args:
        n_steps: The number of timesteps.
        scale: A factor for all output values, to make runs different.

    Returns:
        time_d is 0.0, 0.6, 1.2, ... The value of feature f at step t is
        scale * (100 * f + t).
    """
    columns = {"time_d": [0.6 * step for step in range(n_steps)]}
    for index, name in enumerate(v2_data.OUTPUT_FEATURES):
        columns[name] = [
            scale * (100.0 * index + step) for step in range(n_steps)
        ]
    return columns


def write_run(
    folder: pathlib.Path, run_id: str, n_steps: int = 3, scale: float = 1.0
) -> None:
    """Write a complete synthetic run: one input file and one output file."""
    write_input(folder, run_id, required_input_values())
    write_output(folder, run_id, output_columns(n_steps, scale))
