"""Read one simulation run into a RunRecord.

A run has two text files. The input file has 57 condition values. The output
file has a table with one row for each timestep. This module reads the two
files and gives back one RunRecord. The other modules use only RunRecord, so
they do not depend on the file format.

Cross-stream import: this module uses conditional_model_v1.data. The frozen
v2 model learned from the values that load_input_parameters returns. That
function converts the mg_L units and puts the 57 features in the model order.
A second implementation could disagree in silence, so we use the v2 function.
The 26 OUTPUT_FEATURES are the v2 targets. All eight rocks have them. Some
rocks have extra columns, which the v2 model does not predict, so we skip them.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

from conditional_input_optimization.run_data import rock_source
from conditional_model_v1 import data as v2_data

CONDITION_FEATURES: tuple[str, ...] = v2_data.CONDITION_FEATURES
OUTPUT_FEATURES: tuple[str, ...] = v2_data.OUTPUT_FEATURES
TIME_COLUMN = "time_d"

_ARRAY_FIELDS = ("conditions", "time_days", "outputs")


# eq=False: the generated __eq__ cannot compare NumPy arrays.
@dataclasses.dataclass(frozen=True, eq=False)
class RunRecord:
    """The data of one simulation run.

    The constructor stores read-only float64 copies of the arrays. A change
    to the caller's arrays does not change the record.

    Attributes:
        rock: The rock label. Metadata only.
        run_id: The run id, for example "1000".
        conditions: The condition values, shape (57,), in the order of
            CONDITION_FEATURES. Solution values are in mol/kgw.
        time_days: The simulation time of each timestep, shape (T,), in days.
        outputs: The output values, shape (T, 26), in the order of
            OUTPUT_FEATURES. The units are the units of the output file.

    Raises:
        ValueError: If an array has the wrong shape or a non-finite value.
    """

    rock: str
    run_id: str
    conditions: np.ndarray
    time_days: np.ndarray
    outputs: np.ndarray

    def __post_init__(self) -> None:
        """Copy the arrays, check them, and make them read-only."""
        for name in _ARRAY_FIELDS:
            array = np.array(getattr(self, name), dtype=np.float64)
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"Run {self.rock}/{self.run_id}: {name} has a "
                    "non-finite value."
                )
            array.setflags(write=False)
            # A frozen dataclass sets its own fields with object.__setattr__.
            object.__setattr__(self, name, array)
        self._check_shapes()

    def _check_shapes(self) -> None:
        """Raise ValueError if an array has the wrong shape."""
        expected = (len(CONDITION_FEATURES),)
        if self.conditions.shape != expected:
            raise ValueError(
                f"Run {self.rock}/{self.run_id}: conditions shape "
                f"{self.conditions.shape}, expected {expected}."
            )
        expected = (len(self.time_days), len(OUTPUT_FEATURES))
        if self.time_days.ndim != 1 or self.outputs.shape != expected:
            raise ValueError(
                f"Run {self.rock}/{self.run_id}: time_days shape "
                f"{self.time_days.shape} and outputs shape "
                f"{self.outputs.shape} do not agree. Expected time_days "
                f"(T,) and outputs (T, {len(OUTPUT_FEATURES)})."
            )


class RunFileReader:
    """Read the run files of one rock and return RunRecord objects.

    Design pattern: Adapter. It puts the PHREEQC text file format behind one
    method, read(run_id). The input file is parsed by the v2 parser, so the
    values are the same values that the v2 model saw in training.

    Args:
        source: The rock dataset to read from.
    """

    def __init__(self, source: rock_source.RockSource) -> None:
        """Store the source and build the v2 dataset spec one time."""
        self._source = source
        self._v2_spec = v2_data.DatasetSpec(
            name=source.name,
            rock=source.name,
            path=source.folder,
            input_units=source.input_units,
        )

    def read(self, run_id: str) -> RunRecord:
        """Read the input file and the output file of one run.

        Complexity: O(T * C) time for T timesteps and C file columns, because
        the parser scans each byte. O(T * 26) memory.

        Args:
            run_id: The run id, for example "1000".

        Returns:
            The RunRecord of the run.

        Raises:
            FileNotFoundError: If a file of the run does not exist.
            ValueError: If a file has a missing or a non-numeric value.
        """
        conditions = self._read_conditions(self._source.input_path(run_id))
        time_days, outputs = _read_time_and_outputs(
            self._source.output_path(run_id)
        )
        return RunRecord(
            rock=self._source.name,
            run_id=run_id,
            conditions=conditions,
            time_days=time_days,
            outputs=outputs,
        )

    def _read_conditions(self, path: pathlib.Path) -> np.ndarray:
        """Return the 57 condition values of one input file, shape (57,)."""
        row = v2_data.load_input_parameters(path, self._v2_spec)
        return np.array(
            [row[name] for name in CONDITION_FEATURES], dtype=np.float64
        )


def _read_time_and_outputs(
    path: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the time column and the 26 output columns of one output file.

    The columns are found by name in the header, so a different column order
    or an extra column does not change the result. numpy.loadtxt reads the
    open file, so the file is opened one time only. On 500 warm files this
    took 0.323 ms per file. Two opens took 0.339 ms. pandas.read_csv took
    0.47 ms with the C engine and 3.42 ms with the python engine that v2 uses.

    Args:
        path: The output file. Its first line has the column names.

    Returns:
        A pair (time_days, outputs) with shapes (T,) and (T, 26).

    Raises:
        ValueError: If a column is missing.
    """
    names = (TIME_COLUMN, *OUTPUT_FEATURES)
    with path.open() as file:
        header = file.readline().split()
        missing = [name for name in names if name not in header]
        if missing:
            raise ValueError(f"{path} has no column {missing}.")
        table = np.loadtxt(
            file,
            usecols=[header.index(name) for name in names],
            ndmin=2,
            dtype=np.float64,
        )
    return table[:, 0], table[:, 1:]
