"""Read all runs of all rocks into one RunTable, in parallel.

A RunTable keeps all runs in a small number of large arrays. Row i of each
array is the same run. The rows are in the order of the config rocks, and
inside one rock in numeric run id order.

The build reads about 110,000 text files. numpy.loadtxt does not release the
GIL, so threads do not make the parse faster. A ProcessPoolExecutor gives one
Python interpreter to each worker. Each worker reads one chunk of run ids and
returns stacked arrays, so the parent process does not unpickle one object
for each run.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
from collections.abc import Iterable, Sequence

import numpy as np

from conditional_input_optimization.run_data import rock_source, run_reader

OUTPUT_FEATURES = run_reader.OUTPUT_FEATURES
DEFAULT_CHUNK_SIZE = 256

_ARRAY_FIELDS = ("rocks", "run_ids", "conditions", "time_days", "outputs")


# eq=False: the generated __eq__ cannot compare NumPy arrays.
@dataclasses.dataclass(frozen=True, eq=False)
class RunTable:
    """All runs of all rocks, with one row for each run.

    The arrays are large, so the constructor does not copy them. It accepts
    only read-only arrays. Thus no caller can change the table in place.

    Attributes:
        rocks: The rock label of each run, shape (N,), unicode strings.
            Metadata only. It is never a model input.
        run_ids: The run id of each run, shape (N,), unicode strings.
        conditions: The condition values, shape (N, 57), float64, in the
            order of run_reader.CONDITION_FEATURES.
        time_days: The time axis, shape (T,), in days. All runs share it.
        outputs: The output values, shape (26, N, T), float64, in the order
            of OUTPUT_FEATURES. outputs[f] is one feature for all runs, and
            it is contiguous in memory.

    Raises:
        ValueError: If an array is writable or a shape does not agree.
    """

    rocks: np.ndarray
    run_ids: np.ndarray
    conditions: np.ndarray
    time_days: np.ndarray
    outputs: np.ndarray

    def __post_init__(self) -> None:
        """Check that the arrays are read-only and that the shapes agree."""
        for name in _ARRAY_FIELDS:
            if getattr(self, name).flags.writeable:
                raise ValueError(
                    f"RunTable.{name} must be read-only. Call "
                    "array.setflags(write=False) before you build the table."
                )
        n_runs = len(self.rocks)
        expected = {
            "run_ids": (n_runs,),
            "conditions": (n_runs, len(run_reader.CONDITION_FEATURES)),
            "outputs": (len(OUTPUT_FEATURES), n_runs, len(self.time_days)),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(
                    f"RunTable.{name} shape {getattr(self, name).shape}, "
                    f"expected {shape}."
                )

    def output(self, name: str) -> np.ndarray:
        """Return one output feature for all runs.

        Args:
            name: A name in OUTPUT_FEATURES, for example "H2_g_mol".

        Returns:
            A read-only view, shape (N, T). It is not a copy.

        Raises:
            KeyError: If name is not in OUTPUT_FEATURES.
        """
        if name not in OUTPUT_FEATURES:
            raise KeyError(f"Unknown output feature {name!r}.")
        return self.outputs[OUTPUT_FEATURES.index(name)]


@dataclasses.dataclass(frozen=True)
class _Chunk:
    """The run ids of one rock that one worker reads."""

    source: rock_source.RockSource
    run_ids: tuple[str, ...]


def build_run_table(
    sources: Sequence[rock_source.RockSource],
    executor: concurrent.futures.Executor,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> RunTable:
    """Read all complete runs of the sources into one RunTable.

    Complexity: O(N * T * C) time for N runs, T timesteps and C file columns,
    divided by the number of workers. O(26 * N * T) memory for the result,
    plus the chunks that wait in the executor queue.

    Args:
        sources: The rocks, in the row order of the table.
        executor: The executor that reads the chunks. Use a
            ProcessPoolExecutor for speed. The executor.map order keeps the
            rows in the order of the sources.
        chunk_size: The number of runs in one worker task.

    Returns:
        The RunTable of all complete runs.

    Raises:
        ValueError: If chunk_size is not positive, there is no run, or two
            runs have different time axes.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    chunks = _plan_chunks(sources, chunk_size)
    if not chunks:
        raise ValueError("The sources have no complete run.")
    results = executor.map(_read_chunk, chunks)
    return _assemble(chunks, results)


def _plan_chunks(
    sources: Sequence[rock_source.RockSource], chunk_size: int
) -> list[_Chunk]:
    """Split the complete run ids of each source into chunks."""
    chunks = []
    for source in sources:
        run_ids = rock_source.find_complete_run_ids(source)
        chunks += [
            _Chunk(source, run_ids[start : start + chunk_size])
            for start in range(0, len(run_ids), chunk_size)
        ]
    return chunks


def _read_chunk(chunk: _Chunk) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one chunk in a worker process.

    Args:
        chunk: The run ids to read.

    Returns:
        A tuple (time_days, conditions, outputs) with shapes (T,), (n, 57)
        and (26, n, T) for n runs.

    Raises:
        ValueError: If two runs in the chunk have different time axes.
    """
    reader = run_reader.RunFileReader(chunk.source)
    records = [reader.read(run_id) for run_id in chunk.run_ids]
    time_days = records[0].time_days
    for record in records:
        _check_time_axis(
            time_days, record.time_days, record.rock, record.run_id
        )
    conditions = np.stack([record.conditions for record in records])
    outputs = np.stack([record.outputs.T for record in records], axis=1)
    return time_days, conditions, outputs


def _assemble(
    chunks: Sequence[_Chunk],
    results: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> RunTable:
    """Copy the chunk results into one preallocated table.

    The arrays are allocated after the first chunk, because T is known only
    then. Each chunk is released after the copy, so the peak memory is about
    one table plus the queued chunks.
    """
    n_runs = sum(len(chunk.run_ids) for chunk in chunks)
    first_time = conditions = outputs = None
    start = 0
    for chunk, (time_days, chunk_conditions, chunk_outputs) in zip(
        chunks, results, strict=True
    ):
        if first_time is None:
            first_time = time_days
            conditions = np.empty((n_runs, chunk_conditions.shape[1]))
            outputs = np.empty((len(OUTPUT_FEATURES), n_runs, len(time_days)))
        _check_time_axis(
            first_time, time_days, chunk.source.name, chunk.run_ids[0]
        )
        stop = start + len(chunk.run_ids)
        conditions[start:stop] = chunk_conditions
        outputs[:, start:stop] = chunk_outputs
        start = stop
    rocks = [chunk.source.name for chunk in chunks for _ in chunk.run_ids]
    run_ids = [run_id for chunk in chunks for run_id in chunk.run_ids]
    return _read_only_table(rocks, run_ids, conditions, first_time, outputs)


def _read_only_table(
    rocks: list[str],
    run_ids: list[str],
    conditions: np.ndarray,
    time_days: np.ndarray,
    outputs: np.ndarray,
) -> RunTable:
    """Make the arrays read-only and build the RunTable."""
    arrays = {
        "rocks": np.array(rocks, dtype=np.str_),
        "run_ids": np.array(run_ids, dtype=np.str_),
        "conditions": conditions,
        "time_days": np.array(time_days),
        "outputs": outputs,
    }
    for array in arrays.values():
        array.setflags(write=False)
    return RunTable(**arrays)


def _check_time_axis(
    expected: np.ndarray, actual: np.ndarray, rock: str, run_id: str
) -> None:
    """Raise ValueError if a run does not use the shared time axis."""
    if not np.array_equal(expected, actual):
        raise ValueError(
            f"Run {rock}/{run_id} has a different time axis: "
            f"{len(actual)} steps, expected {len(expected)} steps."
        )
