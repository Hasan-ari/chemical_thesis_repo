"""Find the run files of each rock dataset.

This module knows where the run files are. It does not read their content.
The module run_reader reads the content.

Cross-stream import: this module uses conditional_model_v1.units. The EDA must
accept the same unit names as the v2 model, so it uses the v2 list, not a copy.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping
from typing import Any

import yaml

from conditional_model_v1 import units

_INPUT_FOLDER = "input"
_OUTPUT_FOLDER = "output"
_INPUT_SUFFIX = "_Input.txt"
_OUTPUT_SUFFIX = "_Output.txt"


@dataclasses.dataclass(frozen=True)
class RockSource:
    """The location and the input unit of one rock dataset.

    Attributes:
        name: The rock label, for example "Calcite". It is metadata only. It
            is never a model input.
        folder: The folder that contains the subfolders input/ and output/.
        input_units: The unit of the solution values in the input files. One
            of conditional_model_v1.units.SUPPORTED_INPUT_UNITS.

    Raises:
        ValueError: If name is empty or input_units is not supported.
        TypeError: If folder is not a pathlib.Path.
    """

    name: str
    folder: pathlib.Path
    input_units: str

    def __post_init__(self) -> None:
        """Check that the object is valid."""
        if not self.name:
            raise ValueError("RockSource.name must not be empty.")
        if not isinstance(self.folder, pathlib.Path):
            raise TypeError(
                f"RockSource.folder must be a pathlib.Path, "
                f"got {type(self.folder).__name__}."
            )
        if self.input_units not in units.SUPPORTED_INPUT_UNITS:
            raise ValueError(
                f"Rock {self.name!r} has unsupported input_units "
                f"{self.input_units!r}. Supported: "
                f"{units.SUPPORTED_INPUT_UNITS}."
            )

    def input_folder(self) -> pathlib.Path:
        """Return the folder of the input files, <folder>/input."""
        return self.folder / _INPUT_FOLDER

    def output_folder(self) -> pathlib.Path:
        """Return the folder of the output files, <folder>/output."""
        return self.folder / _OUTPUT_FOLDER

    def input_path(self, run_id: str) -> pathlib.Path:
        """Return the path of the input file of one run.

        Args:
            run_id: The run id, for example "1000".

        Returns:
            The path <folder>/input/<run_id>_Input.txt. The file can be absent.
        """
        return self.input_folder() / f"{run_id}{_INPUT_SUFFIX}"

    def output_path(self, run_id: str) -> pathlib.Path:
        """Return the path of the output file of one run.

        Args:
            run_id: The run id, for example "1000".

        Returns:
            The path <folder>/output/<run_id>_Output.txt. The file can be
            absent.
        """
        return self.output_folder() / f"{run_id}{_OUTPUT_SUFFIX}"


def find_complete_run_ids(source: RockSource) -> tuple[str, ...]:
    """Return the ids of the runs that have an input and an output file.

    A failed run has no file in output/, so this function skips it.

    Complexity: O(n log n) time and O(n) memory for n files, because of the
    sort. The directory listing is O(n).

    Args:
        source: The rock dataset to examine.

    Returns:
        The run ids in numeric order, for example ("1", "2", "10").

    Raises:
        FileNotFoundError: If the input/ or the output/ folder does not exist.
    """
    input_ids = _run_ids_in(source.input_folder(), _INPUT_SUFFIX)
    output_ids = _run_ids_in(source.output_folder(), _OUTPUT_SUFFIX)
    return tuple(sorted(input_ids & output_ids, key=_run_sort_key))


def load_rock_sources(
    config_path: pathlib.Path, base_dir: pathlib.Path
) -> tuple[RockSource, ...]:
    """Read the rock sources from a YAML config file.

    Args:
        config_path: The YAML file. It has the keys data_root and rocks.
        base_dir: The folder that a relative data_root starts from. Usually
            the repository root.

    Returns:
        One RockSource for each rock, in the order of the config file.

    Raises:
        ValueError: If a key is missing, the rock list is empty, or two rocks
            have the same name.
    """
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, Mapping):
        raise ValueError(f"{config_path} must contain a YAML mapping.")
    data_root = base_dir / _required(config, "data_root", config_path)
    rocks = _required(config, "rocks", config_path)
    if not rocks:
        raise ValueError(f"{config_path} has an empty 'rocks' list.")
    sources = tuple(
        _rock_source(entry, data_root, config_path) for entry in rocks
    )
    names = [source.name for source in sources]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"{config_path} has duplicate rock names: {duplicates}."
        )
    return sources


def _rock_source(
    entry: Mapping[str, Any], data_root: pathlib.Path, config_path: pathlib.Path
) -> RockSource:
    """Build one RockSource from one entry of the 'rocks' list."""
    name = _required(entry, "name", config_path)
    return RockSource(
        name=name,
        folder=data_root / name,
        input_units=_required(entry, "input_units", config_path),
    )


def _required(
    mapping: Mapping[str, Any], key: str, config_path: pathlib.Path
) -> Any:
    """Return mapping[key], or raise a clear error if the key is missing."""
    if key not in mapping:
        raise ValueError(f"{config_path} is missing the key {key!r}.")
    return mapping[key]


def _run_ids_in(folder: pathlib.Path, suffix: str) -> set[str]:
    """Return the run ids of the files in folder whose names end with suffix."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return {
        path.name.removesuffix(suffix)
        for path in folder.iterdir()
        if path.name.endswith(suffix)
    }


def _run_sort_key(run_id: str) -> tuple[int, int, str]:
    """Sort numeric ids by value first, then other ids by text."""
    if run_id.isdigit():
        return (0, int(run_id), "")
    return (1, 0, run_id)
