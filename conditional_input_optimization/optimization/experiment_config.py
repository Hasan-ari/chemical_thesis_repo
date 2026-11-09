"""Read and check the YAML config of an optimization experiment.

Each config section becomes one frozen dataclass. The loader rejects an
unknown key and a missing key, so a typing error in the YAML file cannot
silently select a default value. Each dataclass checks its values in
__post_init__.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
from collections.abc import Callable, Mapping
from typing import Any

import yaml

from conditional_input_optimization.optimization import (
    constraints,
    diagnostics,
    objectives,
    projections,
)
from conditional_model_v1 import data as v2_data


def _require_number(owner: str, name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{owner}.{name} must be a number, got {value!r}.")
    if not math.isfinite(value):
        raise ValueError(f"{owner}.{name} must be finite, got {value!r}.")


def _require_positive(owner: str, **values: object) -> None:
    for name, value in values.items():
        _require_number(owner, name, value)
        if not value > 0:
            raise ValueError(f"{owner}.{name} must be > 0, got {value!r}.")


def _require_count(owner: str, minimum: int, **values: object) -> None:
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{owner}.{name} must be an int, got {value!r}.")
        if value < minimum:
            raise ValueError(f"{owner}.{name} must be >= {minimum}.")


def _require_subset(owner: str, values: tuple, allowed: tuple) -> None:
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(f"{owner} has unknown values {unknown}.")
    if len(set(values)) != len(values):
        raise ValueError(f"{owner} has duplicate values.")


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """The run data. Paths are relative to the repository root."""

    rock_config: pathlib.Path
    cache_dir: pathlib.Path


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """The frozen v2 model files and the constructor values."""

    checkpoint: pathlib.Path
    preprocessors: pathlib.Path
    hidden_size: int
    num_layers: int
    device: str

    def __post_init__(self) -> None:
        """Check the values."""
        _require_count(
            "model", 1, hidden_size=self.hidden_size, num_layers=self.num_layers
        )


@dataclasses.dataclass(frozen=True)
class SplitConfig:
    """The v2 rock-aware split and the tolerance of the scaler check."""

    train: float
    val: float
    test: float
    seed: int
    scaler_mean_tolerance: float

    def __post_init__(self) -> None:
        """Check the values."""
        _require_positive("split", train=self.train, val=self.val)
        _require_positive("split", test=self.test)
        _require_positive(
            "split", scaler_mean_tolerance=self.scaler_mean_tolerance
        )
        _require_count("split", 0, seed=self.seed)
        if not math.isclose(self.train + self.val + self.test, 1.0):
            raise ValueError("split.train + split.val + split.test must be 1.")


@dataclasses.dataclass(frozen=True)
class TargetConfig:
    """The output, the time window (days), and the limits (mol)."""

    output: str
    start_days: float
    end_days: float
    eps_mol: float
    min_start_mol: float

    def __post_init__(self) -> None:
        """Check the values."""
        _require_subset(
            "target.output", (self.output,), v2_data.OUTPUT_FEATURES
        )
        _require_number("target", "start_days", self.start_days)
        _require_positive("target", end_days=self.end_days, eps=self.eps_mol)
        _require_positive("target", min_start_mol=self.min_start_mol)
        if not self.start_days < self.end_days:
            raise ValueError("target.start_days must be < target.end_days.")
        # An eligible real run has H2(t_start) >= min_start_mol. A larger
        # eps would change the real objective of these runs.
        if self.eps_mol > self.min_start_mol:
            raise ValueError("target.eps_mol must be <= target.min_start_mol.")


@dataclasses.dataclass(frozen=True)
class InputSpaceConfig:
    """The tie test and the inputs that the search must not change."""

    tie_r2_threshold: float
    context_inputs: tuple[str, ...]

    def __post_init__(self) -> None:
        """Check the values."""
        _require_number(
            "input_space", "tie_r2_threshold", self.tie_r2_threshold
        )
        _require_subset(
            "input_space.context_inputs",
            self.context_inputs,
            v2_data.CONDITION_FEATURES,
        )


@dataclasses.dataclass(frozen=True)
class ConstraintConfig:
    """The penalty and the linear constraints of the search."""

    penalty_weight: float
    feasibility_tolerance: float
    linear: tuple[constraints.LinearConstraintSpec, ...]

    def __post_init__(self) -> None:
        """Check the values."""
        _require_positive(
            "constraints",
            penalty_weight=self.penalty_weight,
            feasibility_tolerance=self.feasibility_tolerance,
        )


@dataclasses.dataclass(frozen=True)
class SearchConfig:
    """The objective, the modes, the starts, and the optimizer."""

    objective: str
    modes: tuple[str, ...]
    n_best_starts: int
    n_random_starts: int
    steps: int
    learning_rate: float
    seed: int

    def __post_init__(self) -> None:
        """Check the values."""
        _require_subset(
            "search.objective", (self.objective,), objectives.OBJECTIVE_NAMES
        )
        _require_subset("search.modes", self.modes, projections.MODE_NAMES)
        if not self.modes:
            raise ValueError("search.modes must not be empty.")
        _require_count(
            "search",
            0,
            n_best_starts=self.n_best_starts,
            n_random_starts=self.n_random_starts,
            steps=self.steps,
            seed=self.seed,
        )
        if self.n_best_starts + self.n_random_starts == 0:
            raise ValueError("search needs at least one start.")
        _require_positive("search", learning_rate=self.learning_rate)


@dataclasses.dataclass(frozen=True)
class ReportConfig:
    """The candidate filter and the batch sizes of the report."""

    top_k: int
    dedup_fraction_of_nn: float
    nn_sample_size: int
    batch_size: int
    distance_chunk_size: int

    def __post_init__(self) -> None:
        """Check the values."""
        _require_count(
            "report",
            1,
            top_k=self.top_k,
            nn_sample_size=self.nn_sample_size,
            batch_size=self.batch_size,
            distance_chunk_size=self.distance_chunk_size,
        )
        _require_number(
            "report", "dedup_fraction_of_nn", self.dedup_fraction_of_nn
        )
        if self.dedup_fraction_of_nn < 0:
            raise ValueError("report.dedup_fraction_of_nn must be >= 0.")


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """The complete experiment config. See configs/optimization_v1.yaml."""

    name: str
    data: DataConfig
    model: ModelConfig
    split: SplitConfig
    target: TargetConfig
    input_space: InputSpaceConfig
    constraints: ConstraintConfig
    search: SearchConfig
    report: ReportConfig
    acceptance: diagnostics.AcceptanceRule
    rocks: tuple[str, ...]
    results_dir: pathlib.Path

    def __post_init__(self) -> None:
        """Check the name and the rocks."""
        if not self.name or not self.rocks:
            raise ValueError("name and rocks must not be empty.")
        if len(set(self.rocks)) != len(self.rocks):
            raise ValueError("rocks has duplicate values.")


def _check_keys(entry: object, expected: set[str], owner: str) -> None:
    if not isinstance(entry, Mapping):
        raise ValueError(f"{owner} must be a mapping, got {entry!r}.")
    unknown = sorted(set(entry) - expected)
    missing = sorted(expected - set(entry))
    if unknown:
        raise ValueError(f"{owner} has unknown keys {unknown}.")
    if missing:
        raise ValueError(f"{owner} has missing keys {missing}.")


def _section(
    raw: Mapping[str, Any],
    name: str,
    cls: type,
    **converters: Callable[[Any], Any],
) -> Any:
    """Build one section dataclass. Convert the values with converters."""
    entry = raw[name]
    _check_keys(entry, {field.name for field in dataclasses.fields(cls)}, name)
    values = {
        key: converters[key](value) if key in converters else value
        for key, value in entry.items()
    }
    return cls(**values)


def parse_config(raw: Mapping[str, Any]) -> ExperimentConfig:
    """Build the config from a parsed YAML mapping.

    Args:
        raw: The mapping, with the keys of ExperimentConfig.

    Returns:
        The checked config.

    Raises:
        ValueError: If a key is unknown or missing, or a value is invalid.
    """
    top_keys = {field.name for field in dataclasses.fields(ExperimentConfig)}
    _check_keys(raw, top_keys, "the config")
    path = pathlib.Path
    return ExperimentConfig(
        name=str(raw["name"]),
        data=_section(
            raw, "data", DataConfig, rock_config=path, cache_dir=path
        ),
        model=_section(
            raw, "model", ModelConfig, checkpoint=path, preprocessors=path
        ),
        split=_section(raw, "split", SplitConfig),
        target=_section(raw, "target", TargetConfig),
        input_space=_section(
            raw, "input_space", InputSpaceConfig, context_inputs=tuple
        ),
        constraints=_section(
            raw,
            "constraints",
            ConstraintConfig,
            linear=constraints.specs_from_config,
        ),
        search=_section(raw, "search", SearchConfig, modes=tuple),
        report=_section(raw, "report", ReportConfig),
        acceptance=_section(raw, "acceptance", diagnostics.AcceptanceRule),
        rocks=tuple(raw["rocks"]),
        results_dir=path(raw["results_dir"]),
    )


def load_config(path: pathlib.Path) -> ExperimentConfig:
    """Read and check a YAML config file.

    Args:
        path: The YAML file.

    Returns:
        The checked config.

    Raises:
        ValueError: If the file is not a mapping, a key is unknown or
            missing, or a value is invalid.
    """
    raw = yaml.safe_load(pathlib.Path(path).read_text())
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return parse_config(raw)
