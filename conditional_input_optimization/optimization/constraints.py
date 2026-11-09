"""Linear inequality constraints on the 57 decoded inputs.

A box on the free inputs is not sufficient. Two examples:

1. A tied input (for example N2 = c - CH4 - CO2) can become negative at a
   corner of the box.
2. The generator scripts put the gas and the water into the pore space
   (GAS_VOLUME <= 1000 * POROSITY - WATER_VOLUME). The box corner with
   high gas, low porosity, and high water breaks this rule.

Each constraint has the form a @ x <= upper. The search adds a penalty for
each violation. The report flags each candidate with a violation.

The violation is divided by norm(a * condition_scale). Thus the violation
is a distance in the normalized input space of the model, and all
constraints have the same unit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np
import torch

ALL_ROCKS = "all"


@dataclasses.dataclass(frozen=True)
class LinearConstraintSpec:
    """One constraint from the config: sum(coefficients * x) <= max(data).

    The upper limit is not in the config. It is the max of the left side
    over the real runs of the rock. Thus the real runs always obey the
    constraint, also when the input files hold rounded values.

    Attributes:
        name: A short name for the report.
        coefficients: (input name, coefficient) pairs.
        rocks: The rocks that use the constraint, or ("all",).

    Raises:
        ValueError: If the name or the coefficients are empty.
    """

    name: str
    coefficients: tuple[tuple[str, float], ...]
    rocks: tuple[str, ...]

    def __post_init__(self) -> None:
        """Check that the spec is not empty."""
        if not self.name or not self.coefficients or not self.rocks:
            raise ValueError("A constraint needs a name, coefficients, rocks.")

    def applies_to(self, rock: str) -> bool:
        """Return True if the rock uses this constraint."""
        return ALL_ROCKS in self.rocks or rock in self.rocks


@dataclasses.dataclass(frozen=True, eq=False)
class LinearConstraints:
    """A set of m constraints: matrix @ x <= upper.

    Attributes:
        names: One name for each constraint, length m.
        matrix: The coefficients, shape (m, 57).
        upper: The upper limits, shape (m,).
        scale: The norm of each row in the normalized input space,
            shape (m,). All positive.

    Raises:
        ValueError: If the shapes do not match or a scale is not positive.
    """

    names: tuple[str, ...]
    matrix: np.ndarray
    upper: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        """Check the shapes and make the arrays read-only."""
        m = len(self.names)
        for name in ("matrix", "upper", "scale"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.shape[0] != m:
                raise ValueError(f"{name} must have {m} rows.")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if self.matrix.ndim != 2:  # noqa: PLR2004
            raise ValueError("matrix must be 2D.")
        if not (self.scale > 0).all():
            raise ValueError("scale must be positive.")

    def violation(self, conditions: torch.Tensor) -> torch.Tensor:
        """Return the scaled violation of each constraint. Differentiable.

        Time: O(B * m * 57).

        Args:
            conditions: Inputs, shape (B, 57).

        Returns:
            max(0, (matrix @ x - upper) / scale), shape (B, m).
        """
        kwargs = {"dtype": conditions.dtype, "device": conditions.device}
        matrix = torch.tensor(self.matrix, **kwargs)
        upper = torch.tensor(self.upper, **kwargs)
        scale = torch.tensor(self.scale, **kwargs)
        return torch.relu((conditions @ matrix.T - upper) / scale)

    def concat(self, other: LinearConstraints) -> LinearConstraints:
        """Return the union of two constraint sets."""
        return LinearConstraints(
            names=self.names + other.names,
            matrix=np.vstack([self.matrix, other.matrix]),
            upper=np.concatenate([self.upper, other.upper]),
            scale=np.concatenate([self.scale, other.scale]),
        )


def _from_rows(
    names: Sequence[str],
    matrix: np.ndarray,
    upper: np.ndarray,
    condition_scale: np.ndarray,
) -> LinearConstraints:
    # reshape(m, -1) fails for m = 0, so give the column count.
    matrix = np.asarray(matrix, dtype=np.float64).reshape(
        len(names), condition_scale.shape[0]
    )
    scale = np.linalg.norm(matrix * condition_scale, axis=1)
    return LinearConstraints(
        names=tuple(names), matrix=matrix, upper=upper, scale=scale
    )


def calibrated_constraints(
    specs: Sequence[LinearConstraintSpec],
    rock: str,
    conditions: np.ndarray,
    feature_names: Sequence[str],
    condition_scale: np.ndarray,
) -> LinearConstraints:
    """Build the config constraints of one rock. Calibrate on the runs.

    Args:
        specs: All config constraints.
        rock: The rock label.
        conditions: The inputs of the rock runs, shape (N, 57).
        feature_names: The 57 input names, in column order.
        condition_scale: The v2 condition scaler std, shape (57,).

    Returns:
        The constraints that apply to the rock. The upper limit of each one
        is the max of matrix @ x over the runs.

    Raises:
        ValueError: If a spec uses an unknown input name.
    """
    selected = [spec for spec in specs if spec.applies_to(rock)]
    position = {name: i for i, name in enumerate(feature_names)}
    matrix = np.zeros((len(selected), len(feature_names)))
    for row, spec in enumerate(selected):
        for name, value in spec.coefficients:
            if name not in position:
                raise ValueError(
                    f"Constraint {spec.name!r} uses unknown input {name!r}."
                )
            matrix[row, position[name]] = value
    upper = (conditions @ matrix.T).max(axis=0) if selected else np.zeros(0)
    names = [spec.name for spec in selected]
    return _from_rows(names, matrix, upper, condition_scale)


def nonnegative_constraints(
    feature_names: Sequence[str], condition_scale: np.ndarray
) -> LinearConstraints:
    """Build -x_i <= 0 for each input.

    All inputs are amounts, volumes, areas, or temperatures in degC. All
    real values are >= 0.

    Args:
        feature_names: The 57 input names.
        condition_scale: The v2 condition scaler std, shape (57,).

    Returns:
        57 constraints.
    """
    count = len(feature_names)
    names = [f"{name} >= 0" for name in feature_names]
    return _from_rows(names, -np.eye(count), np.zeros(count), condition_scale)


def observed_range_constraints(
    feature_names: Sequence[str],
    observed_min: np.ndarray,
    observed_max: np.ndarray,
    condition_scale: np.ndarray,
) -> LinearConstraints:
    """Build min_i <= x_i <= max_i for each input that varies.

    The bounded mode uses these constraints. The box projection keeps the
    free inputs in range. These constraints also keep the tied inputs in
    range.

    Args:
        feature_names: The 57 input names.
        observed_min: The min of each input over the rock runs, shape (57,).
        observed_max: The max of each input over the rock runs, shape (57,).
        condition_scale: The v2 condition scaler std, shape (57,).

    Returns:
        Two constraints for each varying input.
    """
    columns = np.flatnonzero(observed_max > observed_min)
    identity = np.eye(len(feature_names))[columns]
    names = [f"{feature_names[i]} <= observed max" for i in columns]
    names += [f"{feature_names[i]} >= observed min" for i in columns]
    matrix = np.vstack([identity, -identity])
    upper = np.concatenate([observed_max[columns], -observed_min[columns]])
    return _from_rows(names, matrix, upper, condition_scale)


def specs_from_config(
    entries: Sequence[Mapping[str, object]],
) -> tuple[LinearConstraintSpec, ...]:
    """Build the specs from the config list.

    Args:
        entries: Mappings with the keys name, coefficients, and rocks.
            rocks is "all" or a list of rock labels.

    Returns:
        The specs.

    Raises:
        ValueError: If an entry has missing or unknown keys.
    """
    specs = []
    for entry in entries:
        if set(entry) != {"name", "coefficients", "rocks"}:
            raise ValueError(
                "Each constraint needs exactly the keys name, coefficients, "
                f"rocks. Got {sorted(entry)}."
            )
        rocks = entry["rocks"]
        rocks = (rocks,) if isinstance(rocks, str) else tuple(rocks)
        coefficients = entry["coefficients"]
        specs.append(
            LinearConstraintSpec(
                name=str(entry["name"]),
                coefficients=tuple(
                    (str(k), float(v)) for k, v in coefficients.items()
                ),
                rocks=rocks,
            )
        )
    return tuple(specs)
