"""The search space of one rock: free, tied, and fixed inputs.

The generator script of a rock computes some inputs from other inputs (for
example CL = NA). detect_ties finds these tied inputs with a linear test.
RockInputSpace then maps a point of the free inputs (in z-space) to all 57
inputs with one affine map: x = base + z @ weights.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import torch

ROLE_FREE = "free"
ROLE_TIED = "tied"
ROLE_FIXED = "fixed"


@dataclasses.dataclass(frozen=True)
class Tie:
    """One tied input: x[column] = intercept + x[sources] @ coefficients.

    Attributes:
        column: The index of the tied input.
        sources: The indices of the free inputs that it depends on.
        intercept: The constant term, in the unit of the tied input.
        coefficients: One coefficient for each source.
        r_squared: The R² of the least-squares fit, in [0, 1].
    """

    column: int
    sources: tuple[int, ...]
    intercept: float
    coefficients: tuple[float, ...]
    r_squared: float

    def __post_init__(self) -> None:
        """Check that each source has one coefficient."""
        if len(self.sources) != len(self.coefficients):
            raise ValueError("Each source needs one coefficient.")
        if self.column in self.sources:
            raise ValueError("A tied input cannot be its own source.")


@dataclasses.dataclass(frozen=True)
class TieDetection:
    """The result of detect_ties.

    Attributes:
        free_columns: The free input indices, in feature order.
        ties: The tied inputs, in feature order.
        fixed_columns: The inputs with one value in all runs.
    """

    free_columns: tuple[int, ...]
    ties: tuple[Tie, ...]
    fixed_columns: tuple[int, ...]


def detect_ties(conditions: np.ndarray, r2_threshold: float) -> TieDetection:
    """Find the fixed, tied, and free inputs of one rock.

    The test is greedy, in feature order. Each varying input is fitted on
    [1, current free inputs] with least squares. If R² >= r2_threshold, the
    input is tied. Otherwise it becomes free. Thus the result depends on
    the feature order: of two equal inputs, the first one is free.

    Time: O(V * N * K^2) for V varying inputs, N runs, and K free inputs.
    Memory: O(N * K).

    Args:
        conditions: The inputs of the runs of one rock, shape (N, 57).
        r2_threshold: The R² limit for a tie, in (0, 1].

    Returns:
        The roles of the inputs.

    Raises:
        ValueError: If the threshold is outside (0, 1] or there are fewer
            than 2 runs.
    """
    if not 0 < r2_threshold <= 1:
        raise ValueError(f"r2_threshold must be in (0, 1], got {r2_threshold}.")
    if conditions.ndim != 2 or conditions.shape[0] < 2:  # noqa: PLR2004
        raise ValueError("conditions must have shape (N, D) with N >= 2.")
    varying = np.ptp(conditions, axis=0) > 0
    free: list[int] = []
    ties: list[Tie] = []
    for column in np.flatnonzero(varying):
        tie = _fit_tie(conditions, int(column), free) if free else None
        if tie is not None and tie.r_squared >= r2_threshold:
            ties.append(tie)
        else:
            free.append(int(column))
    return TieDetection(
        free_columns=tuple(free),
        ties=tuple(ties),
        fixed_columns=tuple(int(i) for i in np.flatnonzero(~varying)),
    )


def _fit_tie(conditions: np.ndarray, column: int, sources: list[int]) -> Tie:
    """Fit one input on the sources. Standardize first for a stable fit."""
    source_values = conditions[:, sources]
    source_mean = source_values.mean(axis=0)
    source_std = source_values.std(axis=0)
    target = conditions[:, column]
    target_mean = target.mean()
    target_std = target.std()
    design = np.column_stack(
        [np.ones(len(target)), (source_values - source_mean) / source_std]
    )
    target_z = (target - target_mean) / target_std
    solution, *_ = np.linalg.lstsq(design, target_z, rcond=None)
    residual = target_z - design @ solution
    r_squared = 1.0 - float(residual @ residual) / float(target_z @ target_z)
    coefficients = target_std * solution[1:] / source_std
    intercept = (
        target_mean + target_std * solution[0] - coefficients @ source_mean
    )
    return Tie(
        column=column,
        sources=tuple(sources),
        intercept=float(intercept),
        coefficients=tuple(float(c) for c in coefficients),
        r_squared=r_squared,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class RockInputSpace:
    """The affine map from z-space to the 57 inputs of one rock.

    z-space is the free inputs after the rock z-score:
    z = (x[free] - free_mean) / free_std.

    Attributes:
        rock: The rock label. It is metadata only.
        detection: The input roles.
        free_mean: The mean of each free input, shape (K,).
        free_std: The std of each free input, shape (K,). All positive.
        observed_min: The min of each input over the rock runs, shape (57,).
        observed_max: The max of each input over the rock runs, shape (57,).
        base: The decoded inputs at z = 0, shape (57,).
        weights: The decode matrix, shape (K, 57).

    Raises:
        ValueError: If a shape does not match the detection.
    """

    rock: str
    detection: TieDetection
    free_mean: np.ndarray
    free_std: np.ndarray
    observed_min: np.ndarray
    observed_max: np.ndarray
    base: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        """Check the shapes and make the arrays read-only."""
        k = len(self.detection.free_columns)
        d = self.observed_min.shape[0]
        expected = {
            "free_mean": (k,),
            "free_std": (k,),
            "observed_min": (d,),
            "observed_max": (d,),
            "base": (d,),
            "weights": (k, d),
        }
        for name, shape in expected.items():
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}.")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if not (self.free_std > 0).all():
            raise ValueError("free_std must be positive.")

    @property
    def n_free(self) -> int:
        """The number of free inputs (K)."""
        return len(self.detection.free_columns)

    def encode(self, conditions: np.ndarray) -> np.ndarray:
        """Map real inputs to z-space.

        Args:
            conditions: Inputs, shape (B, 57).

        Returns:
            The z values, shape (B, K).
        """
        free = conditions[:, list(self.detection.free_columns)]
        return (free - self.free_mean) / self.free_std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map z-space points to the 57 inputs. Differentiable.

        Time: O(B * K * 57).

        Args:
            z: Points, shape (B, K).

        Returns:
            The inputs, shape (B, 57), with the dtype and device of z.
        """
        base = torch.tensor(self.base, dtype=z.dtype, device=z.device)
        weights = torch.tensor(self.weights, dtype=z.dtype, device=z.device)
        return base + z @ weights

    def free_bounds_z(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the observed range of the free inputs in z-space.

        Returns:
            The lower and the upper limit, each shape (K,).
        """
        free = list(self.detection.free_columns)
        lower = (self.observed_min[free] - self.free_mean) / self.free_std
        upper = (self.observed_max[free] - self.free_mean) / self.free_std
        return lower, upper

    def roles(self) -> list[str]:
        """Return the role of each of the 57 inputs."""
        roles = [ROLE_FIXED] * self.base.shape[0]
        for column in self.detection.free_columns:
            roles[column] = ROLE_FREE
        for tie in self.detection.ties:
            roles[tie.column] = ROLE_TIED
        return roles


def build_input_space(
    rock: str, conditions: np.ndarray, r2_threshold: float
) -> RockInputSpace:
    """Detect the input roles of one rock and build its affine map.

    Pattern: Factory.

    Args:
        rock: The rock label. It is metadata only.
        conditions: The inputs of the runs of the rock, shape (N, 57).
        r2_threshold: The R² limit for a tie.

    Returns:
        The input space of the rock.
    """
    detection = detect_ties(conditions, r2_threshold)
    free = list(detection.free_columns)
    free_mean = conditions[:, free].mean(axis=0)
    free_std = conditions[:, free].std(axis=0)
    base, weights = _affine_map(conditions[0], detection, free_mean, free_std)
    return RockInputSpace(
        rock=rock,
        detection=detection,
        free_mean=free_mean,
        free_std=free_std,
        observed_min=conditions.min(axis=0),
        observed_max=conditions.max(axis=0),
        base=base,
        weights=weights,
    )


def _affine_map(
    first_run: np.ndarray,
    detection: TieDetection,
    free_mean: np.ndarray,
    free_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build base (57,) and weights (K, 57) of x = base + z @ weights."""
    free = list(detection.free_columns)
    position = {column: k for k, column in enumerate(free)}
    base = first_run.astype(np.float64).copy()
    weights = np.zeros((len(free), base.shape[0]))
    base[free] = free_mean
    weights[np.arange(len(free)), free] = free_std
    for tie in detection.ties:
        rows = [position[source] for source in tie.sources]
        coefficients = np.asarray(tie.coefficients)
        base[tie.column] = tie.intercept + coefficients @ free_mean[rows]
        weights[rows, tie.column] = coefficients * free_std[rows]
    return base, weights


def role_table(spaces: Sequence[RockInputSpace]) -> list[dict[str, object]]:
    """Return one row for each rock and input: the role and the tie fit.

    Args:
        spaces: The input spaces.

    Returns:
        Rows with the keys rock, column, role, r_squared, and sources.
    """
    rows: list[dict[str, object]] = []
    for space in spaces:
        ties = {tie.column: tie for tie in space.detection.ties}
        for column, role in enumerate(space.roles()):
            tie = ties.get(column)
            rows.append(
                {
                    "rock": space.rock,
                    "column": column,
                    "role": role,
                    "r_squared": tie.r_squared if tie else np.nan,
                    "sources": list(tie.sources) if tie else [],
                }
            )
    return rows
