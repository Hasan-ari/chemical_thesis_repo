"""The H2 loss objectives that the search minimizes.

Pattern: Strategy. RelativeLoss and AbsoluteLoss share the Objective
interface. build_objective is a small Factory that selects one by name.

Each objective works on numpy arrays and on torch tensors, because it uses
only subtraction, division, and element-wise maximum.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, TypeVar

import numpy as np
import torch

ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)


@dataclasses.dataclass(frozen=True)
class TimeWindow:
    """The two time indices of the loss.

    Attributes:
        start_index: The index of t_start on the time axis.
        end_index: The index of t_end on the time axis.

    Raises:
        ValueError: If start_index < 0 or end_index <= start_index.
    """

    start_index: int
    end_index: int

    def __post_init__(self) -> None:
        """Check the order of the indices."""
        if self.start_index < 0 or self.end_index <= self.start_index:
            raise ValueError(
                "TimeWindow needs 0 <= start_index < end_index, got "
                f"{self.start_index} and {self.end_index}."
            )

    @classmethod
    def from_days(
        cls, time_days: np.ndarray, start_days: float, end_days: float
    ) -> TimeWindow:
        """Find the indices of two time values on the time axis.

        Args:
            time_days: The time axis, shape (T,), in days.
            start_days: t_start in days. It must be on the axis.
            end_days: t_end in days. It must be on the axis.

        Returns:
            The window.

        Raises:
            ValueError: If a value is not on the axis.
        """
        return cls(
            start_index=_index_of(time_days, start_days),
            end_index=_index_of(time_days, end_days),
        )


def _index_of(time_days: np.ndarray, value: float) -> int:
    matches = np.flatnonzero(np.isclose(time_days, value))
    if matches.size != 1:
        raise ValueError(
            f"Time {value} days must match exactly one point of the time "
            f"axis, found {matches.size}."
        )
    return int(matches[0])


class Objective(Protocol):
    """The interface of an H2 loss objective."""

    def __call__(self, h2: ArrayT) -> ArrayT:
        """Return the loss of each run. Smaller is better.

        Args:
            h2: The H2 gas amount, shape (B, T), in mol.

        Returns:
            The loss, shape (B,).
        """


@dataclasses.dataclass(frozen=True)
class RelativeLoss:
    """(H2(t_start) - H2(t_end)) / max(H2(t_start), eps).

    Pattern: Strategy (one Objective).

    Attributes:
        window: The two time indices.
        eps: The lower limit of the divisor, in mol. It prevents a division
            by zero.

    Raises:
        ValueError: If eps is not positive.
    """

    window: TimeWindow
    eps: float

    def __post_init__(self) -> None:
        """Check eps."""
        if not self.eps > 0:
            raise ValueError(f"eps must be positive, got {self.eps}.")

    def __call__(self, h2: ArrayT) -> ArrayT:
        """Return the relative loss, shape (B,). See Objective."""
        start = h2[:, self.window.start_index]
        end = h2[:, self.window.end_index]
        if isinstance(start, torch.Tensor):
            divisor = torch.clamp(start, min=self.eps)
        else:
            divisor = np.maximum(start, self.eps)
        return (start - end) / divisor


@dataclasses.dataclass(frozen=True)
class AbsoluteLoss:
    """H2(t_start) - H2(t_end), in mol.

    Pattern: Strategy (one Objective).

    Attributes:
        window: The two time indices.
    """

    window: TimeWindow

    def __call__(self, h2: ArrayT) -> ArrayT:
        """Return the absolute loss, shape (B,). See Objective."""
        return h2[:, self.window.start_index] - h2[:, self.window.end_index]


OBJECTIVE_NAMES = ("relative", "absolute")


def build_objective(name: str, window: TimeWindow, eps: float) -> Objective:
    """Build an objective from its config name.

    Pattern: Factory.

    Args:
        name: "relative" or "absolute".
        window: The two time indices.
        eps: The divisor limit of the relative loss, in mol.

    Returns:
        The objective.

    Raises:
        ValueError: If the name is unknown.
    """
    if name == "relative":
        return RelativeLoss(window=window, eps=eps)
    if name == "absolute":
        return AbsoluteLoss(window=window)
    raise ValueError(
        f"Unknown objective {name!r}. Expected one of {OBJECTIVE_NAMES}."
    )
