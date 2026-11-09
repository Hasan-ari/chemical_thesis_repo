"""The rule that the search applies to z after each optimizer step.

Pattern: Strategy. BoxProjection and NoProjection share the Projection
interface. build_projection is a small Factory that selects one by mode.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

import torch

from conditional_input_optimization.optimization import input_space

MODE_BOUNDED = "bounded"
MODE_UNBOUNDED = "unbounded"
MODE_NAMES = (MODE_BOUNDED, MODE_UNBOUNDED)


class Projection(Protocol):
    """The interface of a projection rule."""

    def __call__(self, z: torch.Tensor) -> None:
        """Change z in place so that it obeys the rule.

        Args:
            z: The search points, shape (B, K). It can need a gradient.
        """


@dataclasses.dataclass(frozen=True, eq=False)
class BoxProjection:
    """Clamp each z column to [lower, upper].

    Pattern: Strategy (one Projection).

    Time: O(B * K).

    Attributes:
        lower: The lower limit of each column, shape (K,).
        upper: The upper limit of each column, shape (K,).

    Raises:
        ValueError: If the shapes are not equal and 1D, or lower > upper.
    """

    lower: torch.Tensor
    upper: torch.Tensor

    def __post_init__(self) -> None:
        """Check the shapes and the order of the limits."""
        if self.lower.ndim != 1 or self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must be 1D with equal shapes.")
        if bool((self.lower > self.upper).any()):
            raise ValueError("lower must be <= upper in each column.")

    def __call__(self, z: torch.Tensor) -> None:
        """Clamp z in place. See Projection."""
        with torch.no_grad():
            z.clamp_(min=self.lower, max=self.upper)


class NoProjection:
    """Do not change z.

    Pattern: Strategy (one Projection).
    """

    def __call__(self, z: torch.Tensor) -> None:
        """Do nothing. See Projection."""


def build_projection(
    mode: str,
    space: input_space.RockInputSpace,
    dtype: torch.dtype,
    device: torch.device,
) -> Projection:
    """Build the projection of a search mode.

    Pattern: Factory.

    Args:
        mode: "bounded" (clamp the free inputs to the observed range of the
            rock) or "unbounded" (no clamp).
        space: The input space of the rock.
        dtype: The dtype of z.
        device: The device of z.

    Returns:
        The projection.

    Raises:
        ValueError: If the mode is unknown.
    """
    if mode == MODE_BOUNDED:
        lower, upper = space.free_bounds_z()
        return BoxProjection(
            lower=torch.tensor(lower, dtype=dtype, device=device),
            upper=torch.tensor(upper, dtype=dtype, device=device),
        )
    if mode == MODE_UNBOUNDED:
        return NoProjection()
    raise ValueError(f"Unknown mode {mode!r}. Expected one of {MODE_NAMES}.")
