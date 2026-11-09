"""Multi-start gradient search in z-space.

All starts are in one (B, K) tensor. The loss of the batch is the sum of
the start losses. A start loss depends only on its own row of z, so the
gradient of the sum, row b, is the gradient of start b alone. Thus the
starts do not interact, and one batch replaces B separate searches.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import torch

from conditional_input_optimization.optimization import projections

LossFunction = Callable[[torch.Tensor], torch.Tensor]
OptimizerFactory = Callable[[list[torch.Tensor]], torch.optim.Optimizer]

NO_STEP = -1
_MATRIX_NDIM = 2


@dataclasses.dataclass(frozen=True)
class SearchSchedule:
    """The optimizer and the number of steps.

    Attributes:
        optimizer_factory: Builds the optimizer from the parameter list,
            for example functools.partial(torch.optim.Adam, lr=0.05).
        n_steps: The number of optimizer steps. 0 evaluates the starts only.

    Raises:
        ValueError: If n_steps is negative.
    """

    optimizer_factory: OptimizerFactory
    n_steps: int

    def __post_init__(self) -> None:
        """Check n_steps."""
        if self.n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {self.n_steps}.")


@dataclasses.dataclass(frozen=True, eq=False)
class SearchResult:
    """The best point of each start over all steps.

    Attributes:
        best_z: The best point of each start, shape (B, K).
        best_loss: The loss at best_z, shape (B,). It is inf if the loss
            of the start was never finite.
        best_step: The step of best_z, shape (B,), int. -1 if none.
        first_nonfinite_step: The first step with a non-finite loss,
            shape (B,), int. -1 if the loss was always finite.
        history: The loss of each start at each step, shape
            (n_steps + 1, B), float32.

    Raises:
        ValueError: If the shapes do not agree.
    """

    best_z: np.ndarray
    best_loss: np.ndarray
    best_step: np.ndarray
    first_nonfinite_step: np.ndarray
    history: np.ndarray

    def __post_init__(self) -> None:
        """Check the shapes and make the arrays read-only."""
        n_starts = self.best_z.shape[0]
        for name in ("best_loss", "best_step", "first_nonfinite_step"):
            if getattr(self, name).shape != (n_starts,):
                raise ValueError(f"{name} must have shape ({n_starts},).")
        if (
            self.history.ndim != _MATRIX_NDIM
            or self.history.shape[1] != n_starts
        ):
            raise ValueError(f"history must have shape (S, {n_starts}).")
        for field in dataclasses.fields(self):
            value = np.array(getattr(self, field.name), copy=True)
            value.flags.writeable = False
            object.__setattr__(self, field.name, value)


@dataclasses.dataclass
class _Tracker:
    """The mutable best-so-far state of one search. Internal only."""

    best_z: torch.Tensor
    best_loss: torch.Tensor
    best_step: torch.Tensor
    first_nonfinite_step: torch.Tensor
    history: torch.Tensor

    def update(self, step: int, z: torch.Tensor, loss: torch.Tensor) -> None:
        """Record the loss of one step. Keep the best finite point."""
        finite = torch.isfinite(loss)
        self.history[step] = loss
        improved = finite & (loss < self.best_loss)
        self.best_loss = torch.where(improved, loss, self.best_loss)
        self.best_z[improved] = z[improved]
        self.best_step[improved] = step
        new_bad = ~finite & (self.first_nonfinite_step == NO_STEP)
        self.first_nonfinite_step[new_bad] = step

    def result(self) -> SearchResult:
        """Return the frozen result as numpy arrays."""
        return SearchResult(
            best_z=self.best_z.cpu().numpy(),
            best_loss=self.best_loss.cpu().numpy(),
            best_step=self.best_step.cpu().numpy(),
            first_nonfinite_step=self.first_nonfinite_step.cpu().numpy(),
            history=self.history.cpu().numpy(),
        )


def _new_tracker(z0: torch.Tensor, n_steps: int) -> _Tracker:
    n_starts = z0.shape[0]
    steps = torch.full((n_starts,), NO_STEP, device=z0.device)
    return _Tracker(
        best_z=z0.clone(),
        best_loss=torch.full(
            (n_starts,), torch.inf, dtype=z0.dtype, device=z0.device
        ),
        best_step=steps.clone(),
        first_nonfinite_step=steps,
        history=torch.empty(
            (n_steps + 1, n_starts), dtype=torch.float32, device=z0.device
        ),
    )


def search(
    loss_fn: LossFunction,
    z0: torch.Tensor,
    projection: projections.Projection,
    schedule: SearchSchedule,
    *,
    control_mask: torch.Tensor | None = None,
) -> SearchResult:
    """Minimize the loss of each start with a gradient optimizer.

    The loop evaluates step 0 (the projected starts) to step n_steps. After
    each evaluation except the last one, it does one optimizer step and
    applies the projection. A start with a non-finite loss gets a zero
    gradient, so it does not move and it cannot affect the optimizer.

    Time: O(n_steps * cost of loss_fn and its backward pass).
    Memory: O(n_steps * B) for the history, plus one loss graph.

    Args:
        loss_fn: Maps z, shape (B, K), to the loss of each start, shape
            (B,). Row b of the loss must depend only on row b of z.
        z0: The start points, shape (B, K).
        projection: The rule after each step, for example a box clamp.
        schedule: The optimizer factory and the number of steps.
        control_mask: Shape (K,), bool. False keeps the column at its z0
            value. None lets all columns move.

    Returns:
        The best point of each start.

    Raises:
        ValueError: If z0 is not 2D or the mask shape is wrong.
    """
    if z0.ndim != _MATRIX_NDIM:
        raise ValueError(f"z0 must have shape (B, K), got {tuple(z0.shape)}.")
    z0 = z0.detach()
    if control_mask is None:
        control_mask = torch.ones(z0.shape[1], dtype=torch.bool)
    if control_mask.shape != (z0.shape[1],):
        raise ValueError(f"control_mask must have shape ({z0.shape[1]},).")
    control_mask = control_mask.to(z0.device)
    z_var = z0.clone().requires_grad_(True)
    optimizer = schedule.optimizer_factory([z_var])
    projection(z_var)
    tracker = _new_tracker(z0, schedule.n_steps)
    for step in range(schedule.n_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        z_full = torch.where(control_mask, z_var, z0)
        loss = loss_fn(z_full)
        tracker.update(step, z_full.detach(), loss.detach())
        if step == schedule.n_steps:
            break
        finite = torch.isfinite(loss)
        torch.where(finite, loss, torch.zeros_like(loss)).sum().backward()
        with torch.no_grad():
            # 0 * inf in the backward pass of a bad row gives NaN.
            z_var.grad[~finite] = 0.0
        optimizer.step()
        projection(z_var)
    return tracker.result()
