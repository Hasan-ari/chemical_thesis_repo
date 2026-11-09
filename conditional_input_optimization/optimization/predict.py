"""Predict one output for many real runs, in batches, without a gradient."""

from __future__ import annotations

import numpy as np
import torch

from conditional_input_optimization.optimization import (
    surrogate as surrogate_lib,
)


def predict_output(
    surrogate: surrogate_lib.FrozenSurrogate,
    conditions: np.ndarray,
    *,
    output_index: int,
    batch_size: int,
) -> np.ndarray:
    """Predict one output trajectory for each run.

    torch.no_grad does not keep the graph, so the memory is the activation
    memory of one batch only.

    Time: O(N * T * H^2). Memory: O(N * T) for the result plus
    O(batch_size * T * H) for one batch.

    Args:
        surrogate: The frozen surrogate.
        conditions: Physical inputs, shape (N, 57).
        output_index: The output column, in [0, 26).
        batch_size: The number of runs in one model call, > 0.

    Returns:
        The predicted output, shape (N, T), float64.

    Raises:
        ValueError: If batch_size is not positive or output_index is out of
            range.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if not 0 <= output_index < surrogate_lib.N_OUTPUTS:
        raise ValueError(f"output_index {output_index} is out of range.")
    n_runs = conditions.shape[0]
    result = np.empty((n_runs, len(surrogate.time_days)), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n_runs, batch_size):
            batch = torch.as_tensor(
                conditions[start : start + batch_size],
                dtype=surrogate.dtype,
                device=surrogate.device,
            )
            predicted = surrogate.predict(batch)[:, :, output_index]
            result[start : start + batch_size] = predicted.cpu().numpy()
    return result
