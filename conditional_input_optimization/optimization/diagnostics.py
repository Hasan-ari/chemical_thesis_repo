"""Checks of the surrogate and of the distance from a candidate to the data.

The search trusts the surrogate. These checks tell a reader how much trust
is reasonable: how well the surrogate ranks real runs of a rock, and how
far a candidate is from the runs that trained the surrogate.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import torch

from conditional_input_optimization.optimization import (
    surrogate as surrogate_lib,
)

TOP_FRACTION = 0.1
ERROR_PERCENTILE = 95.0
_MIN_RUNS = 2
_SPEARMAN_LIMIT = 1.0
_PERCENT = 100.0


@dataclasses.dataclass(frozen=True)
class AcceptanceRule:
    """The limits for a resolvable surrogate in one rock.

    A rock is resolvable if the surrogate ranks its runs well (high
    Spearman) and its error is small compared with the spread of the real
    values (low MAE / std).

    Attributes:
        min_spearman: The minimum Spearman correlation, in [-1, 1].
        max_mae_over_std: The maximum MAE / std of the real values, > 0.

    Raises:
        ValueError: If a limit is out of range.
    """

    min_spearman: float
    max_mae_over_std: float

    def __post_init__(self) -> None:
        """Check the limits."""
        if not -_SPEARMAN_LIMIT <= self.min_spearman <= _SPEARMAN_LIMIT:
            raise ValueError("min_spearman must be in [-1, 1].")
        if not self.max_mae_over_std > 0:
            raise ValueError("max_mae_over_std must be positive.")


def regression_metrics(
    predicted: np.ndarray, real: np.ndarray
) -> dict[str, float]:
    """Return the MAE, the R², and the Spearman correlation.

    Time: O(n log n) for the ranks of the Spearman correlation.

    Args:
        predicted: The predicted values, shape (n,).
        real: The real values, shape (n,).

    Returns:
        The keys mae, r2, and spearman. r2 = 1 - SS_res / SS_tot. It is
        NaN if the real values are constant. spearman is NaN if one of the
        arrays is constant.

    Raises:
        ValueError: If the shapes differ or n < 2.
    """
    _check_pair(predicted, real)
    residual = predicted - real
    total = float(((real - real.mean()) ** 2).sum())
    r2 = 1.0 - float((residual**2).sum()) / total if total > 0 else math.nan
    # A constant array has no ranks. Do not call scipy, which warns then.
    constant = np.ptp(predicted) == 0 or np.ptp(real) == 0
    spearman = (
        math.nan
        if constant
        else pd.Series(predicted).corr(pd.Series(real), method="spearman")
    )
    return {
        "mae": float(np.abs(residual).mean()),
        "r2": r2,
        "spearman": float(spearman),
    }


def _check_pair(predicted: np.ndarray, real: np.ndarray) -> None:
    if predicted.shape != real.shape or predicted.ndim != 1:
        raise ValueError("predicted and real must be 1D with equal shapes.")
    if predicted.shape[0] < _MIN_RUNS:
        raise ValueError(f"Need at least {_MIN_RUNS} values.")
    if not (np.isfinite(predicted).all() and np.isfinite(real).all()):
        raise ValueError("predicted and real must be finite.")


def surrogate_check(
    predicted: np.ndarray,
    real: np.ndarray,
    *,
    n_top: int,
    acceptance: AcceptanceRule,
) -> dict[str, float | int | bool]:
    """Compare the predicted and the real objective of one group of runs.

    Time: O(n log n). Memory: O(n).

    Args:
        predicted: The predicted objective, shape (n,).
        real: The real objective, shape (n,).
        n_top: The number of runs with the lowest prediction for the
            optimism gap, > 0. It is reduced to n if n is smaller.
        acceptance: The limits for the resolvable flag.

    Returns:
        The keys n, mae, real_std, mae_over_std, r2, spearman,
        top_decile_precision (of the 10% lowest predictions, the fraction
        that is in the 10% lowest real values), share_predicted_negative,
        abs_error_p95, optimism_gap_top (mean real - mean predicted over
        the n_top lowest predictions; > 0 means the surrogate is too
        optimistic there), and resolvable.

    Raises:
        ValueError: If the shapes differ, n < 2, or n_top <= 0.
    """
    if n_top <= 0:
        raise ValueError(f"n_top must be positive, got {n_top}.")
    metrics = regression_metrics(predicted, real)
    real_std = float(real.std())
    mae_over_std = metrics["mae"] / real_std if real_std > 0 else math.inf
    order = np.argsort(predicted, kind="stable")
    top = order[: min(n_top, len(order))]
    return {
        "n": int(predicted.shape[0]),
        **metrics,
        "real_std": real_std,
        "mae_over_std": mae_over_std,
        "top_decile_precision": _top_fraction_precision(predicted, real),
        "share_predicted_negative": float((predicted < 0).mean()),
        "abs_error_p95": float(
            np.percentile(np.abs(predicted - real), ERROR_PERCENTILE)
        ),
        "optimism_gap_top": float(real[top].mean() - predicted[top].mean()),
        "resolvable": bool(
            metrics["spearman"] >= acceptance.min_spearman
            and mae_over_std <= acceptance.max_mae_over_std
        ),
    }


def _top_fraction_precision(predicted: np.ndarray, real: np.ndarray) -> float:
    count = max(1, math.ceil(TOP_FRACTION * predicted.shape[0]))
    predicted_top = np.argsort(predicted, kind="stable")[:count]
    real_top = np.argsort(real, kind="stable")[:count]
    return float(np.isin(predicted_top, real_top).mean())


def nearest_distances(
    points: np.ndarray, reference: np.ndarray, *, chunk_size: int
) -> np.ndarray:
    """Return the Euclidean distance from each point to its nearest reference.

    Time: O(n * m * d) for n points and m references.
    Memory: O(chunk_size * m) for one block of the distance matrix.

    Args:
        points: Shape (n, d).
        reference: Shape (m, d), m >= 1.
        chunk_size: The number of points in one torch.cdist call, > 0.

    Returns:
        The distances, shape (n,), float64.

    Raises:
        ValueError: If the reference is empty or a shape is wrong.
    """
    return _kth_smallest_distances(points, reference, 1, chunk_size)


def reference_nn_distances(
    reference: np.ndarray, *, sample_size: int, seed: int, chunk_size: int
) -> np.ndarray:
    """Return the leave-one-out nearest-neighbor distance of a subsample.

    The typical distance between a reference run and its nearest other run
    is the scale for "near the data". The second smallest distance of a
    row skips the distance of the run to itself.

    Time: O(s * m * d) for s = min(sample_size, m). Memory:
    O(chunk_size * m).

    Args:
        reference: Shape (m, d), m >= 2.
        sample_size: The maximum number of sampled reference rows, > 0.
        seed: The seed of numpy.random.default_rng for the subsample.
        chunk_size: The number of rows in one torch.cdist call, > 0.

    Returns:
        The distances, shape (s,), float64.

    Raises:
        ValueError: If m < 2 or sample_size <= 0.
    """
    if reference.shape[0] < _MIN_RUNS or sample_size <= 0:
        raise ValueError("Need m >= 2 reference rows and sample_size > 0.")
    rng = np.random.default_rng(seed)
    size = min(sample_size, reference.shape[0])
    sample = rng.choice(reference.shape[0], size=size, replace=False)
    return _kth_smallest_distances(reference[sample], reference, 2, chunk_size)


def _kth_smallest_distances(
    points: np.ndarray, reference: np.ndarray, k: int, chunk_size: int
) -> np.ndarray:
    """Return the k-th smallest distance of each point, with chunks."""
    if points.ndim != reference.ndim or points.shape[1:] != reference.shape[1:]:
        raise ValueError("points and reference must have shape (n, d), (m, d).")
    if reference.shape[0] < k or chunk_size <= 0:
        raise ValueError(f"Need >= {k} reference rows and chunk_size > 0.")
    reference_tensor = torch.as_tensor(reference, dtype=torch.float64)
    result = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], chunk_size):
        block = torch.as_tensor(
            points[start : start + chunk_size], dtype=torch.float64
        )
        distances = torch.cdist(block, reference_tensor)
        values = torch.topk(distances, k, dim=1, largest=False).values
        result[start : start + chunk_size] = values[:, k - 1].numpy()
    return result


def distance_percentile(
    distances: np.ndarray, reference_nn: np.ndarray
) -> np.ndarray:
    """Return the percent of reference_nn values <= each distance.

    A value of 50 means a typical distance between real runs. A value of
    100 means that the point is farther than all sampled real runs are
    from their nearest neighbor.

    Time: O((n + s) log s).

    Args:
        distances: Shape (n,).
        reference_nn: The reference distances, shape (s,), s >= 1.

    Returns:
        The percentiles, shape (n,), in [0, 100].

    Raises:
        ValueError: If reference_nn is empty.
    """
    if reference_nn.size == 0:
        raise ValueError("reference_nn must not be empty.")
    ordered = np.sort(reference_nn)
    counts = np.searchsorted(ordered, distances, side="right")
    return _PERCENT * counts / ordered.size


def scaler_mean_gap(
    train_conditions: np.ndarray, scaling: surrogate_lib.SurrogateScaling
) -> float:
    """Return max |mean(train) - condition_mean| / condition_scale.

    The v2 condition scaler was fitted on the v2 train runs. A gap near 0
    shows that the rebuilt split selects the same train runs.

    Time: O(N * 57).

    Args:
        train_conditions: The inputs of the train runs, shape (N, 57).
        scaling: The v2 scaling values.

    Returns:
        The largest gap over the 57 inputs, in condition_scale units.
    """
    gap = np.abs(train_conditions.mean(axis=0) - scaling.condition_mean)
    return float((gap / scaling.condition_scale).max())
