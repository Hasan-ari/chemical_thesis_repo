"""Select the real runs that start the search."""

from __future__ import annotations

import numpy as np


def select_starts(
    scores: np.ndarray,
    eligible: np.ndarray,
    *,
    n_best: int,
    n_random: int,
    seed: int,
) -> np.ndarray:
    """Select the best eligible runs, then random other eligible runs.

    The best runs put the search near known good points. The random runs
    cover the rest of the data, so the search can find other minima.

    Time: O(N log N) for the sort. Memory: O(N).

    Args:
        scores: The score of each run, shape (N,). Smaller is better.
        eligible: Shape (N,), bool. Only these runs can be a start.
        n_best: The number of eligible runs with the smallest score.
        n_random: The number of other eligible runs, drawn without
            replacement with numpy.random.default_rng(seed).
        seed: The seed of the random draw.

    Returns:
        The run indices, shape (n_best + n_random,), int64. The n_best
        runs come first, in ascending score order.

    Raises:
        ValueError: If a count is negative, the shapes do not agree, an
            eligible score is not finite, or there are too few eligible
            runs.
    """
    if n_best < 0 or n_random < 0:
        raise ValueError("n_best and n_random must be >= 0.")
    if scores.shape != eligible.shape or scores.ndim != 1:
        raise ValueError("scores and eligible must be 1D with equal shapes.")
    candidates = np.flatnonzero(eligible)
    if candidates.size < n_best + n_random:
        raise ValueError(
            f"Need {n_best + n_random} eligible runs, found {candidates.size}."
        )
    if not np.isfinite(scores[candidates]).all():
        raise ValueError("Each eligible run needs a finite score.")
    ordered = candidates[np.argsort(scores[candidates], kind="stable")]
    rng = np.random.default_rng(seed)
    others = rng.choice(ordered[n_best:], size=n_random, replace=False)
    return np.concatenate([ordered[:n_best], others]).astype(np.int64)
