"""Select the top candidates and remove near duplicates."""

from __future__ import annotations

import numpy as np

_MATRIX_NDIM = 2


def select_candidates(
    points: np.ndarray,
    scores: np.ndarray,
    usable: np.ndarray,
    *,
    min_distance: float,
    top_k: int,
) -> np.ndarray:
    """Select up to top_k usable points with a greedy distance filter.

    The usable points are visited in ascending score order. A point is kept
    if its Euclidean distance to each kept point is >= min_distance. Many
    starts converge to the same minimum, so this filter keeps different
    candidates. A point with a non-finite score or coordinate is not usable.

    Time: O(n log n + n * k * d) for n points, k = top_k, and d
    dimensions. Memory: O(n + k * d).

    Args:
        points: The candidate points, shape (n, d).
        scores: The score of each point, shape (n,). Smaller is better.
        usable: Shape (n,), bool. Only these points can be selected.
        min_distance: The minimum distance between two kept points, >= 0.
        top_k: The maximum number of kept points, >= 0.

    Returns:
        The indices of the kept points, in ascending score order, int64.

    Raises:
        ValueError: If the shapes do not agree or a limit is negative.
    """
    if points.ndim != _MATRIX_NDIM or points.shape[0] != scores.shape[0]:
        raise ValueError("points must have shape (n, d) with n = len(scores).")
    if usable.shape != scores.shape:
        raise ValueError("usable must have the shape of scores.")
    if min_distance < 0 or top_k < 0:
        raise ValueError("min_distance and top_k must be >= 0.")
    valid = usable & np.isfinite(scores) & np.isfinite(points).all(axis=1)
    order = np.flatnonzero(valid)
    order = order[np.argsort(scores[order], kind="stable")]
    kept: list[int] = []
    for index in order:
        if len(kept) == top_k:
            break
        distances = np.linalg.norm(points[kept] - points[index], axis=1)
        if (distances >= min_distance).all():
            kept.append(int(index))
    return np.asarray(kept, dtype=np.int64)
