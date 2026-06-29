from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def rock_aware_split(
    *,
    rocks: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> RunSplit:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = np.random.RandomState(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for rock in sorted(set(rocks.tolist())):
        indices = np.where(rocks == rock)[0]
        rng.shuffle(indices)
        n = len(indices)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3 and test_ratio > 0:
            n_test = max(n_test, 1)
        if n >= 3 and val_ratio > 0:
            n_val = max(n_val, 1)
        if n_test + n_val >= n:
            n_test = max(0, min(n_test, n - 2))
            n_val = max(0, min(n_val, n - n_test - 1))

        test_parts.append(indices[:n_test])
        val_parts.append(indices[n_test : n_test + n_val])
        train_parts.append(indices[n_test + n_val :])

    return RunSplit(
        train=np.sort(np.concatenate(train_parts)),
        val=np.sort(np.concatenate(val_parts)),
        test=np.sort(np.concatenate(test_parts)),
    )
