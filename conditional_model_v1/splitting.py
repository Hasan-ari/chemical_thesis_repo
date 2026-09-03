from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from conditional_model_v1.config import SplitConfig


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


def leave_one_rock_out_split(
    *,
    rocks: np.ndarray,
    held_out_rock: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> RunSplit:
    """Hold one rock out entirely; split the remaining rocks by the ratios.

    Every run of ``held_out_rock`` lands in ``test`` and never in ``train`` or
    ``val``. The remaining rocks are split with :func:`rock_aware_split`, so
    ``test`` also contains their ``test_ratio`` share (use ``test_ratio=0`` for a
    pure held-out-rock test set). Indices are returned sorted, like
    :func:`rock_aware_split`.
    """
    rocks = np.asarray(rocks)
    held_mask = rocks == held_out_rock
    if not held_mask.any():
        raise ValueError(f"held_out_rock {held_out_rock!r} has no runs in the bundle")
    if held_mask.all():
        raise ValueError("leave_one_rock_out needs at least one rock besides the held-out one")

    held_indices = np.flatnonzero(held_mask)
    remaining = np.flatnonzero(~held_mask)
    inner = rock_aware_split(
        rocks=rocks[remaining],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    return RunSplit(
        train=np.sort(remaining[inner.train]),
        val=np.sort(remaining[inner.val]),
        test=np.sort(np.concatenate([held_indices, remaining[inner.test]])),
    )


def build_split(
    split_config: SplitConfig,
    rocks: np.ndarray,
    *,
    held_out_rock: str | None = None,
) -> RunSplit:
    """Dispatch on ``split.strategy``; the rock-aware path is unchanged."""
    if split_config.strategy == "rock_aware_run_level":
        if held_out_rock is not None:
            raise ValueError("held_out_rock is only valid with split.strategy=leave_one_rock_out")
        return rock_aware_split(
            rocks=rocks,
            train_ratio=split_config.train,
            val_ratio=split_config.val,
            test_ratio=split_config.test,
            seed=split_config.seed,
        )
    if split_config.strategy == "leave_one_rock_out":
        if held_out_rock is None:
            raise ValueError("split.strategy=leave_one_rock_out requires a held_out_rock")
        return leave_one_rock_out_split(
            rocks=rocks,
            held_out_rock=held_out_rock,
            train_ratio=split_config.train,
            val_ratio=split_config.val,
            test_ratio=split_config.test,
            seed=split_config.seed,
        )
    raise ValueError(f"Unsupported split.strategy {split_config.strategy!r}")
