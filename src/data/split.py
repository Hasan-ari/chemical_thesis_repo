from __future__ import annotations

import numpy as np


def split_trajectories(
    data: np.ndarray,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    """Split trajectories into train/test sets.

    Randomly assigns trajectories to train or test.
    Does NOT shuffle within trajectories.

    Args:
        data: shape (n_traj, n_steps, n_feat)
        test_ratio: fraction for test set
        seed: random seed for reproducibility

    Returns:
        (train_data, test_data, train_indices, test_indices)
    """
    n_traj = data.shape[0]
    n_test = int(n_traj * test_ratio)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_traj)

    test_indices = np.sort(indices[:n_test])
    train_indices = np.sort(indices[n_test:])

    train_data = data[train_indices]
    test_data = data[test_indices]

    print(f"Split: {len(train_indices)} train, {len(test_indices)} test trajectories")
    return train_data, test_data, train_indices, test_indices
