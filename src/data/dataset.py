from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrajectoryWindowDataset(Dataset):
    """Sliding window dataset across multiple trajectories.

    For each trajectory of length T with seq_len S:
      - Generates (T - S) windows: X[i:i+S] -> Y[i+S]
      - Windows do NOT cross trajectory boundaries

    Total samples: n_trajectories * (n_timesteps - seq_len)
    """

    def __init__(self, data_norm: np.ndarray, seq_len: int):
        """
        Args:
            data_norm: Normalized data, shape (n_traj, n_timesteps, n_features)
            seq_len: Sliding window length
        """
        n_traj, n_steps, n_feat = data_norm.shape
        windows_per_traj = n_steps - seq_len

        # Pre-extract all windows for maximum speed
        all_X = np.zeros((n_traj * windows_per_traj, seq_len, n_feat), dtype=np.float32)
        all_Y = np.zeros((n_traj * windows_per_traj, n_feat), dtype=np.float32)

        idx = 0
        for t in range(n_traj):
            for i in range(windows_per_traj):
                all_X[idx] = data_norm[t, i : i + seq_len]
                all_Y[idx] = data_norm[t, i + seq_len]
                idx += 1

        self.X = torch.from_numpy(all_X)
        self.Y = torch.from_numpy(all_Y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.Y[idx]


def create_dataloader(
    data_norm: np.ndarray,
    seq_len: int,
    batch_size: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader from normalized trajectory data."""
    dataset = TrajectoryWindowDataset(data_norm, seq_len)
    print(f"Dataset: {len(dataset)} windows (seq_len={seq_len}, batch_size={batch_size})")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
