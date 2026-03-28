from __future__ import annotations

import numpy as np
from pathlib import Path

from src.data.constants import DATA_DIR, N_TRAJECTORIES, N_TIMESTEPS, N_FEATURES


def load_single_trajectory(filepath: Path) -> np.ndarray:
    """Load one output .txt file.

    Returns array of shape (N_TIMESTEPS, N_FEATURES) — features only, no time_d.
    """
    data = np.loadtxt(filepath, skiprows=1)  # skip header
    return data[:, 1:]  # drop time_d column


def load_all_trajectories(
    output_dir: Path | None = None,
    n_trajectories: int = N_TRAJECTORIES,
) -> np.ndarray:
    """Load all output files.

    Returns array of shape (n_trajectories, N_TIMESTEPS, N_FEATURES).
    """
    if output_dir is None:
        output_dir = DATA_DIR / "output"

    all_data = np.zeros((n_trajectories, N_TIMESTEPS, N_FEATURES))
    for i in range(n_trajectories):
        filepath = output_dir / f"{i + 1}_Output.txt"
        all_data[i] = load_single_trajectory(filepath)

    print(f"Loaded {n_trajectories} trajectories | shape={all_data.shape}")
    return all_data


def load_time_axis(output_dir: Path | None = None) -> np.ndarray:
    """Load time_d column from the first output file. Returns shape (N_TIMESTEPS,)."""
    if output_dir is None:
        output_dir = DATA_DIR / "output"
    data = np.loadtxt(output_dir / "1_Output.txt", skiprows=1)
    return data[:, 0]
