from __future__ import annotations

import numpy as np
from typing import Optional

from src.data.constants import FEATURE_NAMES


def rmse_per_variable(
    pred: np.ndarray,
    truth: np.ndarray,
    feature_names: Optional[list] = None,
) -> dict:
    """Compute RMSE per variable.

    Args:
        pred: (n_steps, n_feat) or (n_traj, n_steps, n_feat)
        truth: same shape as pred
        feature_names: variable names for dict keys

    Returns:
        dict {var_name: rmse_value}
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    if pred.ndim == 3:
        # Average over trajectories and timesteps
        axis = (0, 1)
    else:
        axis = 0

    rmse = np.sqrt(np.mean((pred - truth) ** 2, axis=axis))
    return {name: float(val) for name, val in zip(feature_names, rmse)}


def rmse_total(pred: np.ndarray, truth: np.ndarray) -> float:
    """Compute overall RMSE across all variables and timesteps."""
    return float(np.sqrt(np.mean((pred - truth) ** 2)))
