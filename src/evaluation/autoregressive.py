from __future__ import annotations

import numpy as np
import torch

from src.data.preprocessing import DataPreprocessor


def generate_trajectory(
    model: torch.nn.Module,
    initial_window: np.ndarray,
    n_steps: int,
    device: torch.device,
) -> np.ndarray:
    """Generate a full trajectory autoregressively.

    Uses the first seq_len steps as seed, then predicts step-by-step.
    Each prediction is fed back as input (no teacher forcing).

    Args:
        model: Trained LSTM model
        initial_window: shape (seq_len, n_features), normalized
        n_steps: total steps to generate (e.g. 101)
        device: torch device

    Returns:
        trajectory_norm: shape (n_steps, n_features), normalized.
        First seq_len rows are the seed (ground truth).
    """
    model.eval()
    seq_len = initial_window.shape[0]
    n_feat = initial_window.shape[1]

    trajectory = np.zeros((n_steps, n_feat))
    trajectory[:seq_len] = initial_window
    window = initial_window.copy()

    with torch.no_grad():
        for t in range(seq_len, n_steps):
            x = torch.FloatTensor(window).unsqueeze(0).to(device)
            pred = model(x).squeeze(0).cpu().numpy()
            trajectory[t] = pred
            window = np.vstack([window[1:], pred])

    return trajectory


def evaluate_on_test_set(
    model: torch.nn.Module,
    test_data_norm: np.ndarray,
    test_data_raw: np.ndarray,
    preprocessor: DataPreprocessor,
    seq_len: int,
    device: torch.device,
) -> dict:
    """Evaluate model on all test trajectories.

    For each test trajectory:
        1. Seed with first seq_len steps
        2. Autoregressively predict remaining steps
        3. Inverse-transform to original scale
        4. Compute RMSE against ground truth (only on predicted portion)

    Args:
        test_data_norm: shape (n_test, n_steps, n_feat), normalized
        test_data_raw: shape (n_test, n_steps, n_feat), original scale
        preprocessor: fitted DataPreprocessor for inverse transform
        seq_len: window length used for seeding
        device: torch device

    Returns:
        dict with rmse_per_var, rmse_total, rmse_per_trajectory, trajectories_pred
    """
    n_test, n_steps, n_feat = test_data_norm.shape
    all_pred = np.zeros_like(test_data_raw)
    all_pred_norm = np.zeros_like(test_data_norm)
    nrmse_per_traj = np.zeros(n_test)

    for i in range(n_test):
        # Generate autoregressive trajectory (normalized space)
        traj_norm = generate_trajectory(
            model, test_data_norm[i, :seq_len], n_steps, device
        )
        all_pred_norm[i] = traj_norm

        # Inverse transform to original scale
        traj_raw = preprocessor.inverse_transform(traj_norm)
        all_pred[i] = traj_raw

        # Per-trajectory RMSE in normalized space (unit-comparable)
        pred_norm = traj_norm[seq_len:]
        truth_norm = test_data_norm[i, seq_len:]
        nrmse_per_traj[i] = np.sqrt(np.mean((pred_norm - truth_norm) ** 2))

    # Per-variable RMSE in original scale (interpretable per variable)
    pred_all = all_pred[:, seq_len:]
    truth_all = test_data_raw[:, seq_len:]
    rmse_per_var = np.sqrt(np.mean((pred_all - truth_all) ** 2, axis=(0, 1)))

    # Total RMSE in normalized space (comparable across experiments)
    pred_norm_all = all_pred_norm[:, seq_len:]
    truth_norm_all = test_data_norm[:, seq_len:]
    nrmse_total = float(np.sqrt(np.mean((pred_norm_all - truth_norm_all) ** 2)))

    return {
        "rmse_per_var": rmse_per_var.tolist(),
        "nrmse_total": nrmse_total,
        "nrmse_per_trajectory": nrmse_per_traj.tolist(),
        "trajectories_pred": all_pred,
    }
