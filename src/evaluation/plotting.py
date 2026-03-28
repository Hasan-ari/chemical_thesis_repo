from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from src.data.constants import FEATURE_NAMES


def plot_trajectory_comparison(
    pred: np.ndarray,
    truth: np.ndarray,
    time_axis: np.ndarray,
    seq_len: int,
    feature_names: Optional[list] = None,
    save_path: Optional[Path] = None,
    title: str = "",
) -> None:
    """Plot predicted vs ground truth for all 12 variables.

    Creates a 4x3 subplot grid. Vertical dashed line marks
    the boundary between seed and predicted regions.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    n_feat = pred.shape[1]
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(n_feat):
        ax = axes[i]
        ax.plot(time_axis, truth[:, i], "b-", label="Ground truth", linewidth=1.5)
        ax.plot(time_axis, pred[:, i], "r--", label="Predicted", linewidth=1.2)
        ax.axvline(x=time_axis[seq_len], color="gray", linestyle=":", alpha=0.7)
        ax.set_title(feature_names[i], fontsize=10)
        ax.set_xlabel("Time (days)", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle(title or "Trajectory Comparison", fontsize=13)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_loss_curve(
    loss_history: list,
    save_path: Optional[Path] = None,
    title: str = "",
) -> None:
    """Plot training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_history, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_yscale("log")
    ax.set_title(title or "Training Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)


def plot_experiment_matrix_heatmap(
    results: dict,
    save_path: Optional[Path] = None,
) -> None:
    """Heatmap of RMSE across (seq_len, hidden_size) combinations."""
    # Extract unique values
    seq_lens = sorted(set(r["seq_len"] for r in results.values()))
    hidden_sizes = sorted(set(r["hidden_size"] for r in results.values()))

    # Build matrix
    matrix = np.full((len(hidden_sizes), len(seq_lens)), np.nan)
    for r in results.values():
        row = hidden_sizes.index(r["hidden_size"])
        col = seq_lens.index(r["seq_len"])
        matrix[row, col] = r["rmse_total"]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(range(len(seq_lens)))
    ax.set_xticklabels([str(s) for s in seq_lens])
    ax.set_yticks(range(len(hidden_sizes)))
    ax.set_yticklabels([str(h) for h in hidden_sizes])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Hidden Size")
    ax.set_title("Autoregressive RMSE (test set)")

    # Annotate cells
    for i in range(len(hidden_sizes)):
        for j in range(len(seq_lens)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                        fontsize=10, fontweight="bold")

    fig.colorbar(im, ax=ax, label="RMSE")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)
