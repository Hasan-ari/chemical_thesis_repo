"""Comprehensive evaluation across ALL test trajectories.

Produces aggregate statistics and detailed visualizations that go beyond
single best/worst trajectory plots. Designed to answer:
"What is the average prediction error across all test trajectories?"
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.data.constants import FEATURE_NAMES, N_FEATURES
from src.data.loader import load_all_trajectories, load_time_axis
from src.data.split import split_trajectories
from src.data.preprocessing import DataPreprocessor
from src.models.lstm import PhreeqcLSTM
from src.training.config import ExperimentConfig
from src.training.trainer import setup_device
from src.evaluation.autoregressive import generate_trajectory


def evaluate_all_trajectories(
    model: torch.nn.Module,
    test_norm: np.ndarray,
    test_raw: np.ndarray,
    preprocessor: DataPreprocessor,
    seq_len: int,
    device: torch.device,
) -> dict:
    """Full evaluation on all test trajectories, returning detailed per-step errors."""
    n_test, n_steps, n_feat = test_norm.shape
    n_pred_steps = n_steps - seq_len

    all_pred_raw = np.zeros((n_test, n_steps, n_feat))
    # Per-trajectory, per-timestep, per-variable absolute error
    abs_errors = np.zeros((n_test, n_pred_steps, n_feat))
    rmse_per_traj = np.zeros(n_test)

    for i in range(n_test):
        traj_norm = generate_trajectory(
            model, test_norm[i, :seq_len], n_steps, device
        )
        traj_raw = preprocessor.inverse_transform(traj_norm)
        all_pred_raw[i] = traj_raw

        pred_portion = traj_raw[seq_len:]
        truth_portion = test_raw[i, seq_len:]
        abs_errors[i] = np.abs(pred_portion - truth_portion)
        rmse_per_traj[i] = np.sqrt(np.mean((pred_portion - truth_portion) ** 2))

    # Per-variable RMSE across all test trajectories (each traj is one sample)
    rmse_per_traj_per_var = np.zeros((n_test, n_feat))
    for i in range(n_test):
        for j in range(n_feat):
            pred_j = all_pred_raw[i, seq_len:, j]
            truth_j = test_raw[i, seq_len:, j]
            rmse_per_traj_per_var[i, j] = np.sqrt(np.mean((pred_j - truth_j) ** 2))

    return {
        "all_pred_raw": all_pred_raw,
        "abs_errors": abs_errors,             # (n_test, n_pred_steps, n_feat)
        "rmse_per_traj": rmse_per_traj,       # (n_test,)
        "rmse_per_traj_per_var": rmse_per_traj_per_var,  # (n_test, n_feat)
        "n_test": n_test,
        "n_pred_steps": n_pred_steps,
    }


def plot_error_histogram(
    rmse_per_traj: np.ndarray,
    save_path: Path,
    title: str = "",
) -> None:
    """Histogram of per-trajectory RMSE across all test trajectories."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(rmse_per_traj, bins=25, edgecolor="black", alpha=0.7, color="#4C72B0")
    mean_rmse = np.mean(rmse_per_traj)
    median_rmse = np.median(rmse_per_traj)

    ax.axvline(mean_rmse, color="red", linestyle="--", linewidth=2,
               label=f"Mean: {mean_rmse:.4f}")
    ax.axvline(median_rmse, color="orange", linestyle=":", linewidth=2,
               label=f"Median: {median_rmse:.4f}")

    ax.set_xlabel("RMSE (original scale)", fontsize=12)
    ax.set_ylabel("Number of Test Trajectories", fontsize=12)
    ax.set_title(title or f"RMSE Distribution — {len(rmse_per_traj)} Test Trajectories",
                 fontsize=13)
    ax.legend(fontsize=11)

    # Stats box
    stats_text = (
        f"n = {len(rmse_per_traj)}\n"
        f"Mean = {mean_rmse:.4f}\n"
        f"Std  = {np.std(rmse_per_traj):.4f}\n"
        f"Min  = {np.min(rmse_per_traj):.4f}\n"
        f"Max  = {np.max(rmse_per_traj):.4f}"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_rmse_boxplot_per_variable(
    rmse_per_traj_per_var: np.ndarray,
    save_path: Path,
    title: str = "",
) -> None:
    """Box plot of per-variable RMSE distribution across test trajectories."""
    n_feat = rmse_per_traj_per_var.shape[1]
    fig, ax = plt.subplots(figsize=(14, 7))

    bp = ax.boxplot(
        [rmse_per_traj_per_var[:, j] for j in range(n_feat)],
        labels=FEATURE_NAMES,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=5),
    )

    colors = plt.cm.Set3(np.linspace(0, 1, n_feat))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("RMSE (original scale)", fontsize=12)
    ax.set_title(title or "Per-Variable RMSE Distribution — All Test Trajectories",
                 fontsize=13)
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Add mean values as text above each box
    means = np.mean(rmse_per_traj_per_var, axis=0)
    for j in range(n_feat):
        ax.text(j + 1, means[j], f"{means[j]:.2e}", ha="center", va="bottom",
                fontsize=7, color="red", fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_error_over_time(
    abs_errors: np.ndarray,
    time_axis: np.ndarray,
    seq_len: int,
    save_path: Path,
    title: str = "",
) -> None:
    """Mean +/- std absolute error at each timestep for each variable.
    Shows error accumulation over autoregressive horizon.
    """
    pred_time = time_axis[seq_len:]
    mean_err = np.mean(abs_errors, axis=0)   # (n_pred_steps, n_feat)
    std_err = np.std(abs_errors, axis=0)

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    axes = axes.flatten()

    for j in range(abs_errors.shape[2]):
        ax = axes[j]
        ax.plot(pred_time, mean_err[:, j], "b-", linewidth=1.5, label="Mean |error|")
        ax.fill_between(
            pred_time,
            np.maximum(mean_err[:, j] - std_err[:, j], 0),
            mean_err[:, j] + std_err[:, j],
            alpha=0.3, color="blue", label="+/- 1 std",
        )
        ax.set_title(FEATURE_NAMES[j], fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (days)", fontsize=9)
        ax.set_ylabel("Absolute Error", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        title or "Error Accumulation Over Time (mean +/- std across all test trajectories)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_predicted_vs_actual(
    all_pred_raw: np.ndarray,
    test_raw: np.ndarray,
    seq_len: int,
    save_path: Path,
    title: str = "",
) -> None:
    """Scatter plot: predicted vs actual for all variables across all test trajectories."""
    pred_all = all_pred_raw[:, seq_len:, :].reshape(-1, all_pred_raw.shape[2])
    truth_all = test_raw[:, seq_len:, :].reshape(-1, test_raw.shape[2])

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    axes = axes.flatten()

    for j in range(pred_all.shape[1]):
        ax = axes[j]
        ax.scatter(truth_all[:, j], pred_all[:, j], s=2, alpha=0.15, color="#4C72B0")

        # Perfect prediction line
        vmin = min(truth_all[:, j].min(), pred_all[:, j].min())
        vmax = max(truth_all[:, j].max(), pred_all[:, j].max())
        ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1.5, label="Perfect")

        # R-squared
        ss_res = np.sum((truth_all[:, j] - pred_all[:, j]) ** 2)
        ss_tot = np.sum((truth_all[:, j] - np.mean(truth_all[:, j])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        ax.set_title(f"{FEATURE_NAMES[j]}  (R² = {r2:.4f})", fontsize=10)
        ax.set_xlabel("Actual", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="box")
        if j == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        title or "Predicted vs Actual — All Test Trajectories",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_trajectory_percentiles(
    all_pred_raw: np.ndarray,
    test_raw: np.ndarray,
    rmse_per_traj: np.ndarray,
    time_axis: np.ndarray,
    seq_len: int,
    save_path: Path,
    title: str = "",
) -> None:
    """Show representative trajectories at P10, P25, P50, P75, P90 error levels."""
    percentiles = [10, 25, 50, 75, 90]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    labels = [f"P{p} (RMSE={np.percentile(rmse_per_traj, p):.4f})" for p in percentiles]

    # Find trajectory index closest to each percentile
    indices = []
    for p in percentiles:
        target = np.percentile(rmse_per_traj, p)
        idx = int(np.argmin(np.abs(rmse_per_traj - target)))
        indices.append(idx)

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    axes = axes.flatten()

    for j in range(test_raw.shape[2]):
        ax = axes[j]
        # Ground truth of P50 trajectory as reference
        ax.plot(time_axis, test_raw[indices[2], :, j], "k-",
                linewidth=2, alpha=0.5, label="Ground truth (P50)")

        for k, (idx, color, label) in enumerate(zip(indices, colors, labels)):
            ax.plot(time_axis, all_pred_raw[idx, :, j], "--",
                    color=color, linewidth=1.2, alpha=0.8,
                    label=label if j == 0 else None)

        ax.axvline(x=time_axis[seq_len], color="gray", linestyle=":", alpha=0.5)
        ax.set_title(FEATURE_NAMES[j], fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (days)", fontsize=8)
        ax.tick_params(labelsize=7)

    # Legend outside
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        title or "Trajectories at Different Error Percentiles (P10–P90)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_all_tests_overlay(
    all_pred_raw: np.ndarray,
    test_raw: np.ndarray,
    time_axis: np.ndarray,
    seq_len: int,
    rmse_per_traj: np.ndarray,
    save_path: Path,
    title: str = "",
) -> None:
    """Single PNG showing ALL test trajectories: ground truth vs prediction.

    For each variable (4x3 grid):
    - Light gray lines: all 100 ground truth trajectories
    - Light red lines: all 100 predicted trajectories
    - Bold blue line: mean ground truth
    - Bold red dashed line: mean prediction
    - Shaded band: mean prediction +/- 1 std
    """
    n_test, n_steps, n_feat = test_raw.shape

    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    axes = axes.flatten()

    mean_truth = np.mean(test_raw, axis=0)
    mean_pred = np.mean(all_pred_raw, axis=0)
    std_pred = np.std(all_pred_raw, axis=0)

    for j in range(n_feat):
        ax = axes[j]

        # All individual trajectories (thin, transparent)
        for i in range(n_test):
            ax.plot(time_axis, test_raw[i, :, j],
                    color="#4C72B0", alpha=0.08, linewidth=0.5)
            ax.plot(time_axis, all_pred_raw[i, :, j],
                    color="#C44E52", alpha=0.08, linewidth=0.5)

        # Mean lines (bold)
        ax.plot(time_axis, mean_truth[:, j], color="#1f3d73",
                linewidth=2.5, label="Ground Truth (mean)")
        ax.plot(time_axis, mean_pred[:, j], color="#C44E52",
                linewidth=2.5, linestyle="--", label="Prediction (mean)")

        # Prediction std band
        ax.fill_between(
            time_axis,
            np.maximum(mean_pred[:, j] - std_pred[:, j], 0),
            mean_pred[:, j] + std_pred[:, j],
            alpha=0.15, color="#C44E52",
        )

        # Seed boundary
        ax.axvline(x=time_axis[seq_len], color="gray", linestyle=":", alpha=0.6)

        # Per-variable mean RMSE annotation
        rmse_j = np.sqrt(np.mean(
            (all_pred_raw[:, seq_len:, j] - test_raw[:, seq_len:, j]) ** 2
        ))
        ax.text(0.97, 0.95, f"RMSE: {rmse_j:.4e}",
                transform=ax.transAxes, fontsize=9,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_title(FEATURE_NAMES[j], fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (days)", fontsize=9)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.legend(fontsize=9, loc="lower right")

    # Global stats annotation
    mean_rmse = np.mean(rmse_per_traj)
    std_rmse = np.std(rmse_per_traj)
    fig.suptitle(
        (title or "All Test Trajectories — Ground Truth vs Prediction")
        + f"\n{n_test} trajectories | Overall RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}",
        fontsize=15, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_individual_deviations(
    all_pred_raw: np.ndarray,
    test_raw: np.ndarray,
    time_axis: np.ndarray,
    seq_len: int,
    rmse_per_traj: np.ndarray,
    save_path: Path,
    title: str = "",
) -> None:
    """Each trajectory's deviation from its OWN ground truth.

    For each variable (4x3 grid):
    - Thin lines: individual trajectory errors (pred - truth) for all 100 tests
    - Color: green=low error trajectory, red=high error trajectory
    - Bold black line: mean absolute error across all trajectories
    - Horizontal dashed line at 0 (perfect prediction)
    """
    n_test, n_steps, n_feat = test_raw.shape
    pred_time = time_axis[seq_len:]

    # Absolute errors for predicted portion: (n_test, n_pred_steps, n_feat)
    abs_errors = np.abs(all_pred_raw[:, seq_len:, :] - test_raw[:, seq_len:, :])

    # Color each trajectory by its overall RMSE (green=good, red=bad)
    norm_rmse = (rmse_per_traj - rmse_per_traj.min()) / (rmse_per_traj.max() - rmse_per_traj.min() + 1e-12)
    # Sort so worst (red) drawn first, best (green) on top
    sort_idx = np.argsort(-norm_rmse)

    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    axes = axes.flatten()

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("grd", ["#2ca02c", "#f0e442", "#d62728"])

    for j in range(n_feat):
        ax = axes[j]

        # Zero line
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.4)

        # Individual trajectory errors (absolute)
        for i in sort_idx:
            color = cmap(norm_rmse[i])
            ax.plot(pred_time, abs_errors[i, :, j],
                    color=color, alpha=0.25, linewidth=0.6)

        # Mean absolute error
        mae_over_time = np.mean(abs_errors[:, :, j], axis=0)
        ax.plot(pred_time, mae_over_time, color="black", linewidth=2.5,
                label="Mean |error|")

        # Per-variable RMSE
        rmse_j = np.sqrt(np.mean(abs_errors[:, :, j] ** 2))
        ax.text(0.97, 0.95, f"RMSE: {rmse_j:.4e}",
                transform=ax.transAxes, fontsize=9,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_title(FEATURE_NAMES[j], fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (days)", fontsize=9)
        ax.set_ylabel("RMSE", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)
        if j == 0:
            ax.legend(fontsize=9)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        vmin=rmse_per_traj.min(), vmax=rmse_per_traj.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.6, pad=0.02)
    cbar.set_label("Trajectory RMSE", fontsize=11)

    mean_rmse = np.mean(rmse_per_traj)
    std_rmse = np.std(rmse_per_traj)
    fig.suptitle(
        (title or "Per-Trajectory Deviation from Ground Truth")
        + f"\n{n_test} test trajectories | RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}"
        + f" | Green=best, Red=worst",
        fontsize=14, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_summary_statistics_table(
    rmse_per_traj_per_var: np.ndarray,
    rmse_per_traj: np.ndarray,
    save_path: Path,
    title: str = "",
) -> None:
    """Render summary statistics as a table image."""
    n_feat = rmse_per_traj_per_var.shape[1]

    headers = ["Variable", "Mean RMSE", "Std", "Median", "Q25", "Q75", "Min", "Max"]
    rows = []
    for j in range(n_feat):
        vals = rmse_per_traj_per_var[:, j]
        rows.append([
            FEATURE_NAMES[j],
            f"{np.mean(vals):.4e}",
            f"{np.std(vals):.4e}",
            f"{np.median(vals):.4e}",
            f"{np.percentile(vals, 25):.4e}",
            f"{np.percentile(vals, 75):.4e}",
            f"{np.min(vals):.4e}",
            f"{np.max(vals):.4e}",
        ])
    # Add total row
    rows.append([
        "TOTAL",
        f"{np.mean(rmse_per_traj):.4e}",
        f"{np.std(rmse_per_traj):.4e}",
        f"{np.median(rmse_per_traj):.4e}",
        f"{np.percentile(rmse_per_traj, 25):.4e}",
        f"{np.percentile(rmse_per_traj, 75):.4e}",
        f"{np.min(rmse_per_traj):.4e}",
        f"{np.max(rmse_per_traj):.4e}",
    ])

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style total row
    for j in range(len(headers)):
        table[len(rows), j].set_facecolor("#f0f0f0")
        table[len(rows), j].set_text_props(fontweight="bold")

    ax.set_title(
        title or "Summary Statistics — RMSE Across All Test Trajectories",
        fontsize=14, fontweight="bold", pad=20,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def run_comprehensive_evaluation(
    experiment_dir: str,
    output_dir: str | None = None,
) -> dict:
    """Load a saved experiment and generate comprehensive evaluation PNGs.

    Args:
        experiment_dir: path to experiment (e.g. "experiments/seq10_h128")
        output_dir: where to save PNGs (default: experiment_dir/comprehensive/)
    """
    exp_dir = Path(experiment_dir)
    config = ExperimentConfig.load(exp_dir / "config.json")

    if output_dir is None:
        out_dir = exp_dir / "comprehensive"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = setup_device()

    # Load data
    raw_data = load_all_trajectories(Path(config.data_dir) / "output")
    time_axis = load_time_axis(Path(config.data_dir) / "output")

    _, test_raw, _, _ = split_trajectories(
        raw_data, test_ratio=config.test_ratio, seed=config.seed
    )

    preprocessor = DataPreprocessor(log_cols=config.log_cols)
    train_raw = load_all_trajectories(Path(config.data_dir) / "output")
    train_split, _, _, _ = split_trajectories(
        train_raw, test_ratio=config.test_ratio, seed=config.seed
    )
    preprocessor.fit(train_split)
    test_norm = preprocessor.transform(test_raw)

    # Load model
    model = PhreeqcLSTM(
        n_features=config.n_features,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    model.load_state_dict(torch.load(
        exp_dir / "best_model.pt", weights_only=True, map_location=device
    ))
    model.to(device)

    print(f"\nEvaluating {config.experiment_name} on {test_raw.shape[0]} test trajectories...")

    # Run evaluation
    result = evaluate_all_trajectories(
        model, test_norm, test_raw, preprocessor, config.seq_len, device
    )

    exp_label = config.experiment_name

    # Generate all plots
    print("\nGenerating comprehensive plots...")

    plot_error_histogram(
        result["rmse_per_traj"],
        save_path=out_dir / "error_histogram.png",
        title=f"[{exp_label}] RMSE Distribution — {result['n_test']} Test Trajectories",
    )

    plot_rmse_boxplot_per_variable(
        result["rmse_per_traj_per_var"],
        save_path=out_dir / "rmse_boxplot_per_variable.png",
        title=f"[{exp_label}] Per-Variable RMSE — All Test Trajectories",
    )

    plot_error_over_time(
        result["abs_errors"],
        time_axis,
        config.seq_len,
        save_path=out_dir / "error_over_time.png",
        title=f"[{exp_label}] Error Accumulation Over Time",
    )

    plot_predicted_vs_actual(
        result["all_pred_raw"],
        test_raw,
        config.seq_len,
        save_path=out_dir / "predicted_vs_actual.png",
        title=f"[{exp_label}] Predicted vs Actual — All Test Trajectories",
    )

    plot_trajectory_percentiles(
        result["all_pred_raw"],
        test_raw,
        result["rmse_per_traj"],
        time_axis,
        config.seq_len,
        save_path=out_dir / "trajectory_percentiles.png",
        title=f"[{exp_label}] Trajectories at P10, P25, P50, P75, P90",
    )

    plot_individual_deviations(
        result["all_pred_raw"],
        test_raw,
        time_axis,
        config.seq_len,
        result["rmse_per_traj"],
        save_path=out_dir / "individual_deviations.png",
        title=f"[{exp_label}] Per-Trajectory Deviation from Ground Truth",
    )

    plot_summary_statistics_table(
        result["rmse_per_traj_per_var"],
        result["rmse_per_traj"],
        save_path=out_dir / "summary_statistics.png",
        title=f"[{exp_label}] Summary Statistics — RMSE Across All Test Trajectories",
    )

    # Save stats as JSON too
    stats = {
        "experiment": exp_label,
        "n_test_trajectories": result["n_test"],
        "n_predicted_steps": result["n_pred_steps"],
        "overall_rmse": {
            "mean": float(np.mean(result["rmse_per_traj"])),
            "std": float(np.std(result["rmse_per_traj"])),
            "median": float(np.median(result["rmse_per_traj"])),
            "min": float(np.min(result["rmse_per_traj"])),
            "max": float(np.max(result["rmse_per_traj"])),
            "q25": float(np.percentile(result["rmse_per_traj"], 25)),
            "q75": float(np.percentile(result["rmse_per_traj"], 75)),
        },
        "per_variable_rmse_mean": {
            name: float(np.mean(result["rmse_per_traj_per_var"][:, j]))
            for j, name in enumerate(FEATURE_NAMES)
        },
        "per_variable_rmse_std": {
            name: float(np.std(result["rmse_per_traj_per_var"][:, j]))
            for j, name in enumerate(FEATURE_NAMES)
        },
    }

    with open(out_dir / "comprehensive_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {out_dir / 'comprehensive_stats.json'}")

    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE EVALUATION COMPLETE: {exp_label}")
    print(f"{'='*60}")
    print(f"Test trajectories: {result['n_test']}")
    print(f"Overall RMSE: {stats['overall_rmse']['mean']:.4f} +/- {stats['overall_rmse']['std']:.4f}")
    print(f"Median RMSE:  {stats['overall_rmse']['median']:.4f}")
    print(f"Range:        [{stats['overall_rmse']['min']:.4f}, {stats['overall_rmse']['max']:.4f}]")
    print(f"\nPer-variable mean RMSE:")
    for name, val in stats["per_variable_rmse_mean"].items():
        print(f"  {name:12s}: {val:.6f}")
    print(f"\nAll PNGs saved to: {out_dir}/")

    return stats
