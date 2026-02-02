"""
============================================================================
LSTM Trajectory Comparison Visualization
============================================================================

Compares three LSTM approaches:
1. Next-step (h64, L1) - Simple single-layer model
2. Next-step (h128, L2) - Deeper model with more capacity
3. Seq-window (W50, h128, L2) - Windowed input approach

Generates publication-quality figures for analysis.

============================================================================
Author: Chemical Thesis Project
Date: 2026-W04
============================================================================
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# State variable names (14 features)
STATE_NAMES = [
    "nH2_g", "nCO2_g", "nCH4_g", "nH2S_g",
    "H2_aq", "CO2_aq", "SO4", "FeS",
    "X", "Acetate", "HCO3", "S_tot", "Lag", "Fe_pool"
]

# Human-readable labels with units
STATE_LABELS = {
    "nH2_g": r"$n_{H_2,g}$ (mmol)",
    "nCO2_g": r"$n_{CO_2,g}$ (mmol)",
    "nCH4_g": r"$n_{CH_4,g}$ (mmol)",
    "nH2S_g": r"$n_{H_2S,g}$ (mmol)",
    "H2_aq": r"$[H_2]_{aq}$ (mM)",
    "CO2_aq": r"$[CO_2]_{aq}$ (mM)",
    "SO4": r"$[SO_4^{2-}]$ (mM)",
    "FeS": r"$[FeS]$ (mM)",
    "X": r"Biomass $X$ (mM)",
    "Acetate": r"$[Acetate]$ (mM)",
    "HCO3": r"$[HCO_3^-]$ (mM)",
    "S_tot": r"$[S_{tot}]$ (mM)",
    "Lag": r"Lag Factor",
    "Fe_pool": r"$[Fe]_{pool}$ (mM)"
}

# Colors for different models
COLORS = {
    "ground_truth": "#2E2E2E",      # Dark gray
    "nextstep_h64_L1": "#E74C3C",   # Red
    "nextstep_h128_L2": "#3498DB",  # Blue
    "seqwin_W50": "#27AE60",        # Green
}

# Line styles
LINESTYLES = {
    "ground_truth": "-",
    "nextstep_h64_L1": "--",
    "nextstep_h128_L2": "-.",
    "seqwin_W50": ":",
}


def load_trajectories(output_dir: Path) -> Dict:
    """Load all trajectory data from output directories."""
    trajectories = {}

    # Next-step v1 (h64, L1)
    path_v1 = output_dir / "lstm_nextstep_h64_L1_20260125" / "trajectory.npz"
    if path_v1.exists():
        data = np.load(path_v1)
        trajectories["nextstep_h64_L1"] = {
            "trajectory": data["trajectory_orig"],
            "ground_truth": data["ground_truth_orig"],
            "label": "Next-step (h64, L1)",
        }

    # Next-step v2 (h128, L2)
    path_v2 = output_dir / "lstm_nextstep_h128_L2_20260125" / "trajectory.npz"
    if path_v2.exists():
        data = np.load(path_v2)
        trajectories["nextstep_h128_L2"] = {
            "trajectory": data["trajectory_orig"],
            "ground_truth": data["ground_truth_orig"],
            "label": "Next-step (h128, L2)",
        }

    # Seq-window (W50, h128, L2)
    path_sw = output_dir / "lstm_seqwin_W50_h128_L2_20260126" / "trajectory.npz"
    if path_sw.exists():
        data = np.load(path_sw)
        trajectories["seqwin_W50"] = {
            "trajectory": data["trajectory_orig"],
            "ground_truth": data["ground_truth_orig"],
            "window_size": int(data["window_size"]),
            "label": "Seq-window (W50, h128, L2)",
        }

    return trajectories


def compute_metrics(trajectories: Dict) -> Dict:
    """Compute RMSE and other metrics for each model."""
    metrics = {}
    ground_truth = trajectories["nextstep_h64_L1"]["ground_truth"]

    for model_name, data in trajectories.items():
        traj = data["trajectory"]

        # Per-variable RMSE
        rmse_per_var = np.sqrt(np.mean((traj - ground_truth) ** 2, axis=0))

        # Total RMSE
        rmse_total = np.sqrt(np.mean((traj - ground_truth) ** 2))

        # Cumulative error over time
        cumulative_error = np.sqrt(np.cumsum((traj - ground_truth) ** 2, axis=0).mean(axis=1))

        # Point-wise error
        pointwise_error = np.sqrt(((traj - ground_truth) ** 2).mean(axis=1))

        metrics[model_name] = {
            "rmse_per_var": rmse_per_var,
            "rmse_total": rmse_total,
            "cumulative_error": cumulative_error,
            "pointwise_error": pointwise_error,
            "label": data["label"],
        }

    return metrics


def create_time_axis(n_points: int, total_days: float = 19.0) -> np.ndarray:
    """Create time axis in days."""
    return np.linspace(0, total_days, n_points)


# ============================================================================
# FIGURE 1: All Trajectories Overview (4x4 grid, 14 variables + 2 summary)
# ============================================================================
def plot_all_trajectories(
    trajectories: Dict,
    output_path: Path,
    figsize: Tuple[int, int] = (20, 16)
):
    """
    Create a 4x4 grid showing all 14 state variables plus summary panels.
    """
    fig, axes = plt.subplots(4, 4, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    ground_truth = trajectories["nextstep_h64_L1"]["ground_truth"]
    time = create_time_axis(len(ground_truth))

    # Plot each state variable
    for idx, var_name in enumerate(STATE_NAMES):
        ax = axes[idx]

        # Ground truth
        ax.plot(time, ground_truth[:, idx],
                color=COLORS["ground_truth"],
                linewidth=2,
                label="Ground Truth (ODE)")

        # Each model
        for model_name, data in trajectories.items():
            ax.plot(time, data["trajectory"][:, idx],
                    color=COLORS[model_name],
                    linestyle=LINESTYLES[model_name],
                    linewidth=1.5,
                    alpha=0.8,
                    label=data["label"])

        ax.set_xlabel("Time (days)", fontsize=9)
        ax.set_ylabel(STATE_LABELS[var_name], fontsize=9)
        ax.set_title(var_name, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # Mark window boundary for seq-window
        if "seqwin_W50" in trajectories:
            window_size = trajectories["seqwin_W50"]["window_size"]
            window_time = time[window_size]
            ax.axvline(x=window_time, color="gray", linestyle=":", alpha=0.5)

    # Summary panel 1: Legend
    ax_legend = axes[14]
    ax_legend.axis("off")
    handles = [
        plt.Line2D([0], [0], color=COLORS["ground_truth"], linewidth=2, label="Ground Truth (ODE)"),
        plt.Line2D([0], [0], color=COLORS["nextstep_h64_L1"], linestyle="--", linewidth=2, label="Next-step (h64, L1)"),
        plt.Line2D([0], [0], color=COLORS["nextstep_h128_L2"], linestyle="-.", linewidth=2, label="Next-step (h128, L2)"),
        plt.Line2D([0], [0], color=COLORS["seqwin_W50"], linestyle=":", linewidth=2, label="Seq-window (W50)"),
    ]
    ax_legend.legend(handles=handles, loc="center", fontsize=12, frameon=True)
    ax_legend.set_title("Model Legend", fontsize=12, fontweight="bold")

    # Summary panel 2: Info text
    ax_info = axes[15]
    ax_info.axis("off")
    info_text = (
        "LSTM Trajectory Comparison\n"
        "─" * 30 + "\n"
        "Data: Basalt @ 25°C\n"
        "Points: 500 (19 days)\n"
        "dt: ~55 minutes\n"
        "─" * 30 + "\n"
        "Gray dashed line:\n"
        "Window boundary (t=1.9 days)"
    )
    ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes,
                 fontsize=11, verticalalignment="center", horizontalalignment="center",
                 family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("LSTM Trajectory Reconstruction: All State Variables",
                 fontsize=16, fontweight="bold", y=1.02)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 2: Key Variables Comparison (Selected Important Variables)
# ============================================================================
def plot_key_variables(
    trajectories: Dict,
    output_path: Path,
    key_vars: List[str] = None,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    Focus plot on key variables that show interesting dynamics.
    """
    if key_vars is None:
        # Select variables with diverse dynamics
        key_vars = ["nH2_g", "nCH4_g", "SO4", "X", "Acetate", "Lag"]

    n_vars = len(key_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    ground_truth = trajectories["nextstep_h64_L1"]["ground_truth"]
    time = create_time_axis(len(ground_truth))

    for idx, var_name in enumerate(key_vars):
        ax = axes[idx]
        var_idx = STATE_NAMES.index(var_name)

        # Ground truth (thicker)
        ax.plot(time, ground_truth[:, var_idx],
                color=COLORS["ground_truth"],
                linewidth=2.5,
                label="Ground Truth (ODE)")

        # Each model
        for model_name, data in trajectories.items():
            ax.plot(time, data["trajectory"][:, var_idx],
                    color=COLORS[model_name],
                    linestyle=LINESTYLES[model_name],
                    linewidth=2,
                    alpha=0.85,
                    label=data["label"])

        ax.set_xlabel("Time (days)", fontsize=11)
        ax.set_ylabel(STATE_LABELS[var_name], fontsize=11)
        ax.set_title(f"{var_name}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

        # Mark window boundary
        if "seqwin_W50" in trajectories:
            window_size = trajectories["seqwin_W50"]["window_size"]
            window_time = time[window_size]
            ax.axvline(x=window_time, color="gray", linestyle=":",
                      alpha=0.7, label="Window boundary")

    # Hide unused axes
    for idx in range(len(key_vars), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Key State Variables: LSTM vs Ground Truth",
                 fontsize=16, fontweight="bold")

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 3: RMSE Comparison Bar Chart
# ============================================================================
def plot_rmse_comparison(
    metrics: Dict,
    output_path: Path,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Bar chart comparing RMSE across models for each variable.
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    n_vars = len(STATE_NAMES)
    n_models = len(metrics)
    bar_width = 0.25
    x = np.arange(n_vars)

    for i, (model_name, data) in enumerate(metrics.items()):
        offset = (i - n_models/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, data["rmse_per_var"], bar_width,
                      color=COLORS[model_name], alpha=0.8,
                      label=data["label"], edgecolor="black", linewidth=0.5)

    ax.set_xlabel("State Variable", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("Per-Variable RMSE Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")  # Log scale to see small differences

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 4: Error Accumulation Over Time
# ============================================================================
def plot_error_accumulation(
    metrics: Dict,
    trajectories: Dict,
    output_path: Path,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Show how prediction error accumulates over time for each model.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    n_points = len(metrics["nextstep_h64_L1"]["pointwise_error"])
    time = create_time_axis(n_points)

    # Panel 1: Point-wise error
    ax1 = axes[0]
    for model_name, data in metrics.items():
        ax1.plot(time, data["pointwise_error"],
                color=COLORS[model_name],
                linewidth=2,
                label=data["label"])

    ax1.set_xlabel("Time (days)", fontsize=11)
    ax1.set_ylabel("Point-wise RMSE", fontsize=11)
    ax1.set_title("Prediction Error Over Time", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Mark window boundary
    if "seqwin_W50" in trajectories:
        window_time = time[trajectories["seqwin_W50"]["window_size"]]
        ax1.axvline(x=window_time, color="gray", linestyle=":",
                   alpha=0.7, linewidth=2)
        ax1.text(window_time + 0.3, ax1.get_ylim()[1] * 0.5,
                "Window\nboundary", fontsize=9, color="gray")

    # Panel 2: Cumulative error
    ax2 = axes[1]
    for model_name, data in metrics.items():
        ax2.plot(time, data["cumulative_error"],
                color=COLORS[model_name],
                linewidth=2,
                label=data["label"])

    ax2.set_xlabel("Time (days)", fontsize=11)
    ax2.set_ylabel("Cumulative RMSE", fontsize=11)
    ax2.set_title("Cumulative Error Accumulation", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    if "seqwin_W50" in trajectories:
        ax2.axvline(x=window_time, color="gray", linestyle=":",
                   alpha=0.7, linewidth=2)

    fig.suptitle("Error Analysis: How Predictions Diverge Over Time",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 5: Summary Statistics Table (as figure)
# ============================================================================
def plot_summary_table(
    metrics: Dict,
    output_path: Path,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Create a summary table as a figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # Prepare data
    headers = ["Model", "Total RMSE", "Best Variable", "Worst Variable"]
    rows = []

    for model_name, data in metrics.items():
        rmse_total = data["rmse_total"]
        rmse_per_var = data["rmse_per_var"]

        best_idx = np.argmin(rmse_per_var)
        worst_idx = np.argmax(rmse_per_var)

        rows.append([
            data["label"],
            f"{rmse_total:.4f}",
            f"{STATE_NAMES[best_idx]} ({rmse_per_var[best_idx]:.2e})",
            f"{STATE_NAMES[worst_idx]} ({rmse_per_var[worst_idx]:.4f})"
        ])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.3, 0.15, 0.25, 0.3]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for i, key in enumerate(headers):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = "#D6EAF8" if i % 2 == 0 else "#EBF5FB"
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)

    ax.set_title("Model Performance Summary", fontsize=14, fontweight="bold", pad=20)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================
def main(output_dir: str = "outputs", figure_dir: str = "figures/comparison"):
    """Generate all comparison figures."""
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LSTM TRAJECTORY COMPARISON VISUALIZATION")
    print("=" * 60)

    # Load data
    print("\nLoading trajectories...")
    trajectories = load_trajectories(output_dir)
    print(f"Loaded {len(trajectories)} models: {list(trajectories.keys())}")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(trajectories)

    # Print summary
    print("\n" + "=" * 60)
    print("RMSE SUMMARY")
    print("=" * 60)
    for model_name, data in metrics.items():
        print(f"\n{data['label']}:")
        print(f"  Total RMSE: {data['rmse_total']:.6f}")

    # Generate figures
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    # Figure 1: All trajectories
    plot_all_trajectories(
        trajectories,
        figure_dir / "fig1_all_trajectories.png"
    )

    # Figure 2: Key variables
    plot_key_variables(
        trajectories,
        figure_dir / "fig2_key_variables.png"
    )

    # Figure 3: RMSE bar chart
    plot_rmse_comparison(
        metrics,
        figure_dir / "fig3_rmse_comparison.png"
    )

    # Figure 4: Error accumulation
    plot_error_accumulation(
        metrics,
        trajectories,
        figure_dir / "fig4_error_accumulation.png"
    )

    # Figure 5: Summary table
    plot_summary_table(
        metrics,
        figure_dir / "fig5_summary_table.png"
    )

    print("\n" + "=" * 60)
    print(f"All figures saved to: {figure_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate LSTM comparison figures")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory containing trajectory outputs")
    parser.add_argument("--figure_dir", type=str, default="figures/comparison",
                        help="Directory to save figures")

    args = parser.parse_args()
    main(args.output_dir, args.figure_dir)
