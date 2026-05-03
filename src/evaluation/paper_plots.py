from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.paper_runtime import configure_paper_runtime

configure_paper_runtime()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.constants import FEATURE_NAMES
from src.evaluation.paper_contracts import (
    DEFAULT_BREAKTHROUGH_FEATURES,
    DEFAULT_PROFILE_FEATURES,
)
from src.evaluation.paper_metrics import (
    normalize_result_metrics,
    percentile_indices,
    select_reference_trajectory_index,
    select_novelty_representative_indices,
    selected_feature_indices,
    summarize_generalization_bins,
)


def save_figure(fig: plt.Figure, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def _draw_box(axis: plt.Axes, x: float, y: float, text: str) -> None:
    axis.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#f5f7fb",
            "edgecolor": "#4C72B0",
            "linewidth": 1.4,
        },
    )


def _draw_arrow(axis: plt.Axes, x0: float, y0: float, x1: float, y1: float) -> None:
    axis.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={"arrowstyle": "->", "linewidth": 1.5, "color": "#333333"},
    )


def _annotate_best_bar(axis: plt.Axes, index: int, value: float) -> None:
    axis.text(
        index,
        value,
        "best",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def plot_workflow_schematic(save_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(14, 5))
    axis.axis("off")

    boxes = [
        (0.08, 0.62, "PHREEQC v23\n1000 trajectories"),
        (0.27, 0.62, "Train/test split\nsequential runs"),
        (0.46, 0.62, "Normalize\ntrain statistics"),
        (0.65, 0.62, "LSTM rollout\nautoregressive"),
        (0.84, 0.62, "Figure bundle\nPNG + MD + manifest"),
    ]
    for x, y, text in boxes:
        _draw_box(axis, x, y, text)

    for (x0, y0, _), (x1, y1, _) in zip(boxes[:-1], boxes[1:]):
        _draw_arrow(axis, x0 + 0.07, y0, x1 - 0.07, y1)

    _draw_box(axis, 0.46, 0.24, "Ground truth: PHREEQC\nPrediction: trained LSTM")
    _draw_arrow(axis, 0.65, 0.52, 0.52, 0.32)
    _draw_arrow(axis, 0.46, 0.34, 0.78, 0.52)

    axis.set_title(
        "Paper-Aligned Figure 1 - Repo Surrogate Evaluation Workflow",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    save_figure(fig, save_path)


def plot_lstm_architecture_schematic(save_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(13, 5))
    axis.axis("off")

    boxes = [
        (0.12, 0.62, "Input window\n(seq_len x 12)"),
        (0.34, 0.62, "LSTM\nhidden_size=128"),
        (0.56, 0.62, "Linear head\n12 outputs"),
        (0.78, 0.62, "Next state\nX(t+1)"),
    ]
    for x, y, text in boxes:
        _draw_box(axis, x, y, text)

    for (x0, y0, _), (x1, y1, _) in zip(boxes[:-1], boxes[1:]):
        _draw_arrow(axis, x0 + 0.08, y0, x1 - 0.08, y1)

    _draw_box(axis, 0.45, 0.22, "Autoregressive loop:\nprediction feeds next window")
    _draw_arrow(axis, 0.78, 0.52, 0.54, 0.30)
    _draw_arrow(axis, 0.36, 0.30, 0.14, 0.52)

    axis.text(
        0.5,
        0.03,
        "Adapted analog of HRTNet architecture: this repo does not encode PDE residuals or hidden pyrite mass.",
        ha="center",
        va="center",
        fontsize=10,
    )
    axis.set_title(
        "Paper-Aligned Figure 2 - Autoregressive LSTM Architecture",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    save_figure(fig, save_path)


def plot_sequence_sensitivity_summary(
    experiment_rows: Sequence[Mapping[str, Any]],
    save_path: Path,
) -> None:
    rows = sorted(experiment_rows, key=lambda item: int(item["seq_len"]))
    seq_lens = [int(row["seq_len"]) for row in rows]
    original_rmse = [float(row["overall_rmse_mean"]) for row in rows]
    normalized_rmse = [float(row["nrmse_total"]) for row in rows]

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        seq_lens,
        original_rmse,
        marker="o",
        linewidth=2,
        label="Original-scale rollout RMSE",
    )
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Original-scale RMSE")
    axis.grid(True, alpha=0.3)

    twin = axis.twinx()
    twin.plot(
        seq_lens,
        normalized_rmse,
        marker="s",
        linewidth=2,
        color="#C44E52",
        label="Normalized RMSE",
    )
    twin.set_ylabel("Normalized RMSE")

    handles, labels = axis.get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axis.legend(handles + twin_handles, labels + twin_labels, loc="upper right")
    axis.set_title(
        "Paper-Aligned Figure 10 - Sequence-Length Sensitivity",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, save_path)


def plot_metric_sensitivity_summary(
    experiment_rows: Sequence[Mapping[str, Any]],
    save_path: Path,
) -> None:
    rows = sorted(experiment_rows, key=lambda item: int(item["seq_len"]))
    labels = [f"seq{int(row['seq_len'])}" for row in rows]
    original_rmse = np.array([float(row["overall_rmse_mean"]) for row in rows])
    normalized_rmse = np.array([float(row["nrmse_total"]) for row in rows])
    best_original = int(np.argmin(original_rmse))
    best_normalized = int(np.argmin(normalized_rmse))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(labels, original_rmse, color="#4C72B0", alpha=0.8)
    axes[0].set_title("Original-scale rollout RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, axis="y", alpha=0.25)
    _annotate_best_bar(axes[0], best_original, original_rmse[best_original])

    axes[1].bar(labels, normalized_rmse, color="#C44E52", alpha=0.8)
    axes[1].set_title("Normalized RMSE")
    axes[1].set_ylabel("NRMSE")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y", alpha=0.25)
    _annotate_best_bar(axes[1], best_normalized, normalized_rmse[best_normalized])

    fig.suptitle(
        "Paper-Aligned Figure 11 - Metric Sensitivity Across Saved Experiments",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, save_path)


def plot_profile_panel_family(
    all_pred_raw: np.ndarray,
    test_raw: np.ndarray,
    time_axis: np.ndarray,
    seq_len: int,
    save_path: Path,
    representative_indices: Sequence[int],
    representative_labels: Sequence[str],
    title: str,
    selected_features: Sequence[str],
) -> list[int]:
    feature_indices = selected_feature_indices(FEATURE_NAMES, selected_features)
    colors = ("#2ca02c", "#1f77b4", "#d62728")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()

    for axis, feat_idx, feat_name in zip(axes, feature_indices, selected_features):
        for traj_idx, label, color in zip(representative_indices, representative_labels, colors):
            axis.plot(
                time_axis,
                test_raw[traj_idx, :, feat_idx],
                color=color,
                linewidth=1.8,
                label=f"{label} truth" if feat_idx == feature_indices[0] else None,
            )
            axis.plot(
                time_axis,
                all_pred_raw[traj_idx, :, feat_idx],
                color=color,
                linewidth=1.5,
                linestyle="--",
                label=f"{label} pred" if feat_idx == feature_indices[0] else None,
            )
        axis.axvline(time_axis[seq_len], color="gray", linestyle=":", alpha=0.6)
        axis.set_title(feat_name)
        axis.set_xlabel("Time (days)")
        axis.grid(True, alpha=0.25)

    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, save_path)
    return [int(idx) for idx in representative_indices]


def plot_representative_profile_panels(
    all_pred_raw: np.ndarray,
    test_raw: np.ndarray,
    rmse_per_traj: np.ndarray,
    time_axis: np.ndarray,
    seq_len: int,
    save_path: Path,
    selected_features: Sequence[str] = DEFAULT_PROFILE_FEATURES,
) -> list[int]:
    representative_indices = percentile_indices(rmse_per_traj, (10, 50, 90))
    representative_labels = ("P10", "P50", "P90")
    return plot_profile_panel_family(
        all_pred_raw=all_pred_raw,
        test_raw=test_raw,
        time_axis=time_axis,
        seq_len=seq_len,
        save_path=save_path,
        representative_indices=representative_indices,
        representative_labels=representative_labels,
        title="Paper-Aligned Figure 4 - Representative Rollout Profiles (RMSE percentiles)",
        selected_features=selected_features,
    )


def plot_novelty_profile_panels(
    all_pred_raw: np.ndarray,
    test_raw: np.ndarray,
    novelty_scores: np.ndarray,
    time_axis: np.ndarray,
    seq_len: int,
    save_path: Path,
    selected_features: Sequence[str] = DEFAULT_PROFILE_FEATURES,
) -> list[int]:
    representative_indices = select_novelty_representative_indices(novelty_scores)
    representative_labels = ("low novelty", "mid novelty", "high novelty")
    return plot_profile_panel_family(
        all_pred_raw=all_pred_raw,
        test_raw=test_raw,
        time_axis=time_axis,
        seq_len=seq_len,
        save_path=save_path,
        representative_indices=representative_indices,
        representative_labels=representative_labels,
        title="Paper-Aligned Figure 5 - Representative Rollout Profiles (novelty bins)",
        selected_features=selected_features,
    )


def plot_feature_time_heatmaps(
    truth: np.ndarray,
    pred: np.ndarray,
    time_axis: np.ndarray,
    save_path: Path,
) -> None:
    truth_rows = truth.T
    pred_rows = pred.T
    row_min = np.minimum(truth_rows.min(axis=1), pred_rows.min(axis=1))[:, None]
    row_max = np.maximum(truth_rows.max(axis=1), pred_rows.max(axis=1))[:, None]
    span = np.maximum(row_max - row_min, 1e-12)

    truth_scaled = (truth_rows - row_min) / span
    pred_scaled = (pred_rows - row_min) / span
    error_scaled = np.abs(pred_rows - truth_rows) / span

    fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
    payloads = (
        ("Truth (row-wise normalized)", truth_scaled, "viridis"),
        ("Prediction (row-wise normalized)", pred_scaled, "viridis"),
        ("Absolute error (row-wise normalized)", error_scaled, "magma"),
    )

    xticks = np.linspace(0, len(time_axis) - 1, 5, dtype=int)
    xticklabels = [f"{time_axis[idx]:.1f}" for idx in xticks]

    for axis, (title, values, cmap) in zip(axes, payloads):
        image = axis.imshow(values, aspect="auto", cmap=cmap, origin="lower")
        axis.set_title(title)
        axis.set_xlabel("Time (days)")
        axis.set_xticks(xticks)
        axis.set_xticklabels(xticklabels)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axes[0].set_yticks(range(len(FEATURE_NAMES)))
    axes[0].set_yticklabels(FEATURE_NAMES)
    axes[0].set_ylabel("Feature")

    fig.suptitle(
        "Paper-Aligned Figure 6 - Truth / Prediction / Error Heatmaps",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, save_path)


def plot_breakthrough_summary(
    all_pred_raw: np.ndarray,
    test_raw: np.ndarray,
    rmse_per_traj: np.ndarray,
    time_axis: np.ndarray,
    seq_len: int,
    save_path: Path,
    selected_features: Sequence[str] = DEFAULT_BREAKTHROUGH_FEATURES,
    reference_idx: int | None = None,
) -> int:
    feature_indices = selected_feature_indices(FEATURE_NAMES, selected_features)
    if reference_idx is None:
        reference_idx = select_reference_trajectory_index(rmse_per_traj)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()

    for axis, feat_idx, feat_name in zip(axes, feature_indices, selected_features):
        axis.plot(
            time_axis,
            test_raw[reference_idx, :, feat_idx],
            color="#1f3d73",
            linewidth=2.2,
            label="Ground truth",
        )
        axis.plot(
            time_axis,
            all_pred_raw[reference_idx, :, feat_idx],
            color="#C44E52",
            linewidth=2.0,
            linestyle="--",
            label="Prediction",
        )
        axis.axvline(time_axis[seq_len], color="gray", linestyle=":", alpha=0.6)
        axis.set_title(feat_name)
        axis.set_xlabel("Time (days)")
        axis.grid(True, alpha=0.25)

    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Paper-Aligned Figure 7 - Representative Output Curves (trajectory #{reference_idx})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, save_path)
    return reference_idx


def plot_ablation_and_sensitivity(
    summary_results: Mapping[str, Mapping[str, Any]],
    experiments_root: Path,
    save_path: Path,
) -> bool:
    seq_lens = sorted({int(values["seq_len"]) for values in summary_results.values()})
    hidden_sizes = sorted({int(values["hidden_size"]) for values in summary_results.values()})

    nrmse_matrix = np.full((len(hidden_sizes), len(seq_lens)), np.nan)
    overall_rmse_matrix = np.full((len(hidden_sizes), len(seq_lens)), np.nan)

    for experiment_name, values in summary_results.items():
        row = hidden_sizes.index(int(values["hidden_size"]))
        col = seq_lens.index(int(values["seq_len"]))
        normalized = normalize_result_metrics(values)
        nrmse_matrix[row, col] = normalized["nrmse_total"]

        stats_path = experiments_root / experiment_name / "comprehensive" / "comprehensive_stats.json"
        if stats_path.exists():
            with open(stats_path) as handle:
                stats = json.load(handle)
            overall_rmse_matrix[row, col] = stats["overall_rmse"]["mean"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    payloads = (
        ("Paper-Aligned Figure 10 - NRMSE ablation", nrmse_matrix, "NRMSE"),
        ("Paper-Aligned Figure 11 - Original-scale RMSE sensitivity", overall_rmse_matrix, "RMSE"),
    )

    for axis, (title, matrix, color_label) in zip(axes, payloads):
        image = axis.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("Sequence Length")
        axis.set_ylabel("Hidden Size")
        axis.set_xticks(range(len(seq_lens)))
        axis.set_xticklabels([str(value) for value in seq_lens])
        axis.set_yticks(range(len(hidden_sizes)))
        axis.set_yticklabels([str(value) for value in hidden_sizes])

        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                if np.isnan(matrix[row_idx, col_idx]):
                    continue
                axis.text(
                    col_idx,
                    row_idx,
                    f"{matrix[row_idx, col_idx]:.4f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label=color_label)

    fig.suptitle(
        "Paper-Aligned Figures 10-11 - Ablation and Sensitivity",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, save_path)
    return True


def plot_generalization_novelty(
    novelty_scores: np.ndarray,
    rmse_per_traj: np.ndarray,
    save_path: Path,
) -> list[dict[str, float]]:
    bin_summary = summarize_generalization_bins(novelty_scores, rmse_per_traj, n_bins=3)
    correlation = float(np.corrcoef(novelty_scores, rmse_per_traj)[0, 1]) if len(novelty_scores) > 1 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(novelty_scores, rmse_per_traj, s=18, alpha=0.7, color="#4C72B0")
    axes[0].set_xlabel("Initial-state novelty score")
    axes[0].set_ylabel("Per-trajectory RMSE (original scale)")
    axes[0].set_title(f"Novelty vs RMSE (corr={correlation:.3f})")
    axes[0].grid(True, alpha=0.25)

    labels = [item["bin_label"] for item in bin_summary]
    means = [item["mean_rmse"] for item in bin_summary]
    errors = [item["std_rmse"] for item in bin_summary]
    axes[1].bar(labels, means, yerr=errors, color=["#2ca02c", "#ffbf00", "#d62728"], alpha=0.8)
    axes[1].set_ylabel("Mean RMSE (original scale)")
    axes[1].set_title("Generalization bins by novelty")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Paper-Aligned Figure 12 - Generalization vs Initial-State Novelty",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, save_path)
    return bin_summary
