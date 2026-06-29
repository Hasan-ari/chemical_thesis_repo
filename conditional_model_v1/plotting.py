from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(history: list[dict[str, float | int]], path: Path | str) -> None:
    """Plot train/validation normalized MSE over epochs."""
    path = Path(path)
    epochs = [int(row["epoch"]) for row in history]
    train = [float(row["train_loss"]) for row in history]
    val = [float(row["val_loss"]) for row in history if row.get("val_loss") is not None]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train, label="train normalized MSE")
    if val:
        plt.plot(epochs[: len(val)], val, label="val normalized MSE")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training dynamics")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def plot_trajectory_examples(
    *,
    time_axis: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_features: tuple[str, ...],
    run_ids: list[str],
    output_dir: Path | str,
    max_runs: int | None = 3,
    feature_names: tuple[str, ...] | None = None,
) -> None:
    """Save real trajectory vs prediction plots.

    `feature_names=None` means all output features.
    `max_runs=None` means render every run in the evaluation split.
    The full numeric prediction arrays are saved by the CLI separately, so PNGs
    are only the human-inspection layer.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_to_index = {name: index for index, name in enumerate(output_features)}
    selected_features = (
        list(output_features)
        if feature_names is None
        else [name for name in feature_names if name in feature_to_index]
    )
    run_count = y_true.shape[0] if max_runs is None else min(max_runs, y_true.shape[0])
    for run_index in range(run_count):
        safe_run = _safe_name(run_ids[run_index])
        _plot_all_outputs_grid(
            time_axis=time_axis,
            y_true=y_true[run_index],
            y_pred=y_pred[run_index],
            selected_features=selected_features,
            feature_to_index=feature_to_index,
            run_id=run_ids[run_index],
            path=output_dir / f"{safe_run}_all_outputs.png",
        )
        for feature in selected_features:
            feature_index = feature_to_index[feature]
            plt.figure(figsize=(8, 5))
            plt.plot(time_axis, y_true[run_index, :, feature_index], label="PHREEQC true")
            plt.plot(time_axis, y_pred[run_index, :, feature_index], label="LSTM prediction")
            plt.xlabel("time_d")
            plt.ylabel(feature)
            plt.title(f"{run_ids[run_index]} - {feature}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"{safe_run}_{feature}.png", dpi=160)
            plt.close()


def _plot_all_outputs_grid(
    *,
    time_axis: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    selected_features: list[str],
    feature_to_index: dict[str, int],
    run_id: str,
    path: Path,
) -> None:
    """Render one compact overview PNG containing every selected output feature."""
    if not selected_features:
        return
    columns = 4
    rows = math.ceil(len(selected_features) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4.2, rows * 2.4), squeeze=False)
    for axis in axes.ravel():
        axis.set_visible(False)
    for plot_index, feature in enumerate(selected_features):
        axis = axes.ravel()[plot_index]
        axis.set_visible(True)
        feature_index = feature_to_index[feature]
        axis.plot(time_axis, y_true[:, feature_index], label="true", linewidth=1.2)
        axis.plot(time_axis, y_pred[:, feature_index], label="pred", linewidth=1.2)
        axis.set_title(feature, fontsize=9)
        axis.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{run_id} - all selected outputs", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _safe_name(value: str) -> str:
    """Make run ids safe for flat PNG filenames."""
    return value.replace(":", "_").replace("/", "_")
