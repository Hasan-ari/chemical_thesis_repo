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


def plot_rock_overviews(
    *,
    time_axis: np.ndarray,
    output_features: tuple[str, ...],
    overviews: dict[str, dict[str, dict[str, object]]],
    output_dir: Path | str,
) -> None:
    """Render best, worst, and mean all-output grids for every rock."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_features = list(output_features)
    feature_to_index = {
        feature: index for index, feature in enumerate(output_features)
    }
    for rock, rock_overviews in overviews.items():
        for kind in ("best", "worst", "mean"):
            overview = rock_overviews[kind]
            _plot_all_outputs_grid(
                time_axis=time_axis,
                y_true=np.asarray(overview["y_true"]),
                y_pred=np.asarray(overview["y_pred"]),
                selected_features=selected_features,
                feature_to_index=feature_to_index,
                run_id=str(overview["label"]),
                path=output_dir / f"{_safe_name(rock)}_{kind}_overview.png",
            )


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


def plot_loro_rock_bars(
    *,
    rows: list[dict[str, object]],
    path_normalized: Path | str,
    path_original: Path | str,
) -> None:
    """Grouped bars per held-out rock: unseen vs seen vs all-rocks reference.

    ``rows`` are ``loro.summary_rows`` dicts. Missing seen/reference values
    (``test_ratio=0`` or no reference run) simply leave that bar out.
    """
    if not rows:
        raise ValueError("plot_loro_rock_bars needs at least one fold row")
    specs = (
        (
            path_normalized,
            "normalized RMSE (each fold uses its own scaler)",
            "unseen_rmse_mean_normalized",
            "seen_rmse_mean_normalized",
            "reference_rmse_mean_normalized",
        ),
        (
            path_original,
            "RMSE, original chemistry units (mean over 26 outputs)",
            "unseen_rmse_mean_original",
            "seen_rmse_mean_original",
            "reference_rmse_mean_original",
        ),
    )
    rocks = [str(row["held_out_rock"]) for row in rows]
    positions = np.arange(len(rocks))
    width = 0.27
    for path, ylabel, unseen_key, seen_key, reference_key in specs:
        fig, axis = plt.subplots(figsize=(max(7.0, 1.3 * len(rocks) + 2.0), 4.8))
        series = (
            ("held-out rock (unseen)", unseen_key, -width),
            ("other rocks' test share (seen)", seen_key, 0.0),
            ("all-rocks reference model", reference_key, width),
        )
        for label, key, offset in series:
            values = [row.get(key) for row in rows]
            if all(value is None for value in values):
                continue
            heights = [float(value) if value is not None else 0.0 for value in values]
            axis.bar(positions + offset, heights, width=width, label=label)
        axis.set_xticks(positions)
        axis.set_xticklabels(rocks, rotation=20, ha="right", fontsize=9)
        axis.set_ylabel(ylabel, fontsize=9)
        axis.set_title("Leave-one-rock-out: error on the rock the model never saw", fontsize=11)
        # Headroom so the legend never covers the tallest bar.
        top = max((float(row.get(key) or 0.0) for row in rows for _, key, _ in series), default=1.0)
        axis.set_ylim(0.0, top * 1.35 if top > 0 else 1.0)
        axis.legend(fontsize=8, ncol=3, loc="upper center")
        axis.grid(axis="y", linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)


def plot_loro_feature_heatmap(
    *,
    feature_rows: list[dict[str, object]],
    rocks: tuple[str, ...],
    output_features: tuple[str, ...],
    path: Path | str,
) -> None:
    """Held-out rock x output feature grid of normalized RMSE."""
    matrix = np.full((len(rocks), len(output_features)), np.nan)
    rock_index = {rock: index for index, rock in enumerate(rocks)}
    feature_index = {feature: index for index, feature in enumerate(output_features)}
    for row in feature_rows:
        rock = str(row["held_out_rock"])
        feature = str(row["feature"])
        if rock not in rock_index or feature not in feature_index:
            raise ValueError(f"Unexpected heatmap cell: rock={rock!r} feature={feature!r}")
        matrix[rock_index[rock], feature_index[feature]] = float(row["rmse_normalized"])
    if np.isnan(matrix).any():
        missing = int(np.isnan(matrix).sum())
        raise ValueError(f"Heatmap is missing {missing} rock x feature cells")

    fig, axis = plt.subplots(
        figsize=(max(10.0, 0.45 * len(output_features) + 2.0), max(3.5, 0.5 * len(rocks) + 1.5))
    )
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(len(output_features)))
    axis.set_xticklabels(output_features, rotation=75, ha="right", fontsize=7)
    axis.set_yticks(np.arange(len(rocks)))
    axis.set_yticklabels(rocks, fontsize=8)
    for row_index in range(len(rocks)):
        for column_index in range(len(output_features)):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if matrix[row_index, column_index] < np.nanmax(matrix) * 0.6 else "black",
            )
    fig.colorbar(image, ax=axis, label="normalized RMSE on the held-out rock")
    axis.set_title("Leave-one-rock-out: which outputs fail on an unseen rock", fontsize=11)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
