"""Per-run RMSE box plots from a conditional_model_v1 eval_predictions.npz.

For every evaluation run r and output feature f the script computes the
full-trajectory RMSE over all timesteps:

    rmse[r, f] = sqrt(mean_t (y_pred[r, t, f] - y_true[r, t, f])^2)

so each box in the plots summarises the distribution of one scalar per test
run: median, quartiles, whiskers, and outliers. This is the view the advisor
asked for -- the spread and the bad cases, not the single best run.

Outputs (under --out-dir):

    per_run_rmse.csv                    long table: run_id, rock, feature, both scales
    rmse_summary.csv                    per rock x feature quantiles
    rmse_box_main_normalized.png        selected high-variance features, 4 rocks
    rmse_box_h2_co2_original.png        H2 and CO2 gas moles in one plot, 4 rocks
    rmse_box_<rock>_all_normalized.png  appendix: all 32 features for one rock
    loss_curve_log.png                  history.csv train/val loss on log axis
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator

ROCK_ORDER: tuple[str, ...] = ("Calcite", "Dolomite", "Halite", "Trona")
ROCK_COLORS: dict[str, str] = {
    "Calcite": "#2563EB",
    "Dolomite": "#D97706",
    "Halite": "#65752A",
    "Trona": "#BE5678",
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d9d8d4"
SURFACE = "#fcfcfb"

MAIN_FEATURES: tuple[str, ...] = (
    "pH2_atm",
    "pCO2_atm",
    "H2_g_mol",
    "CO2_g_mol",
    "CH4_g_mol",
    "pH",
)
H2_CO2_FEATURES: tuple[str, ...] = ("H2_g_mol", "CO2_g_mol")

DEFAULT_RUN_DIR = (
    Path("conditional_model_v1")
    / "from_google_drive"
    / "20260713_073611_four_rock_condition_lstm_v1"
)
DEFAULT_OUT_DIR = Path("results") / "rmse_boxplots"


def per_run_rmse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Full-trajectory RMSE per (run, feature): sqrt over the time axis mean."""
    error = y_pred.astype(np.float64) - y_true.astype(np.float64)
    return np.sqrt(np.mean(error**2, axis=1))


def load_arrays(run_dir: Path) -> dict[str, np.ndarray]:
    with np.load(run_dir / "eval_predictions.npz") as bundle:
        return {key: bundle[key] for key in bundle.files}


def write_per_run_csv(
    path: Path,
    *,
    run_ids: np.ndarray,
    rocks: np.ndarray,
    features: list[str],
    rmse_original: np.ndarray,
    rmse_normalized: np.ndarray,
) -> None:
    with path.open("w", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["run_id", "rock", "feature", "rmse_original", "rmse_normalized"])
        for run_index, (run_id, rock) in enumerate(zip(run_ids, rocks)):
            for feature_index, feature in enumerate(features):
                writer.writerow(
                    [
                        run_id,
                        rock,
                        feature,
                        f"{rmse_original[run_index, feature_index]:.10e}",
                        f"{rmse_normalized[run_index, feature_index]:.10e}",
                    ]
                )


def write_summary_csv(
    path: Path,
    *,
    rocks: np.ndarray,
    features: list[str],
    rmse_original: np.ndarray,
    rmse_normalized: np.ndarray,
) -> None:
    quantiles = (1, 25, 50, 75, 95, 99)
    with path.open("w", newline="") as file_obj:
        writer = csv.writer(file_obj)
        header = ["rock", "feature", "n_runs"]
        for scale in ("original", "normalized"):
            header.extend(f"{scale}_p{q}" for q in quantiles)
            header.append(f"{scale}_max")
        writer.writerow(header)
        for rock in ROCK_ORDER:
            mask = rocks == rock
            for feature_index, feature in enumerate(features):
                row: list[object] = [rock, feature, int(mask.sum())]
                for values in (
                    rmse_original[mask, feature_index],
                    rmse_normalized[mask, feature_index],
                ):
                    row.extend(f"{v:.10e}" for v in np.percentile(values, quantiles))
                    row.append(f"{values.max():.10e}")
                writer.writerow(row)


def _label_log_axis(ax: plt.Axes) -> None:
    """Label intermediate log-axis values; decade labels stay bold."""
    decimal = FuncFormatter(lambda value, _pos: f"{value:g}")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5)))
    ax.yaxis.set_major_formatter(decimal)
    ax.yaxis.set_minor_formatter(decimal)
    ax.tick_params(axis="y", which="major", labelsize=10)
    ax.tick_params(axis="y", which="minor", labelsize=7, labelcolor=TEXT_SECONDARY)
    for label in ax.yaxis.get_ticklabels(which="major"):
        label.set_fontweight("bold")
        label.set_color(TEXT_PRIMARY)
    ax.yaxis.grid(True, which="minor", color=GRID_COLOR, linewidth=0.3, alpha=0.6)


def _styled_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def grouped_rmse_boxplot(
    *,
    rmse: np.ndarray,
    rocks: np.ndarray,
    features: list[str],
    plot_features: tuple[str, ...],
    title: str,
    ylabel: str,
    out_path: Path,
    figsize: tuple[float, float] = (10.0, 5.0),
) -> None:
    """One box per (feature, rock); rocks keep a fixed order and color."""
    n_rocks = len(ROCK_ORDER)
    box_width = 0.16
    group_span = box_width * n_rocks + 0.08 * (n_rocks - 1)

    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _styled_axes(ax)

    for rock_index, rock in enumerate(ROCK_ORDER):
        mask = rocks == rock
        data = [
            rmse[mask, features.index(feature)]
            for feature in plot_features
        ]
        offset = (rock_index - (n_rocks - 1) / 2) * (group_span / n_rocks)
        positions = np.arange(len(plot_features)) + offset
        color = ROCK_COLORS[rock]
        ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            boxprops={"facecolor": color, "edgecolor": color, "linewidth": 0.8},
            whiskerprops={"color": color, "linewidth": 1.0},
            capprops={"color": color, "linewidth": 1.0},
            medianprops={"color": SURFACE, "linewidth": 1.2},
            flierprops={
                "marker": "o",
                "markersize": 2.0,
                "markerfacecolor": color,
                "markeredgecolor": "none",
                "alpha": 0.35,
            },
        )

    ax.set_yscale("log")
    _label_log_axis(ax)
    ax.set_xticks(np.arange(len(plot_features)))
    ax.set_xticklabels(plot_features, color=TEXT_PRIMARY, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_PRIMARY, fontsize=10)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, pad=12)
    ax.legend(
        handles=[Patch(facecolor=ROCK_COLORS[rock], label=rock) for rock in ROCK_ORDER],
        loc="upper right",
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_PRIMARY,
    )
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def per_rock_all_feature_boxplot(
    *,
    rmse: np.ndarray,
    rocks: np.ndarray,
    features: list[str],
    rock: str,
    ylabel: str,
    out_path: Path,
) -> None:
    mask = rocks == rock
    data = [rmse[mask, feature_index] for feature_index in range(len(features))]

    fig, ax = plt.subplots(figsize=(14.0, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _styled_axes(ax)
    color = ROCK_COLORS[rock]
    ax.boxplot(
        data,
        positions=np.arange(len(features)),
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": color, "edgecolor": color, "linewidth": 0.8},
        whiskerprops={"color": color, "linewidth": 0.9},
        capprops={"color": color, "linewidth": 0.9},
        medianprops={"color": SURFACE, "linewidth": 1.1},
        flierprops={
            "marker": "o",
            "markersize": 1.6,
            "markerfacecolor": color,
            "markeredgecolor": "none",
            "alpha": 0.3,
        },
    )
    ax.set_yscale("log")
    _label_log_axis(ax)
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=60, ha="right", color=TEXT_PRIMARY, fontsize=8)
    ax.set_ylabel(ylabel, color=TEXT_PRIMARY, fontsize=10)
    ax.set_title(
        f"{rock}: per-run RMSE across all output features (test split)",
        color=TEXT_PRIMARY,
        fontsize=12,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def loss_curve_log(history_path: Path, out_path: Path) -> None:
    epochs: list[int] = []
    train_loss: list[float] = []
    val_loss: list[float] = []
    with history_path.open() as file_obj:
        for row in csv.DictReader(file_obj):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))

    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _styled_axes(ax)
    ax.plot(epochs, train_loss, color="#2563EB", linewidth=1.6, label="train")
    ax.plot(epochs, val_loss, color="#D97706", linewidth=1.6, label="validation")
    ax.set_yscale("log")
    _label_log_axis(ax)
    ax.set_xlabel("epoch", color=TEXT_PRIMARY, fontsize=10)
    ax.set_ylabel("normalized MSE (log)", color=TEXT_PRIMARY, fontsize=10)
    ax.set_title("Training dynamics (log scale)", color=TEXT_PRIMARY, fontsize=12, pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-csv", action="store_true")
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_arrays(run_dir)
    features = [str(name) for name in arrays["output_features"]]
    rocks = arrays["rocks"]
    run_ids = arrays["run_ids"]

    rmse_original = per_run_rmse(arrays["y_pred"], arrays["y_true"])
    rmse_normalized = per_run_rmse(arrays["y_pred_norm"], arrays["y_true_norm"])

    if not args.skip_csv:
        write_per_run_csv(
            out_dir / "per_run_rmse.csv",
            run_ids=run_ids,
            rocks=rocks,
            features=features,
            rmse_original=rmse_original,
            rmse_normalized=rmse_normalized,
        )
        write_summary_csv(
            out_dir / "rmse_summary.csv",
            rocks=rocks,
            features=features,
            rmse_original=rmse_original,
            rmse_normalized=rmse_normalized,
        )

    grouped_rmse_boxplot(
        rmse=rmse_normalized,
        rocks=rocks,
        features=features,
        plot_features=MAIN_FEATURES,
        title="Per-run RMSE of high-variance outputs by rock (test split)",
        ylabel="per-run RMSE, normalized scale (log)",
        out_path=out_dir / "rmse_box_main_normalized.png",
    )
    grouped_rmse_boxplot(
        rmse=rmse_original,
        rocks=rocks,
        features=features,
        plot_features=H2_CO2_FEATURES,
        title="Per-run RMSE of H2 and CO2 gas amounts by rock (test split)",
        ylabel="per-run RMSE, original scale [mol] (log)",
        out_path=out_dir / "rmse_box_h2_co2_original.png",
        figsize=(7.0, 5.0),
    )
    for rock in ROCK_ORDER:
        per_rock_all_feature_boxplot(
            rmse=rmse_normalized,
            rocks=rocks,
            features=features,
            rock=rock,
            ylabel="per-run RMSE, normalized scale (log)",
            out_path=out_dir / f"rmse_box_{rock.lower()}_all_normalized.png",
        )

    loss_curve_log(run_dir / "history.csv", out_dir / "loss_curve_log.png")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
