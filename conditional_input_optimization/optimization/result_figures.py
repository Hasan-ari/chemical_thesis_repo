"""Draw the PNG figures of one optimization results folder.

The figures show the surrogate acceptance gate, the search, and the final
candidates. The module reads the CSV files that report.write_results
writes. It uses the object-oriented Matplotlib API with the Agg canvas, so
it changes no global pyplot state. See
https://matplotlib.org/stable/gallery/user_interfaces/web_application_server_sgskip.html

The colors are the validated default palette of the data-viz guide: the
categorical slots 1 to 3 (blue, orange, aqua) and the status red. Red
only means "fails the gate" or "predicted H2 gain (not physical)".

Usage:
    python -m conditional_input_optimization.optimization.result_figures \
        --results conditional_input_optimization/results/<run>
"""

from __future__ import annotations

import argparse
import functools
import logging
import math
import pathlib
from collections.abc import Callable, Sequence
from typing import ParamSpec, TypeVar

import matplotlib
import matplotlib.axes
import matplotlib.backends.backend_agg
import matplotlib.figure
import matplotlib.lines
import matplotlib.ticker
import matplotlib.transforms
import numpy as np
import pandas as pd

from conditional_input_optimization.optimization import (
    diagnostics,
    experiment_config,
)

_LOGGER = logging.getLogger(__name__)
_P = ParamSpec("_P")
_R = TypeVar("_R")

FIGURE_FILE_NAMES = (
    "01_surrogate_gate.png",
    "02_predicted_negative_share.png",
    "03_search_convergence.png",
    "04_candidates_vs_starts.png",
    "05_input_roles.png",
    "06_usable_candidate_h2.png",
)
FIGURES_DIR_NAME = "figures"
CONFIG_FILE_NAME = "config.yaml"
CSV_FILE_NAMES = {
    "surrogate_check": "surrogate_check.csv",
    "search_history": "search_history.csv",
    "all_candidates": "all_candidates.csv",
    "baseline": "baseline.csv",
    "input_space": "input_space.csv",
    "h2_trajectories": "h2_trajectories.csv",
}
MODES = ("bounded", "unbounded")
OBJECTIVES = ("relative", "absolute")
ROLES = ("free", "tied", "fixed")
TEST_SPLIT = "test"
_CANDIDATE_COLUMNS = (
    "rock",
    "mode",
    "start_index",
    "predicted_relative_loss",
    "negative",
    "usable",
)

_DPI = 150
# Data-viz default palette, light mode: categorical slots 1-3 pass the
# all-pairs CVD check on white (validate_palette.js, worst delta E 9.2).
_SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
_CRITICAL = "#d03b3b"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_AXIS_LINE = "#c3c2b7"
_BACKGROUND = "#ffffff"
_INSIDE_LABEL = "#ffffff"
_FAIL_WASH_ALPHA = 0.07
_LINE_WIDTH = 1.2
_HAIRLINE = 0.6
_THRESHOLD_DASH = (0, (4, 3))
_SMALL_FONT = 7
_BASE_FONT = 9
_RC_PARAMS = {
    "font.size": _BASE_FONT,
    "font.family": "sans-serif",
    "text.color": _INK_PRIMARY,
    "axes.edgecolor": _AXIS_LINE,
    "axes.labelcolor": _INK_SECONDARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": _BASE_FONT,
    "axes.titlelocation": "left",
    "xtick.color": _INK_MUTED,
    "ytick.color": _INK_MUTED,
    "xtick.labelcolor": _INK_SECONDARY,
    "ytick.labelcolor": _INK_SECONDARY,
    "legend.frameon": False,
    "figure.facecolor": _BACKGROUND,
    "savefig.facecolor": _BACKGROUND,
}
# The symmetric log scale is linear in [-limit, limit] and logarithmic
# outside. The losses span 0.001 to 100, so a linear axis hides the
# bounded mode.
_SYMLOG_LIMIT = 0.01
_SYMLOG_LINEAR_SCALE = 0.5
_LOSS_TICKS = (-1000.0, -100.0, -10.0, -1.0, -0.1, 0.0)
_LOSS_AXIS_RIGHT = 0.02
_LOSS_AXIS_MARGIN = 1.5
_BAR_HEIGHT = 0.32
_ROCK_ROW_HEIGHT = 0.55
_FIGURE_WIDTH = 9.0
_GRID_COLUMNS = 2
_PANEL_HEIGHT = 1.6
_JITTER_PRIME = 37
_JITTER_WIDTH = 0.3
_MODE_ROW_OFFSET = 0.2
_DOT_AREA = 16
_MARKER_AREA = 34
_PERCENT = 100.0
_MAX_TRAJECTORIES = len(_SERIES_COLORS)
_LABEL_OFFSET = 8


def _style(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run a figure function inside the module rcParams.

    Decorator pattern: the rcParams change only inside the call.
    """

    @functools.wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with matplotlib.rc_context(_RC_PARAMS):
            return function(*args, **kwargs)

    return wrapper


def _new_figure(
    height: float, nrows: int = 1, ncols: int = 1, **share: bool
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Return a figure on an Agg canvas and its 2D array of axes."""
    figure = matplotlib.figure.Figure(
        figsize=(_FIGURE_WIDTH, height), layout="constrained"
    )
    matplotlib.backends.backend_agg.FigureCanvasAgg(figure)
    axes = figure.subplots(nrows, ncols, squeeze=False, **share)
    return figure, axes


def _save(
    figure: matplotlib.figure.Figure, output_path: pathlib.Path
) -> pathlib.Path:
    """Write the figure as a PNG file and return its path."""
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=_DPI)
    return output_path


def _style_axes(axes: matplotlib.axes.Axes, grid_axis: str) -> None:
    """Draw a hairline grid behind the data and hide the tick marks."""
    axes.grid(axis=grid_axis, color=_GRIDLINE, linewidth=_HAIRLINE)
    axes.set_axisbelow(True)
    axes.tick_params(length=0)


def _rock_axis(axes: matplotlib.axes.Axes, rocks: Sequence[str]) -> None:
    """Put the rocks on the y-axis, the first rock at the top."""
    axes.set_yticks(np.arange(len(rocks)))
    axes.set_yticklabels([rock.replace("_", " ") for rock in rocks])
    axes.set_ylim(len(rocks) - 0.5, -0.5)


def _require_columns(
    frame: pd.DataFrame, columns: Sequence[str], table: str
) -> None:
    """Raise ValueError if the table does not have all the columns."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{table} has no columns {missing}.")


def _by_rock(
    frame: pd.DataFrame, rocks: Sequence[str], table: str
) -> pd.DataFrame:
    """Return one row for each rock, in the order of rocks.

    Raises:
        ValueError: If a rock has no row or more than one row.
    """
    if frame["rock"].duplicated().any():
        raise ValueError(f"{table} has more than one row for a rock.")
    missing = sorted(set(rocks) - set(frame["rock"]))
    if missing:
        raise ValueError(f"{table} has no rows for the rocks {missing}.")
    return frame.set_index("rock").loc[list(rocks)]


def _test_metrics(
    surrogate_check: pd.DataFrame, objective: str, rocks: Sequence[str]
) -> pd.DataFrame:
    """Return the test-split rows of one objective, one row for each rock."""
    rows = surrogate_check[
        (surrogate_check["split"] == TEST_SPLIT)
        & (surrogate_check["objective"] == objective)
    ]
    return _by_rock(rows, rocks, f"surrogate_check ({objective}, test)")


def _draw_threshold(
    axes: matplotlib.axes.Axes, value: float, label: str, fail_low: bool
) -> None:
    """Draw a dashed gate line, its label, and a red wash on the fail side."""
    left, right = axes.get_xlim()
    fail_span = (left, value) if fail_low else (value, right)
    axes.axvspan(*fail_span, color=_CRITICAL, alpha=_FAIL_WASH_ALPHA, lw=0)
    axes.axvline(
        value, color=_INK_SECONDARY, linewidth=1, linestyle=_THRESHOLD_DASH
    )
    transform = matplotlib.transforms.blended_transform_factory(
        axes.transData, axes.transAxes
    )
    axes.text(
        value,
        1.01,
        label,
        transform=transform,
        ha="center",
        va="bottom",
        fontsize=_SMALL_FONT,
        color=_INK_SECONDARY,
    )
    axes.set_xlim(left, right)


def _draw_spearman_panel(
    axes: matplotlib.axes.Axes,
    metrics: dict[str, pd.DataFrame],
    acceptance: diagnostics.AcceptanceRule,
) -> None:
    """Draw grouped bars of the test Spearman correlation of each rock."""
    for index, (objective, table) in enumerate(metrics.items()):
        positions = np.arange(len(table)) + (index - 0.5) * _BAR_HEIGHT
        values = table["spearman"].to_numpy()
        axes.barh(
            positions,
            values,
            height=_BAR_HEIGHT * 0.9,
            color=_SERIES_COLORS[index],
            label=f"{objective.capitalize()} loss",
        )
        for position, value in zip(positions, values, strict=True):
            axes.text(
                value,
                position,
                f" {value:.2f} ",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=_SMALL_FONT,
                color=_INK_SECONDARY,
            )
    axes.axvline(0, color=_AXIS_LINE, linewidth=_HAIRLINE)
    axes.set_xlim(-0.6, 1.0)
    _draw_threshold(
        axes, acceptance.min_spearman, f"Gate ≥ {acceptance.min_spearman}", True
    )
    axes.set_xlabel("Spearman correlation, test runs (higher is better)")
    _style_axes(axes, "x")


def _draw_error_panel(
    axes: matplotlib.axes.Axes,
    metrics: dict[str, pd.DataFrame],
    acceptance: diagnostics.AcceptanceRule,
) -> None:
    """Draw dots of the test MAE / std of each rock on a log axis."""
    for index, table in enumerate(metrics.values()):
        positions = np.arange(len(table)) + (index - 0.5) * _BAR_HEIGHT
        axes.scatter(
            table["mae_over_std"],
            positions,
            s=_MARKER_AREA,
            color=_SERIES_COLORS[index],
            edgecolors=_BACKGROUND,
            linewidths=1,
            zorder=3,
        )
    values = pd.concat([table["mae_over_std"] for table in metrics.values()])
    low = min(values.min(), acceptance.max_mae_over_std) / 2
    axes.set_xscale("log")
    axes.set_xlim(low, values.max() * 2)
    axes.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    _draw_threshold(
        axes,
        acceptance.max_mae_over_std,
        f"Gate ≤ {acceptance.max_mae_over_std}",
        False,
    )
    axes.set_xlabel("MAE / std, test runs (log scale, lower is better)")
    _style_axes(axes, "x")


def _pass_count(
    table: pd.DataFrame, acceptance: diagnostics.AcceptanceRule
) -> int:
    """Return the number of rocks that pass both gate limits."""
    passed = (table["spearman"] >= acceptance.min_spearman) & (
        table["mae_over_std"] <= acceptance.max_mae_over_std
    )
    return int(passed.sum())


def _gate_title(
    metrics: dict[str, pd.DataFrame], acceptance: diagnostics.AcceptanceRule
) -> str:
    """Return the title with the pass count of each objective."""
    passes = ", ".join(
        f"{objective} loss {_pass_count(table, acceptance)} of {len(table)}"
        for objective, table in metrics.items()
    )
    return f"Surrogate acceptance gate. Rocks that pass: {passes}"


@_style
def plot_surrogate_gate(
    surrogate_check: pd.DataFrame,
    rocks: Sequence[str],
    acceptance: diagnostics.AcceptanceRule,
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Draw the test Spearman and MAE / std of each rock with the gate.

    Args:
        surrogate_check: The rows of surrogate_check.csv.
        rocks: The rocks, in the display order (top to bottom).
        acceptance: The gate limits of the experiment.
        output_path: The PNG file to write.

    Returns:
        The path of the written PNG file.

    Raises:
        ValueError: If a column or a test row of a rock is missing.
    """
    _require_columns(
        surrogate_check,
        ("rock", "objective", "split", "spearman", "mae_over_std"),
        "surrogate_check",
    )
    metrics = {
        objective: _test_metrics(surrogate_check, objective, rocks)
        for objective in OBJECTIVES
    }
    figure, axes = _new_figure(
        len(rocks) * _ROCK_ROW_HEIGHT + 1.6, ncols=2, sharey=True
    )
    _draw_spearman_panel(axes[0, 0], metrics, acceptance)
    _draw_error_panel(axes[0, 1], metrics, acceptance)
    _rock_axis(axes[0, 0], rocks)
    figure.suptitle(_gate_title(metrics, acceptance))
    figure.legend(loc="outside lower center", ncols=len(OBJECTIVES))
    return _save(figure, output_path)


def _real_note(lowest_real: pd.Series) -> str:
    """Return the note on the real share, from the lowest real loss."""
    if (lowest_real > 0).all():
        return "Real runs: 0 % (the lowest real loss of each rock is positive)."
    return "Real runs: some rocks have a negative real loss."


@_style
def plot_negative_share(
    surrogate_check: pd.DataFrame,
    baseline: pd.DataFrame,
    rocks: Sequence[str],
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Draw the share of test runs with a predicted negative relative loss.

    Args:
        surrogate_check: The rows of surrogate_check.csv.
        baseline: The rows of baseline.csv. It gives the lowest real loss.
        rocks: The rocks, in the display order (top to bottom).
        output_path: The PNG file to write.

    Returns:
        The path of the written PNG file.

    Raises:
        ValueError: If a column or a row of a rock is missing.
    """
    _require_columns(
        surrogate_check, ("share_predicted_negative",), "surrogate_check"
    )
    _require_columns(baseline, ("rock", "real_best_real"), "baseline")
    table = _test_metrics(surrogate_check, OBJECTIVES[0], rocks)
    shares = table["share_predicted_negative"].to_numpy() * _PERCENT
    lowest_real = _by_rock(baseline, rocks, "baseline")["real_best_real"]
    figure, axes = _new_figure(len(rocks) * _ROCK_ROW_HEIGHT * 0.8 + 1.2)
    plot = axes[0, 0]
    plot.barh(np.arange(len(rocks)), shares, height=0.6, color=_CRITICAL)
    for position, share in enumerate(shares):
        plot.text(share, position, f" {share:.1f} %", va="center")
    plot.set_xlim(0, _PERCENT)
    plot.set_xlabel("Test runs with a predicted H2 gain (%)")
    _rock_axis(plot, rocks)
    _style_axes(plot, "x")
    figure.suptitle(
        "Share of test runs with a predicted negative relative loss\n"
        + _real_note(lowest_real)
    )
    return _save(figure, output_path)


def _loss_axis(
    axes: matplotlib.axes.Axes, axis_name: str, lowest: float
) -> None:
    """Set a symmetric log scale and plain tick labels on one loss axis.

    Args:
        axes: The plot.
        axis_name: "x" or "y".
        lowest: The lowest loss to show.
    """
    set_scale, set_ticks, set_limits = {
        "x": (axes.set_xscale, axes.set_xticks, axes.set_xlim),
        "y": (axes.set_yscale, axes.set_yticks, axes.set_ylim),
    }[axis_name]
    set_scale("symlog", linthresh=_SYMLOG_LIMIT, linscale=_SYMLOG_LINEAR_SCALE)
    low = min(lowest * _LOSS_AXIS_MARGIN, -_SYMLOG_LIMIT)
    ticks = [tick for tick in _LOSS_TICKS if tick >= low]
    set_ticks(ticks, labels=[f"{tick:g}" for tick in ticks])
    set_limits(low, _LOSS_AXIS_RIGHT)


def _draw_convergence_panel(
    axes: matplotlib.axes.Axes, rows: pd.DataFrame, rock: str, lowest: float
) -> None:
    """Draw the min loss of each mode against the step for one rock."""
    for color, mode in zip(_SERIES_COLORS, MODES, strict=False):
        curve = rows[rows["mode"] == mode].sort_values("step")
        axes.plot(curve["step"], curve["min_loss"], color=color, lw=_LINE_WIDTH)
    axes.axhline(0, color=_INK_MUTED, linewidth=_HAIRLINE)
    axes.set_title(rock.replace("_", " "))
    _loss_axis(axes, "y", lowest)
    _style_axes(axes, "y")


@_style
def plot_search_convergence(
    search_history: pd.DataFrame,
    rocks: Sequence[str],
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Draw the min loss of the starts at each step, one panel for each rock.

    Args:
        search_history: The rows of search_history.csv.
        rocks: The rocks, in the panel order (row by row).
        output_path: The PNG file to write.

    Returns:
        The path of the written PNG file.

    Raises:
        ValueError: If a column is missing.
    """
    _require_columns(
        search_history, ("rock", "mode", "step", "min_loss"), "search_history"
    )
    nrows = math.ceil(len(rocks) / _GRID_COLUMNS)
    figure, grid = _new_figure(
        nrows * _PANEL_HEIGHT + 0.9, nrows, _GRID_COLUMNS, sharex=True
    )
    lowest = float(search_history["min_loss"].min())
    for axes, rock in zip(grid.flat, rocks, strict=False):
        rows = search_history[search_history["rock"] == rock]
        _draw_convergence_panel(axes, rows, rock, lowest)
    for axes in grid.flat[len(rocks) :]:
        axes.set_visible(False)
    figure.supxlabel("Search step", color=_INK_SECONDARY)
    figure.supylabel("Min penalized loss (symlog)", color=_INK_SECONDARY)
    figure.suptitle("Search convergence: the lowest loss of all starts")
    figure.legend(handles=_line_keys(MODES), loc="outside upper right", ncols=2)
    return _save(figure, output_path)


def _line_keys(names: Sequence[str]) -> list[matplotlib.lines.Line2D]:
    """Return one line legend key for each name, in the series colors."""
    return [
        matplotlib.lines.Line2D(
            [], [], color=color, lw=_LINE_WIDTH, label=name.capitalize()
        )
        for color, name in zip(_SERIES_COLORS, names, strict=False)
    ]


def _row_positions(
    candidates: pd.DataFrame, rocks: Sequence[str]
) -> np.ndarray:
    """Return the y position of each candidate: rock row, mode, and jitter.

    The jitter is a fixed function of the start index, so the figure is the
    same in each run. Time and memory: O(n).
    """
    rock_index = candidates["rock"].map({r: i for i, r in enumerate(rocks)})
    mode_sign = np.where(candidates["mode"] == MODES[0], -1.0, 1.0)
    spread = candidates["start_index"].to_numpy() * _JITTER_PRIME
    count = max(int(candidates["start_index"].max()) + 1, 1)
    jitter = ((spread % count) / count - 0.5) * _JITTER_WIDTH
    return rock_index.to_numpy() + mode_sign * _MODE_ROW_OFFSET + jitter


def _draw_candidate_dots(
    axes: matplotlib.axes.Axes, candidates: pd.DataFrame, rows: np.ndarray
) -> None:
    """Draw hollow dots for negative and filled dots for other candidates."""
    for color, mode in zip(_SERIES_COLORS, MODES, strict=False):
        in_mode = (candidates["mode"] == mode).to_numpy()
        negative = candidates["negative"].to_numpy(dtype=bool)
        for selection, face in ((negative, "none"), (~negative, color)):
            chosen = in_mode & selection
            axes.scatter(
                candidates.loc[chosen, "predicted_relative_loss"],
                rows[chosen],
                s=_DOT_AREA if face == "none" else _MARKER_AREA,
                facecolors=face,
                edgecolors=color if face == "none" else _INK_PRIMARY,
                linewidths=_HAIRLINE,
                zorder=3,
            )


def _candidate_legend() -> list[matplotlib.lines.Line2D]:
    """Return the legend keys of the candidate figure."""
    keys = [
        matplotlib.lines.Line2D(
            [], [], ls="", marker="s", color=color, label=mode.capitalize()
        )
        for color, mode in zip(_SERIES_COLORS, MODES, strict=False)
    ]
    marker = {"ls": "", "marker": "o", "color": _INK_MUTED}
    keys += [
        matplotlib.lines.Line2D(
            [], [], mfc="none", label="Negative (H2 gain)", **marker
        ),
        matplotlib.lines.Line2D([], [], label="Not negative", **marker),
        matplotlib.lines.Line2D(
            [],
            [],
            ls="",
            marker="D",
            color=_INK_PRIMARY,
            label="Best real run (PHREEQC)",
        ),
    ]
    return keys


def _label_usable(
    axes: matplotlib.axes.Axes, candidates: pd.DataFrame, rows: np.ndarray
) -> None:
    """Write the predicted loss next to each usable candidate."""
    usable = candidates["usable"].to_numpy(dtype=bool)
    for value, row in zip(
        candidates.loc[usable, "predicted_relative_loss"],
        rows[usable],
        strict=True,
    ):
        axes.annotate(
            f"usable: {value:.4f}",
            (value, row),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=_SMALL_FONT,
            color=_INK_PRIMARY,
        )


def _negative_region(axes: matplotlib.axes.Axes) -> None:
    """Shade the loss values below 0 and label them as not physical."""
    left, right = axes.get_xlim()
    axes.axvspan(left, 0, color=_CRITICAL, alpha=_FAIL_WASH_ALPHA, lw=0)
    axes.axvline(0, color=_INK_SECONDARY, linewidth=_HAIRLINE)
    transform = matplotlib.transforms.blended_transform_factory(
        axes.transData, axes.transAxes
    )
    axes.text(
        left,
        1.01,
        "Predicted H2 gain (not physical)",
        transform=transform,
        va="bottom",
        fontsize=_SMALL_FONT,
        color=_INK_SECONDARY,
    )
    axes.set_xlim(left, right)


def _draw_best_real(axes: matplotlib.axes.Axes, best_real: pd.Series) -> None:
    """Draw a diamond at the best real (PHREEQC) loss of each rock row."""
    axes.scatter(
        best_real,
        np.arange(len(best_real)),
        marker="D",
        s=_DOT_AREA,
        color=_INK_PRIMARY,
        zorder=4,
    )


def _candidate_title(candidates: pd.DataFrame) -> str:
    """Return the title with the negative and the usable counts."""
    negative = int(candidates["negative"].astype(bool).sum())
    usable = int(candidates["usable"].astype(bool).sum())
    return (
        f"Final candidates: {negative} of {len(candidates)} predict an H2 "
        f"gain, {usable} usable"
    )


@_style
def plot_candidates(
    all_candidates: pd.DataFrame,
    baseline: pd.DataFrame,
    rocks: Sequence[str],
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Draw the predicted relative loss of each final candidate by rock.

    Args:
        all_candidates: The rows of all_candidates.csv.
        baseline: The rows of baseline.csv. It gives the best real loss.
        rocks: The rocks, in the display order (top to bottom).
        output_path: The PNG file to write.

    Returns:
        The path of the written PNG file.

    Raises:
        ValueError: If a column or a baseline row of a rock is missing.
    """
    _require_columns(all_candidates, _CANDIDATE_COLUMNS, "all_candidates")
    _require_columns(baseline, ("rock", "real_best_real"), "baseline")
    candidates = all_candidates[all_candidates["rock"].isin(rocks)]
    rows = _row_positions(candidates, rocks)
    best_real = _by_rock(baseline, rocks, "baseline")["real_best_real"]
    figure, axes = _new_figure(len(rocks) * _ROCK_ROW_HEIGHT * 1.3 + 1.6)
    plot = axes[0, 0]
    _draw_candidate_dots(plot, candidates, rows)
    _draw_best_real(plot, best_real)
    _label_usable(plot, candidates, rows)
    _loss_axis(plot, "x", candidates["predicted_relative_loss"].min())
    _negative_region(plot)
    _rock_axis(plot, rocks)
    _style_axes(plot, "x")
    plot.set_xlabel("Predicted relative H2 loss (symlog, lower is better)")
    figure.suptitle(_candidate_title(candidates))
    figure.legend(
        handles=_candidate_legend(), loc="outside lower center", ncols=5
    )
    return _save(figure, output_path)


def _draw_role_bars(axes: matplotlib.axes.Axes, counts: pd.DataFrame) -> None:
    """Draw stacked role bars with the count inside each segment."""
    left = np.zeros(len(counts))
    positions = np.arange(len(counts))
    for color, role in zip(_SERIES_COLORS, ROLES, strict=True):
        widths = counts[role].to_numpy()
        axes.barh(
            positions,
            widths,
            left=left,
            height=0.6,
            color=color,
            edgecolor=_BACKGROUND,
            linewidth=1.5,
            label=role.capitalize(),
        )
        for position, start, width in zip(positions, left, widths, strict=True):
            if width > 0:
                axes.text(
                    start + width / 2,
                    position,
                    str(width),
                    ha="center",
                    va="center",
                    color=_INSIDE_LABEL,
                    fontsize=_SMALL_FONT,
                )
        left += widths


@_style
def plot_input_roles(
    input_space: pd.DataFrame,
    rocks: Sequence[str],
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Draw the free, tied, and fixed input counts of each rock.

    Args:
        input_space: The rows of input_space.csv.
        rocks: The rocks, in the display order (top to bottom).
        output_path: The PNG file to write.

    Returns:
        The path of the written PNG file.

    Raises:
        ValueError: If a column is missing or a role is unknown.
    """
    _require_columns(input_space, ("rock", "input", "role"), "input_space")
    unknown = sorted(set(input_space["role"]) - set(ROLES))
    if unknown:
        raise ValueError(f"input_space has unknown roles {unknown}.")
    counts = pd.crosstab(input_space["rock"], input_space["role"]).reindex(
        index=list(rocks), columns=list(ROLES), fill_value=0
    )
    totals = sorted(set(counts.sum(axis=1)))
    figure, axes = _new_figure(len(rocks) * _ROCK_ROW_HEIGHT * 0.8 + 1.2)
    plot = axes[0, 0]
    _draw_role_bars(plot, counts)
    plot.set_xlim(0, max(totals))
    plot.set_xlabel("Number of inputs")
    _rock_axis(plot, rocks)
    _style_axes(plot, "x")
    total_text = " or ".join(str(total) for total in totals)
    figure.suptitle(f"Input roles of each rock ({total_text} inputs)")
    figure.legend(loc="outside lower center", ncols=len(ROLES))
    return _save(figure, output_path)


def _mark_window(
    axes: matplotlib.axes.Axes,
    curve: pd.DataFrame,
    window_days: tuple[float, float],
    color: str,
) -> float:
    """Mark H2 at the window start and end and return the relative loss."""
    times = curve["time_days"].to_numpy()
    amounts = curve["h2_mol"].to_numpy()
    start, end = np.interp(window_days, times, amounts)
    # The start label goes right of its dot and the end label goes left,
    # so that no label crosses the curve or the plot edge.
    placements = (("left", _LABEL_OFFSET), ("right", -_LABEL_OFFSET))
    for day, amount, (side, offset) in zip(
        window_days, (start, end), placements, strict=True
    ):
        axes.scatter(
            day,
            amount,
            s=_MARKER_AREA,
            color=color,
            zorder=3,
            edgecolors=_BACKGROUND,
        )
        axes.annotate(
            f"H2({day:g} d) = {amount:.4f} mol",
            (day, amount),
            xytext=(offset, -_LABEL_OFFSET),
            textcoords="offset points",
            ha=side,
            va="top",
            fontsize=_SMALL_FONT,
            color=_INK_PRIMARY,
        )
    return float((start - end) / start)


def _empty_note(axes: matplotlib.axes.Axes) -> None:
    """Write a note into an empty plot."""
    axes.text(
        0.5,
        0.5,
        "No usable candidate in this run.",
        transform=axes.transAxes,
        ha="center",
        color=_INK_SECONDARY,
    )
    axes.set_axis_off()


def _draw_trajectories(
    axes: matplotlib.axes.Axes,
    groups: list[tuple[tuple[str, str, float], pd.DataFrame]],
    window_days: tuple[float, float],
) -> list[str]:
    """Draw one H2 curve for each candidate and return the title lines."""
    titles = []
    for color, ((rock, mode, rank), group) in zip(
        _SERIES_COLORS, groups, strict=False
    ):
        curve = group.sort_values("time_days")
        label = f"{rock.replace('_', ' ')}, {mode}, rank {int(rank)}"
        axes.plot(
            curve["time_days"],
            curve["h2_mol"],
            color=color,
            lw=_LINE_WIDTH,
            label=label,
        )
        loss = _mark_window(axes, curve, window_days, color)
        titles.append(f"{label}: relative loss {loss:.4f}")
    return titles


@_style
def plot_usable_candidate_h2(
    h2_trajectories: pd.DataFrame,
    window_days: tuple[float, float],
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Draw the predicted H2 amount over time of the usable candidates.

    The figure shows the first three candidates in rock, mode, and rank
    order. An empty table gives a note.

    Args:
        h2_trajectories: The rows of h2_trajectories.csv (mol, days).
        window_days: The start and end time of the objective, in days.
        output_path: The PNG file to write.

    Returns:
        The path of the written PNG file.

    Raises:
        ValueError: If a column is missing.
    """
    keys = ["rock", "mode", "rank"]
    _require_columns(h2_trajectories, (*keys, "time_days", "h2_mol"), "h2")
    figure, axes = _new_figure(3.6)
    plot = axes[0, 0]
    groups = list(h2_trajectories.groupby(keys, sort=True))
    if not groups:
        _empty_note(plot)
        return _save(figure, output_path)
    titles = _draw_trajectories(plot, groups[:_MAX_TRAJECTORIES], window_days)
    plot.set_ylim(0, None)
    plot.set_xlabel("Time (days)")
    plot.set_ylabel("Predicted H2 gas (mol)")
    _style_axes(plot, "y")
    if len(titles) > 1:
        plot.legend(loc="lower right")
    figure.suptitle(
        "Predicted H2 of the usable candidate\n" + "\n".join(titles)
    )
    return _save(figure, output_path)


README_TEXT = """\
# Figures

NOTE: All values are surrogate predictions, except the best real runs and
the real share in figure 02. Run PHREEQC on a candidate before you use it.

- `01_surrogate_gate.png`: the test Spearman correlation and the test
  MAE / std of each rock, for the relative and the absolute loss. The red
  wash is the fail side of each gate limit.
- `02_predicted_negative_share.png`: the share of test runs with a
  predicted H2 gain (relative loss below 0). No real run has an H2 gain.
- `03_search_convergence.png`: the lowest loss of the starts at each search
  step, one panel for each rock. The y-axis is symmetric log (linear in
  [-0.01, 0.01]), because the unbounded mode goes below -100.
- `04_candidates_vs_starts.png`: the predicted relative loss of each final
  candidate. A hollow dot is a negative (not physical) candidate. The
  diamond is the best real run of the rock. The x-axis is symmetric log.
- `05_input_roles.png`: the number of free, tied, and fixed inputs of each
  rock. The search changes only the free inputs.
- `06_usable_candidate_h2.png`: the predicted H2 amount over time of the
  usable candidates. The dots mark the start and the end of the objective.
"""


def write_readme(output_dir: pathlib.Path) -> pathlib.Path:
    """Write README.md that explains each figure.

    Args:
        output_dir: The figures folder.

    Returns:
        The path of the written README.md file.
    """
    path = pathlib.Path(output_dir) / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(README_TEXT)
    return path


def read_tables(results_dir: pathlib.Path) -> dict[str, pd.DataFrame]:
    """Read the CSV files of a results folder.

    Args:
        results_dir: The results folder of one experiment.

    Returns:
        One DataFrame for each key of CSV_FILE_NAMES.

    Raises:
        FileNotFoundError: If a CSV file is missing.
    """
    tables = {}
    for key, name in CSV_FILE_NAMES.items():
        path = pathlib.Path(results_dir) / name
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} does not exist. The figures need all of "
                f"{sorted(CSV_FILE_NAMES.values())}."
            )
        tables[key] = pd.read_csv(path)
    return tables


def write_all_figures(
    results_dir: pathlib.Path, output_dir: pathlib.Path
) -> list[pathlib.Path]:
    """Read a results folder and write all figures and their README.

    Facade pattern: one call reads the config and the CSV files and runs
    each figure function.

    Args:
        results_dir: The results folder. It must contain config.yaml and
            the files in CSV_FILE_NAMES.
        output_dir: The folder for the PNG files and README.md.

    Returns:
        The paths of the PNG files, in the order of FIGURE_FILE_NAMES.

    Raises:
        FileNotFoundError: If config.yaml or a CSV file is missing.
        ValueError: If a file has missing columns or rows.
    """
    config_path = pathlib.Path(results_dir) / CONFIG_FILE_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path} does not exist.")
    config = experiment_config.load_config(config_path)
    tables = read_tables(results_dir)
    paths = [pathlib.Path(output_dir) / name for name in FIGURE_FILE_NAMES]
    rocks = config.rocks
    window = (config.target.start_days, config.target.end_days)
    check, baseline = tables["surrogate_check"], tables["baseline"]
    written = [
        plot_surrogate_gate(check, rocks, config.acceptance, paths[0]),
        plot_negative_share(check, baseline, rocks, paths[1]),
        plot_search_convergence(tables["search_history"], rocks, paths[2]),
        plot_candidates(tables["all_candidates"], baseline, rocks, paths[3]),
        plot_input_roles(tables["input_space"], rocks, paths[4]),
        plot_usable_candidate_h2(tables["h2_trajectories"], window, paths[5]),
    ]
    write_readme(output_dir)
    return written


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=pathlib.Path, required=True)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help=f"The figures folder. Default: <results>/{FIGURES_DIR_NAME}.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Write the figures of one results folder.

    Args:
        argv: The command line arguments. None uses sys.argv.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    output_dir = args.output or args.results / FIGURES_DIR_NAME
    for path in write_all_figures(args.results, output_dir):
        _LOGGER.info("Wrote %s", path)


if __name__ == "__main__":
    main()
