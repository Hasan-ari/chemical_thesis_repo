from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from conditional_model_v1.data import (
    CONDITION_FEATURES,
    OUTPUT_FEATURES,
    DatasetSpec,
    load_input_parameters,
    run_id_from_path,
)

ROCK_ORDER: tuple[str, ...] = ("Calcite", "Dolomite", "Halite", "Trona")
ROCK_STYLES: dict[str, tuple[str, str]] = {
    "Calcite": ("#2563EB", "-"),
    "Dolomite": ("#D97706", "--"),
    "Halite": ("#65752A", "-."),
    "Trona": ("#BE5678", ":"),
}
DECREASE_COLOR = "#D95F59"
UNCHANGED_COLOR = "#CBD5E1"
INCREASE_COLOR = "#2A9D8F"


@dataclass(frozen=True)
class DatasetLayout:
    rock: str
    dataset_name: str
    dataset_root: Path


@dataclass(frozen=True)
class RockSamples:
    rock: str
    dataset_name: str
    run_ids: tuple[str, ...]
    conditions: np.ndarray
    outputs: np.ndarray
    time_values: np.ndarray


@dataclass(frozen=True)
class ReportArtifacts:
    output_png_paths: tuple[Path, ...]
    condition_png_path: Path

    @property
    def png_paths(self) -> tuple[Path, ...]:
        return (*self.output_png_paths, self.condition_png_path)


def default_dataset_layouts(data_root: Path | str) -> tuple[DatasetLayout, ...]:
    """Return the exact four local datasets used by the shared-model experiment."""
    data_root = Path(data_root)
    return (
        DatasetLayout(
            rock="Calcite",
            dataset_name="Calcite_wat_sat_data_3",
            dataset_root=data_root / "wat_sat" / "Calcite" / "Calcite_wat_sat_data_3",
        ),
        DatasetLayout(
            rock="Dolomite",
            dataset_name="Dolomite_wat_sat_data_2",
            dataset_root=data_root / "wat_sat" / "Dolomite" / "Dolomite_wat_sat_data_2",
        ),
        DatasetLayout(
            rock="Halite",
            dataset_name="Halite_wat_sat_data_2",
            dataset_root=data_root / "wat_sat" / "Halite" / "Halite_wat_sat_data_2",
        ),
        DatasetLayout(
            rock="Trona",
            dataset_name="Trona_par_sat_data_3",
            dataset_root=data_root / "wat_sat" / "Trona" / "Trona_par_sat_data_3",
        ),
    )


def load_rock_samples(
    layout: DatasetLayout,
    *,
    progress_every: int = 1000,
) -> RockSamples:
    """Read every matched run, but only output rows at time indices 0 and 1."""
    input_dir = layout.dataset_root / "input"
    output_dir = layout.dataset_root / "output"
    if not input_dir.is_dir() or not output_dir.is_dir():
        raise FileNotFoundError(
            f"Expected input/ and output/ directories under {layout.dataset_root}"
        )

    input_paths = _index_run_paths(input_dir, "*_Input.txt")
    output_paths = _index_run_paths(output_dir, "*_Output.txt")
    input_ids = set(input_paths)
    output_ids = set(output_paths)
    if input_ids != output_ids:
        missing_inputs = sorted(output_ids - input_ids, key=_run_sort_key)
        missing_outputs = sorted(input_ids - output_ids, key=_run_sort_key)
        raise ValueError(
            f"{layout.dataset_name} input/output run ids do not match: "
            f"missing_inputs={len(missing_inputs)} missing_outputs={len(missing_outputs)}"
        )
    if not input_ids:
        raise ValueError(f"{layout.dataset_name} has no matched input/output runs")

    ordered_ids = sorted(input_ids, key=_run_sort_key)
    n_runs = len(ordered_ids)
    conditions = np.empty((n_runs, len(CONDITION_FEATURES)), dtype=np.float64)
    outputs = np.empty((n_runs, 2, len(OUTPUT_FEATURES)), dtype=np.float64)
    qualified_run_ids: list[str] = []
    reference_times: np.ndarray | None = None
    spec = DatasetSpec(
        name=layout.dataset_name,
        rock=layout.rock,
        path=layout.dataset_root,
    )

    for run_index, run_id in enumerate(ordered_ids):
        condition_row = load_input_parameters(input_paths[run_id], spec)
        condition_values = np.asarray(
            [condition_row[feature] for feature in CONDITION_FEATURES],
            dtype=np.float64,
        )
        time_values, output_values = _read_initial_output_pair(output_paths[run_id])
        if reference_times is None:
            reference_times = time_values
        elif not np.allclose(time_values, reference_times, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{output_paths[run_id]} time indices 0 and 1 do not match "
                f"the dataset reference {reference_times.tolist()}"
            )

        conditions[run_index] = condition_values
        outputs[run_index] = output_values
        qualified_run_ids.append(f"{layout.dataset_name}:{run_id}")
        if progress_every > 0 and (run_index + 1) % progress_every == 0:
            print(
                f"loaded_runs={run_index + 1}/{n_runs} rock={layout.rock}",
                flush=True,
            )

    if reference_times is None:  # Defensive; the empty case is rejected above.
        raise RuntimeError(f"Could not read timestamps for {layout.dataset_name}")
    print(
        f"loaded rock={layout.rock} runs={n_runs} "
        f"time_index_0={reference_times[0]:.12g} "
        f"time_index_1={reference_times[1]:.12g}",
        flush=True,
    )
    return RockSamples(
        rock=layout.rock,
        dataset_name=layout.dataset_name,
        run_ids=tuple(qualified_run_ids),
        conditions=conditions,
        outputs=outputs,
        time_values=reference_times,
    )


def load_all_rocks(data_root: Path | str) -> tuple[RockSamples, ...]:
    """Load the four exact local datasets in stable rock order."""
    return tuple(
        load_rock_samples(layout) for layout in default_dataset_layouts(data_root)
    )


def generate_report(
    samples_by_rock: Sequence[RockSamples],
    output_dir: Path | str,
) -> ReportArtifacts:
    """Write four paired-output visuals and one shared-condition visual."""
    samples = tuple(samples_by_rock)
    _validate_report_samples(samples)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_paths: list[Path] = []
    for rock_samples in samples:
        path = output_dir / f"{rock_samples.rock.lower()}_t0_t1_output_transition.png"
        _plot_output_transition(rock_samples, path)
        png_paths.append(path)

    condition_path = output_dir / "condition_distributions_all_rocks.png"
    _plot_condition_distributions(samples, condition_path)
    png_paths.append(condition_path)

    return ReportArtifacts(
        output_png_paths=tuple(png_paths[:-1]),
        condition_png_path=condition_path,
    )


def _read_initial_output_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    expected_header = ("time_d", *OUTPUT_FEATURES)
    with path.open(encoding="utf-8") as file_obj:
        header = tuple(file_obj.readline().split())
        row_0 = np.fromstring(file_obj.readline(), sep=" ", dtype=np.float64)
        row_1 = np.fromstring(file_obj.readline(), sep=" ", dtype=np.float64)

    if header != expected_header:
        raise ValueError(f"{path} output header does not match the 32-feature contract")
    expected_columns = len(expected_header)
    if row_0.size != expected_columns or row_1.size != expected_columns:
        raise ValueError(f"{path} must contain output rows at time indices 0 and 1")
    values = np.stack((row_0, row_1))
    if not np.isfinite(values).all():
        raise ValueError(f"{path} first two output rows must be finite")
    time_values = values[:, 0]
    if not time_values[1] > time_values[0]:
        raise ValueError(f"{path} first two timestamps must be strictly increasing")
    return time_values, values[:, 1:]


def _plot_output_transition(samples: RockSamples, path: Path) -> None:
    fig, (bar_axis, value_axis) = plt.subplots(
        1,
        2,
        figsize=(20, 20),
        layout="constrained",
        sharey=True,
        gridspec_kw={"width_ratios": (2.8, 3.6)},
    )
    y_positions = np.arange(len(OUTPUT_FEATURES))
    for feature_index, feature in enumerate(OUTPUT_FEATURES):
        at_t0 = samples.outputs[:, 0, feature_index]
        at_t1 = samples.outputs[:, 1, feature_index]
        decreased_count, unchanged_count, increased_count = _change_counts(
            at_t0,
            at_t1,
        )
        n_runs = len(at_t0)
        decreased = decreased_count / n_runs
        unchanged = unchanged_count / n_runs
        increased = increased_count / n_runs
        y = y_positions[feature_index]

        bar_axis.barh(y, decreased, color=DECREASE_COLOR, height=0.72)
        bar_axis.barh(
            y,
            unchanged,
            left=decreased,
            color=UNCHANGED_COLOR,
            height=0.72,
        )
        bar_axis.barh(
            y,
            increased,
            left=decreased + unchanged,
            color=INCREASE_COLOR,
            height=0.72,
        )
        for width, left in (
            (decreased, 0.0),
            (unchanged, decreased),
            (increased, decreased + unchanged),
        ):
            if width >= 0.08:
                bar_axis.text(
                    left + width / 2,
                    y,
                    f"{width:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#1F2937",
                )
        value_axis.text(
            0.02,
            y,
            f"{_format_distribution(at_t0)}  →  {_format_distribution(at_t1)}\n"
            f"↓ {decreased_count:,}     = {unchanged_count:,}     "
            f"↑ {increased_count:,}",
            ha="left",
            va="center",
            fontsize=8.5,
        )
        if feature_index % 2:
            for axis in (bar_axis, value_axis):
                axis.axhspan(y - 0.5, y + 0.5, color="#F4F6F8", zorder=-2)

    bar_axis.set_yticks(y_positions, OUTPUT_FEATURES)
    bar_axis.invert_yaxis()
    bar_axis.set_xlim(0.0, 1.0)
    bar_axis.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0), ("0%", "25%", "50%", "75%", "100%"))
    bar_axis.set_xlabel("share of runs")
    bar_axis.set_title("Direction of change", fontsize=11)
    bar_axis.grid(axis="x", color="#D9DEE7", linewidth=0.6)
    value_axis.set_xlim(0.0, 1.0)
    value_axis.axis("off")
    value_axis.set_title(
        "index 0 median [P5–P95]  →  index 1 median [P5–P95]\n"
        "run counts: ↓ decreased   = unchanged   ↑ increased",
        fontsize=11,
    )

    legend_handles = [
        Line2D([0], [0], color=DECREASE_COLOR, linewidth=8, label="decreased"),
        Line2D([0], [0], color=UNCHANGED_COLOR, linewidth=8, label="unchanged"),
        Line2D([0], [0], color=INCREASE_COLOR, linewidth=8, label="increased"),
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncols=3, frameon=False)

    fig.suptitle(
        f"{samples.rock}: what changed from index 0 to index 1?\n"
        f"n={len(samples.run_ids):,} matched runs; "
        f"time_d={samples.time_values[0]:.12g} → {samples.time_values[1]:.12g}; "
        "P5–P95 contains the central 90% of runs",
        fontsize=13,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_condition_distributions(
    samples_by_rock: tuple[RockSamples, ...],
    path: Path,
) -> None:
    columns = 5
    rows = 4
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 4.0, rows * 3.2),
        layout="constrained",
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for feature_index, feature in enumerate(CONDITION_FEATURES):
        axis = flat_axes[feature_index]
        for rock_index, samples in enumerate(samples_by_rock):
            values = samples.conditions[:, feature_index]
            low, q1, median, q3, high = np.quantile(
                values,
                (0.05, 0.25, 0.5, 0.75, 0.95),
            )
            color = ROCK_STYLES[samples.rock][0]
            axis.hlines(rock_index, low, high, color=color, linewidth=1.5)
            axis.hlines(rock_index, q1, q3, color=color, linewidth=6, alpha=0.45)
            axis.plot(median, rock_index, marker="o", color=color, markersize=4)
        axis.set_title(feature, fontsize=9)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.set_yticks(range(len(samples_by_rock)), ("Cal", "Dol", "Hal", "Tro"))
        axis.invert_yaxis()
        axis.tick_params(labelsize=7)
        axis.grid(axis="x", color="#D9DEE7", linewidth=0.5)

    legend_axis = flat_axes[-1]
    legend_axis.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            color=ROCK_STYLES[samples.rock][0],
            linestyle=ROCK_STYLES[samples.rock][1],
            linewidth=2,
        )
        for samples in samples_by_rock
    ]
    labels = [f"{samples.rock} (n={len(samples.run_ids):,})" for samples in samples_by_rock]
    legend_axis.legend(handles, labels, loc="center", frameon=False, fontsize=10)
    legend_axis.text(
        0.5,
        0.15,
        "Thin: 5th–95th percentile\nThick: middle 50%\nDot: median",
        ha="center",
        va="center",
        fontsize=9,
        color="#30343B",
        transform=legend_axis.transAxes,
    )
    fig.suptitle(
        "Input-condition summary across all four rocks\n"
        "Intervals show where most runs lie; rock is grouping metadata only",
        fontsize=13,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _validate_report_samples(samples: tuple[RockSamples, ...]) -> None:
    if tuple(sample.rock for sample in samples) != ROCK_ORDER:
        raise ValueError(f"Expected rocks in order {ROCK_ORDER}")
    reference_times: np.ndarray | None = None
    seen_run_ids: set[str] = set()
    for sample in samples:
        n_runs = len(sample.run_ids)
        if n_runs == 0:
            raise ValueError(f"{sample.rock} has no runs")
        if sample.conditions.shape != (n_runs, len(CONDITION_FEATURES)):
            raise ValueError(f"{sample.rock} condition shape is invalid")
        if sample.outputs.shape != (n_runs, 2, len(OUTPUT_FEATURES)):
            raise ValueError(f"{sample.rock} output shape is invalid")
        if sample.time_values.shape != (2,):
            raise ValueError(f"{sample.rock} must contain exactly two time values")
        if not np.isfinite(sample.conditions).all() or not np.isfinite(sample.outputs).all():
            raise ValueError(f"{sample.rock} samples must be finite")
        if len(set(sample.run_ids)) != n_runs or seen_run_ids.intersection(sample.run_ids):
            raise ValueError("Run ids must be unique within and across rocks")
        seen_run_ids.update(sample.run_ids)
        if reference_times is None:
            reference_times = sample.time_values
        elif not np.allclose(sample.time_values, reference_times, rtol=0.0, atol=1e-12):
            raise ValueError("All rocks must share time values at indices 0 and 1")


def _index_run_paths(directory: Path, pattern: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in directory.glob(pattern):
        run_id = run_id_from_path(path)
        if run_id in indexed:
            raise ValueError(f"Duplicate run id {run_id} in {directory}")
        indexed[run_id] = path
    return indexed


def _run_sort_key(run_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(run_id))
    except ValueError:
        return (1, run_id)


def _change_counts(
    at_t0: np.ndarray,
    at_t1: np.ndarray,
) -> tuple[int, int, int]:
    delta = at_t1 - at_t0
    scale = np.maximum(np.maximum(np.abs(at_t0), np.abs(at_t1)), 1.0)
    tolerance = np.finfo(np.float64).eps * scale * 32
    decreased = int(np.count_nonzero(delta < -tolerance))
    increased = int(np.count_nonzero(delta > tolerance))
    unchanged = len(delta) - decreased - increased
    return decreased, unchanged, increased


def _format_distribution(values: np.ndarray) -> str:
    if np.all(values == values[0]):
        return f"fixed {_format_number(float(values[0]))}"
    low, median, high = np.quantile(values, (0.05, 0.5, 0.95))
    return (
        f"{_format_number(float(median))} "
        f"[{_format_number(float(low))}–{_format_number(float(high))}]"
    )


def _format_number(value: float) -> str:
    return f"{value:.3g}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot first-two-timestamp output transitions and four-rock input "
            "condition distributions from local raw TXT datasets"
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Repository data root containing wat_sat/ (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination for five PNG files",
    )
    args = parser.parse_args(argv)

    samples = load_all_rocks(args.data_root)
    artifacts = generate_report(samples, args.output_dir)
    for path in artifacts.png_paths:
        print(f"wrote={path}", flush=True)


if __name__ == "__main__":
    main()
