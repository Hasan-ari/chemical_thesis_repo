r"""Run the input optimization experiment from one config file.

Usage, from the repository root:

    PYTHONPATH=. env312/bin/python -m \
        conditional_input_optimization.optimization.experiment \
        --config conditional_input_optimization/configs/optimization_v1.yaml

Pattern: Facade. run_experiment hides the data loading, the split, the
surrogate checks, and the search of each rock behind one call.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import logging
import pathlib
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from conditional_input_optimization.optimization import (
    diagnostics,
    experiment_config,
    objectives,
    predict,
    report,
    rock_optimizer,
)
from conditional_input_optimization.optimization import (
    surrogate as surrogate_lib,
)
from conditional_input_optimization.run_data import (
    rock_source,
    run_table,
    run_table_cache,
)
from conditional_model_v1 import splitting

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
CHECKED_SPLITS = (SPLIT_TEST, SPLIT_TRAIN)
_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, eq=False)
class ExperimentResult:
    """The in-memory result of one experiment.

    Attributes:
        config: The config that the run used (after CLI overrides).
        repo_root: The folder that the config paths start from.
        rock_results: One result for each config rock, in config order.
        surrogate_checks: One row for each rock, objective, and checked
            split.
        input_units: The input file unit of each rock.
        time_days: The time axis, shape (T,), in days.
        scaler_gap: The value of diagnostics.scaler_mean_gap.
        started_at: The local start time.
        run_seconds: The wall time of the run, in seconds.
    """

    config: experiment_config.ExperimentConfig
    repo_root: pathlib.Path
    rock_results: tuple[rock_optimizer.RockResult, ...]
    surrogate_checks: pd.DataFrame
    input_units: dict[str, str]
    time_days: np.ndarray
    scaler_gap: float
    started_at: datetime.datetime
    run_seconds: float


def split_labels(
    rocks: np.ndarray, split: experiment_config.SplitConfig
) -> np.ndarray:
    """Rebuild the v2 rock-aware split. Return the split name of each run.

    The split must use all runs of the table, because the random draw of
    one rock depends on the rocks before it.

    Args:
        rocks: The rock label of each run, shape (N,).
        split: The split ratios and the seed.

    Returns:
        "train", "val", or "test" for each run, shape (N,).
    """
    parts = splitting.rock_aware_split(
        rocks=rocks,
        train_ratio=split.train,
        val_ratio=split.val,
        test_ratio=split.test,
        seed=split.seed,
    )
    labels = np.full(rocks.shape[0], SPLIT_TRAIN, dtype=object)
    labels[parts.val] = SPLIT_VAL
    labels[parts.test] = SPLIT_TEST
    return labels


def _check_rocks(selected: Sequence[str], available: np.ndarray) -> None:
    missing = sorted(set(selected) - set(np.unique(available).tolist()))
    if missing:
        raise ValueError(f"The run table has no runs of the rocks {missing}.")


def _load_checked_surrogate(
    config: experiment_config.ExperimentConfig,
    repo_root: pathlib.Path,
    table: run_table.RunTable,
    train_rows: np.ndarray,
) -> tuple[surrogate_lib.FrozenSurrogate, float]:
    """Load the surrogate. Check that its scaler saw the rebuilt train set."""
    model = config.model
    surrogate = surrogate_lib.load_surrogate(
        repo_root / model.checkpoint,
        repo_root / model.preprocessors,
        surrogate_lib.ModelShape(model.hidden_size, model.num_layers),
        table.time_days,
        torch.device(model.device),
    )
    gap = diagnostics.scaler_mean_gap(
        table.conditions[train_rows], surrogate.scaling
    )
    if gap > config.split.scaler_mean_tolerance:
        raise ValueError(
            f"The rebuilt train split does not match the v2 scaler: gap "
            f"{gap:.3g} > {config.split.scaler_mean_tolerance}."
        )
    return surrogate, gap


def _rock_data(
    table: run_table.RunTable,
    rock: str,
    labels: np.ndarray,
    surrogate: surrogate_lib.FrozenSurrogate,
    config: experiment_config.ExperimentConfig,
) -> tuple[rock_optimizer.RockData, np.ndarray]:
    """Return the data of one rock and the split label of each of its runs."""
    rows = np.flatnonzero(table.rocks == rock)
    conditions = np.asarray(table.conditions[rows])
    output_index = run_table.OUTPUT_FEATURES.index(config.target.output)
    data = rock_optimizer.RockData(
        rock=rock,
        run_ids=np.asarray(table.run_ids[rows]),
        conditions=conditions,
        real_h2=np.asarray(table.output(config.target.output)[rows]),
        predicted_h2=predict.predict_output(
            surrogate,
            conditions,
            output_index=output_index,
            batch_size=config.report.batch_size,
        ),
        is_train=labels[rows] == SPLIT_TRAIN,
    )
    return data, labels[rows]


def check_rock(
    data: rock_optimizer.RockData,
    labels: np.ndarray,
    config: experiment_config.ExperimentConfig,
    time_days: np.ndarray,
) -> list[dict[str, object]]:
    """Compare the predicted and the real objective of the eligible runs.

    Args:
        data: The runs of one rock.
        labels: The split label of each run, shape (n,).
        config: The experiment config.
        time_days: The time axis, shape (T,), in days.

    Returns:
        One row for each objective in objectives.OBJECTIVE_NAMES and each
        split in CHECKED_SPLITS (test first). All rows use the same
        eligible runs and the same acceptance rule. See
        diagnostics.surrogate_check for the metrics.
    """
    target = config.target
    window = objectives.TimeWindow.from_days(
        time_days, target.start_days, target.end_days
    )
    eligible = data.real_h2[:, window.start_index] >= target.min_start_mol
    rows = []
    for name in objectives.OBJECTIVE_NAMES:
        objective = objectives.build_objective(name, window, target.eps_mol)
        real = objective(data.real_h2)
        predicted = objective(data.predicted_h2)
        for split_name in CHECKED_SPLITS:
            mask = eligible & (labels == split_name)
            metrics = diagnostics.surrogate_check(
                predicted[mask],
                real[mask],
                n_top=config.report.top_k,
                acceptance=config.acceptance,
            )
            rows.append(
                {"rock": data.rock, "objective": name, "split": split_name}
                | metrics
            )
    return rows


def _search_test_check(
    rows: Sequence[dict[str, object]], objective: str
) -> dict[str, object]:
    """Return the test row of the search objective. Rows from check_rock."""
    (row,) = (
        row
        for row in rows
        if row["objective"] == objective and row["split"] == SPLIT_TEST
    )
    return row


def run_experiment(
    config: experiment_config.ExperimentConfig, repo_root: pathlib.Path
) -> ExperimentResult:
    """Run the complete experiment in memory.

    Pattern: Facade.

    Args:
        config: The experiment config.
        repo_root: The folder that the config paths start from.

    Returns:
        The result. Call report.write_results to save it.

    Raises:
        ValueError: If a config rock has no runs or the split check fails.
    """
    started_at = datetime.datetime.now().astimezone()
    started = time.perf_counter()
    sources = rock_source.load_rock_sources(
        repo_root / config.data.rock_config, repo_root
    )
    table = run_table_cache.load_or_build_run_table(
        sources, repo_root / config.data.cache_dir
    )
    _check_rocks(config.rocks, table.rocks)
    labels = split_labels(table.rocks, config.split)
    surrogate, gap = _load_checked_surrogate(
        config, repo_root, table, labels == SPLIT_TRAIN
    )
    _LOGGER.info("Loaded %d runs. Scaler mean gap: %.3g.", len(labels), gap)
    source_order = [source.name for source in sources]
    checks, results = [], []
    for rock in config.rocks:
        data, rock_labels = _rock_data(table, rock, labels, surrogate, config)
        rock_checks = check_rock(data, rock_labels, config, table.time_days)
        checks.extend(rock_checks)
        results.append(
            rock_optimizer.optimize_rock(
                data,
                surrogate,
                config,
                rock_index=source_order.index(rock),
                error_p95=float(
                    _search_test_check(rock_checks, config.search.objective)[
                        "abs_error_p95"
                    ]
                ),
            )
        )
    return ExperimentResult(
        config=config,
        repo_root=repo_root,
        rock_results=tuple(results),
        surrogate_checks=pd.DataFrame(checks),
        input_units={source.name: source.input_units for source in sources},
        time_days=np.asarray(table.time_days),
        scaler_gap=gap,
        started_at=started_at,
        run_seconds=time.perf_counter() - started,
    )


def apply_overrides(
    config: experiment_config.ExperimentConfig,
    rocks: Sequence[str] | None,
    steps: int | None,
) -> experiment_config.ExperimentConfig:
    """Return a copy of the config with the CLI overrides.

    Args:
        config: The config from the file.
        rocks: A rock subset, or None to keep the config rocks.
        steps: The number of search steps, or None to keep the config value.

    Returns:
        The new config. The constructors check the new values again.
    """
    if rocks is not None:
        config = dataclasses.replace(config, rocks=tuple(rocks))
    if steps is not None:
        search = dataclasses.replace(config.search, steps=steps)
        config = dataclasses.replace(config, search=search)
    return config


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument(
        "--rocks", nargs="+", default=None, help="A subset of the rocks."
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Override search.steps."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the experiment and write the results folder.

    Args:
        argv: The command line arguments. None uses sys.argv.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)
    repo_root = pathlib.Path.cwd()
    config = apply_overrides(
        experiment_config.load_config(args.config), args.rocks, args.steps
    )
    result = run_experiment(config, repo_root)
    folder = report.write_results(
        result, args.config, repo_root / config.results_dir
    )
    _LOGGER.info("Wrote the results to %s", folder)


if __name__ == "__main__":
    main()
