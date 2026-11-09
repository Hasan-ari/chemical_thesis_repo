"""Run the search for one rock in each mode and evaluate the candidates.

The steps for one rock:

1. Build the input space (free, tied, and fixed inputs).
2. Select the starts from the eligible real runs.
3. For each mode, run the batched search on the penalized loss.
4. Evaluate the best point of each start without a gradient: objective,
   constraints, observed range, and distance to the train runs.
5. Select the top candidates with a distance filter.

This module imports conditional_model_v1.data. Reason: it holds the names
of the 57 inputs in model column order.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from conditional_input_optimization.optimization import (
    candidates,
    constraints,
    diagnostics,
    experiment_config,
    input_space,
    objectives,
    predict,
    projections,
    search,
    starts,
)
from conditional_input_optimization.optimization import (
    surrogate as surrogate_lib,
)
from conditional_model_v1 import data as v2_data

FEATURE_NAMES = v2_data.CONDITION_FEATURES
_LOGGER = logging.getLogger(__name__)
_LIST_SEPARATOR = ";"


@dataclasses.dataclass(frozen=True, eq=False)
class RockData:
    """The real runs of one rock and the surrogate prediction for them.

    Attributes:
        rock: The rock label. It is metadata only.
        run_ids: The run ids, shape (n,).
        conditions: The inputs, shape (n, 57), solution inputs in mol/kgw.
        real_h2: The PHREEQC H2 gas amount, shape (n, T), in mol.
        predicted_h2: The surrogate H2 gas amount, shape (n, T), in mol.
        is_train: Shape (n,), bool. True for a run of the v2 train split.

    Raises:
        ValueError: If the shapes do not agree.
    """

    rock: str
    run_ids: np.ndarray
    conditions: np.ndarray
    real_h2: np.ndarray
    predicted_h2: np.ndarray
    is_train: np.ndarray

    def __post_init__(self) -> None:
        """Check the shapes."""
        n_runs = self.run_ids.shape[0]
        if self.conditions.shape != (n_runs, len(FEATURE_NAMES)):
            raise ValueError("conditions must have shape (n, 57).")
        if self.real_h2.shape != self.predicted_h2.shape:
            raise ValueError("real_h2 and predicted_h2 must have equal shapes.")
        if self.real_h2.shape[0] != n_runs or self.is_train.shape != (n_runs,):
            raise ValueError("Each array must have one row for each run.")


@dataclasses.dataclass(frozen=True, eq=False)
class ModeResult:
    """The search result of one rock in one mode.

    Attributes:
        mode: "bounded" or "unbounded".
        candidates: One row for each start. See _candidate_table.
        trajectories: The predicted H2 at the best point of each start,
            shape (B, T), in mol.
        history: The penalized loss of each start at each step, shape
            (n_steps + 1, B).
    """

    mode: str
    candidates: pd.DataFrame
    trajectories: np.ndarray
    history: np.ndarray


@dataclasses.dataclass(frozen=True, eq=False)
class RockResult:
    """The result of one rock: input space, baseline, and each mode."""

    rock: str
    space: input_space.RockInputSpace
    baseline: dict[str, object]
    modes: tuple[ModeResult, ...]


@dataclasses.dataclass(frozen=True, eq=False)
class _RockProblem:
    """The fixed parts of the search of one rock. Internal only."""

    data: RockData
    surrogate: surrogate_lib.FrozenSurrogate
    config: experiment_config.ExperimentConfig
    space: input_space.RockInputSpace
    window: objectives.TimeWindow
    objective: objectives.Objective
    h2_index: int
    control_mask: np.ndarray
    start_indices: np.ndarray
    physical: constraints.LinearConstraints
    observed: constraints.LinearConstraints
    reference: np.ndarray
    reference_nn: np.ndarray
    seed: int
    error_p95: float
    baseline_objective: float


def start_shortfall(
    h2_start: torch.Tensor, min_start_mol: float
) -> torch.Tensor:
    """Return the scaled shortfall of the predicted start H2.

    shortfall = relu(min_start_mol - h2_start) / min_start_mol. It is 0 for
    an eligible start and 1 for a start of 0 mol. The division by
    min_start_mol makes it unitless, as the scaled linear violations.

    Args:
        h2_start: The predicted H2(t_start), shape (B,), in mol.
        min_start_mol: The eligibility limit, in mol. Must be positive.

    Returns:
        The shortfall, shape (B,), unitless.

    Raises:
        ValueError: If min_start_mol is not positive.
    """
    if not min_start_mol > 0:
        raise ValueError(f"min_start_mol must be > 0, got {min_start_mol}.")
    return torch.relu(min_start_mol - h2_start) / min_start_mol


@dataclasses.dataclass(frozen=True)
class _PenalizedLoss:
    """objective + weight * (sum(violation ** 2) + start_shortfall ** 2).

    The squared penalty is 0 inside the feasible set, and its gradient is
    continuous at the border. Thus Adam does not jump at the border.
    The start shortfall is a hinge on the predicted H2(t_start). Without
    it, the search can push the predicted start H2 to 0 or below and get a
    very large negative relative loss that the real chemistry cannot give.
    """

    problem: _RockProblem
    rules: constraints.LinearConstraints

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """Return the loss of each start, shape (B,), from z (B, K)."""
        problem = self.problem
        inputs = problem.space.decode(z)
        h2 = problem.surrogate.predict(inputs)[:, :, problem.h2_index]
        shortfall = start_shortfall(
            h2[:, problem.window.start_index],
            problem.config.target.min_start_mol,
        )
        penalty = self.rules.violation(inputs).pow(2).sum(dim=1)
        penalty = penalty + shortfall.pow(2)
        weight = problem.config.constraints.penalty_weight
        return problem.objective(h2) + weight * penalty


def context_control_mask(
    space: input_space.RockInputSpace, context_inputs: Sequence[str]
) -> np.ndarray:
    """Return the control mask of the free inputs. False = context input.

    Args:
        space: The input space of the rock.
        context_inputs: Input names that stay at the start-run value.

    Returns:
        Shape (K,), bool.

    Raises:
        ValueError: If a context input is tied in this rock. A tied input
            moves with its sources, so it cannot stay fixed.
    """
    roles = space.roles()
    free = list(space.detection.free_columns)
    mask = np.ones(space.n_free, dtype=bool)
    for name in context_inputs:
        column = FEATURE_NAMES.index(name)
        if roles[column] == input_space.ROLE_TIED:
            raise ValueError(
                f"Context input {name!r} is tied in rock {space.rock!r}. "
                "Use a free or a fixed input."
            )
        if roles[column] == input_space.ROLE_FREE:
            mask[free.index(column)] = False
    return mask


def optimize_rock(
    data: RockData,
    surrogate: surrogate_lib.FrozenSurrogate,
    config: experiment_config.ExperimentConfig,
    *,
    rock_index: int,
    error_p95: float,
) -> RockResult:
    """Run the search of one rock in each config mode.

    Pattern: Facade. One call runs the full pipeline of one rock.

    Time: O(modes * steps * B * T * H^2) for the search, with B starts.

    Args:
        data: The runs of the rock.
        surrogate: The frozen surrogate.
        config: The experiment config.
        rock_index: The position of the rock in the rock source config.
            The start draw and the distance subsample use seed =
            search.seed + rock_index. Thus each rock has its own draw, and
            a run of a rock subset gives the same draw.
        error_p95: The 95th percentile of the absolute objective error on
            the eligible test runs of the rock.

    Returns:
        The result of the rock.
    """
    problem = _build_problem(data, surrogate, config, rock_index, error_p95)
    modes = []
    for mode in config.search.modes:
        started = time.perf_counter()
        modes.append(_optimize_mode(problem, mode))
        _LOGGER.info(
            "rock=%s mode=%s seconds=%.1f",
            data.rock,
            mode,
            time.perf_counter() - started,
        )
    return RockResult(
        rock=data.rock,
        space=problem.space,
        baseline=_baseline(problem),
        modes=tuple(modes),
    )


def _build_problem(
    data: RockData,
    surrogate: surrogate_lib.FrozenSurrogate,
    config: experiment_config.ExperimentConfig,
    rock_index: int,
    error_p95: float,
) -> _RockProblem:
    target = config.target
    space = input_space.build_input_space(
        data.rock, data.conditions, config.input_space.tie_r2_threshold
    )
    window = objectives.TimeWindow.from_days(
        surrogate.time_days, target.start_days, target.end_days
    )
    objective = objectives.build_objective(
        config.search.objective, window, target.eps_mol
    )
    seed = config.search.seed + rock_index
    eligible = _eligible(data.real_h2, window, target.min_start_mol)
    reference, reference_nn = _reference(data, surrogate.scaling, config, seed)
    return _RockProblem(
        data=data,
        surrogate=surrogate,
        config=config,
        space=space,
        window=window,
        objective=objective,
        h2_index=v2_data.OUTPUT_FEATURES.index(target.output),
        control_mask=context_control_mask(
            space, config.input_space.context_inputs
        ),
        start_indices=starts.select_starts(
            objective(data.real_h2),
            eligible,
            n_best=config.search.n_best_starts,
            n_random=config.search.n_random_starts,
            seed=seed,
        ),
        physical=_physical_constraints(data, config, surrogate.scaling),
        observed=constraints.observed_range_constraints(
            FEATURE_NAMES,
            space.observed_min,
            space.observed_max,
            surrogate.scaling.condition_scale,
        ),
        reference=reference,
        reference_nn=reference_nn,
        seed=seed,
        error_p95=error_p95,
        baseline_objective=float(objective(data.predicted_h2[eligible]).min()),
    )


def _reference(
    data: RockData,
    scaling: surrogate_lib.SurrogateScaling,
    config: experiment_config.ExperimentConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the normalized train runs (n, 57) and their NN distances."""
    reference = scaling.normalize_conditions(data.conditions[data.is_train])
    reference_nn = diagnostics.reference_nn_distances(
        reference,
        sample_size=config.report.nn_sample_size,
        seed=seed,
        chunk_size=config.report.distance_chunk_size,
    )
    return reference, reference_nn


def _eligible(
    h2: np.ndarray, window: objectives.TimeWindow, min_start_mol: float
) -> np.ndarray:
    """Return True for each run with H2(t_start) >= min_start_mol."""
    return h2[:, window.start_index] >= min_start_mol


def _physical_constraints(
    data: RockData,
    config: experiment_config.ExperimentConfig,
    scaling: surrogate_lib.SurrogateScaling,
) -> constraints.LinearConstraints:
    calibrated = constraints.calibrated_constraints(
        config.constraints.linear,
        data.rock,
        data.conditions,
        FEATURE_NAMES,
        scaling.condition_scale,
    )
    return calibrated.concat(
        constraints.nonnegative_constraints(
            FEATURE_NAMES, scaling.condition_scale
        )
    )


def _baseline(problem: _RockProblem) -> dict[str, object]:
    """Return the best eligible real runs by prediction and by PHREEQC."""
    data = problem.data
    eligible = np.flatnonzero(
        _eligible(
            data.real_h2, problem.window, problem.config.target.min_start_mol
        )
    )
    real = problem.objective(data.real_h2[eligible])
    predicted = problem.objective(data.predicted_h2[eligible])
    by_prediction = int(np.argmin(predicted))
    by_real = int(np.argmin(real))
    return {
        "rock": data.rock,
        "n_eligible": int(eligible.size),
        "predicted_best_run_id": str(data.run_ids[eligible[by_prediction]]),
        "predicted_best_predicted": float(predicted[by_prediction]),
        "predicted_best_real": float(real[by_prediction]),
        "real_best_run_id": str(data.run_ids[eligible[by_real]]),
        "real_best_real": float(real[by_real]),
        "real_best_predicted": float(predicted[by_real]),
    }


def _optimize_mode(problem: _RockProblem, mode: str) -> ModeResult:
    """Run the search of one mode and evaluate the best points."""
    surrogate = problem.surrogate
    rules = problem.physical
    if mode == projections.MODE_BOUNDED:
        rules = rules.concat(problem.observed)
    start_conditions = problem.data.conditions[problem.start_indices]
    z0 = torch.as_tensor(
        problem.space.encode(start_conditions),
        dtype=surrogate.dtype,
        device=surrogate.device,
    )
    schedule = search.SearchSchedule(
        optimizer_factory=functools.partial(
            torch.optim.Adam, lr=problem.config.search.learning_rate
        ),
        n_steps=problem.config.search.steps,
    )
    torch.manual_seed(problem.config.search.seed)
    result = search.search(
        _PenalizedLoss(problem, rules),
        z0,
        projections.build_projection(
            mode, problem.space, surrogate.dtype, surrogate.device
        ),
        schedule,
        control_mask=torch.as_tensor(problem.control_mask),
    )
    table, trajectories = _candidate_table(problem, mode, rules, result)
    return ModeResult(
        mode=mode,
        candidates=table,
        trajectories=trajectories,
        history=result.history,
    )


def _evaluate(
    problem: _RockProblem, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Decode z (B, K) in float64. Return inputs (B, 57) and H2 (B, T)."""
    with torch.no_grad():
        # SearchResult arrays are read-only. torch needs a writable array.
        z_tensor = torch.tensor(z, dtype=torch.float64)
        inputs = problem.space.decode(z_tensor)
    inputs = inputs.numpy()
    h2 = predict.predict_output(
        problem.surrogate,
        inputs,
        output_index=problem.h2_index,
        batch_size=problem.config.report.batch_size,
    )
    return inputs, h2


def _flag_names(flags: np.ndarray, labels: Sequence[str]) -> list[str]:
    """Join the labels of the True entries of each row of flags (B, m)."""
    return [
        _LIST_SEPARATOR.join(labels[i] for i in np.flatnonzero(row))
        for row in flags
    ]


def _constraint_columns(
    problem: _RockProblem,
    rules: constraints.LinearConstraints,
    inputs: np.ndarray,
) -> dict[str, object]:
    """Return the violation and observed-range columns. Shapes (B,)."""
    tolerance = problem.config.constraints.feasibility_tolerance
    with torch.no_grad():
        violation = rules.violation(torch.as_tensor(inputs)).numpy()
    scale = problem.surrogate.scaling.condition_scale
    below = (problem.space.observed_min - inputs) / scale > tolerance
    above = (inputs - problem.space.observed_max) / scale > tolerance
    max_violation = violation.max(axis=1, initial=0.0)
    return {
        "max_violation": max_violation,
        "violated_constraints": _flag_names(violation > tolerance, rules.names),
        "out_of_range_inputs": [
            _LIST_SEPARATOR.join(filter(None, pair))
            for pair in zip(
                _flag_names(above, [f"{n}>max" for n in FEATURE_NAMES]),
                _flag_names(below, [f"{n}<min" for n in FEATURE_NAMES]),
                strict=True,
            )
        ],
        "feasible": max_violation <= tolerance,
    }


def _objective_columns(
    problem: _RockProblem, h2: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the predicted objective columns of H2 (B, T). Shapes (B,)."""
    eps = problem.config.target.eps_mol
    relative = objectives.RelativeLoss(problem.window, eps)(h2)
    h2_start = h2[:, problem.window.start_index]
    return {
        "predicted_objective": problem.objective(h2),
        "predicted_relative_loss": relative,
        "predicted_absolute_loss_mol": objectives.AbsoluteLoss(problem.window)(
            h2
        ),
        "predicted_h2_start_mol": h2_start,
        "predicted_h2_end_mol": h2[:, problem.window.end_index],
        "low_start": h2_start < problem.config.target.min_start_mol,
        "negative": relative < 0,
    }


def _start_columns(
    problem: _RockProblem, z0: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the start run columns. Shapes (B,)."""
    indices = problem.start_indices
    n_best = problem.config.search.n_best_starts
    _, start_h2 = _evaluate(problem, z0)
    return {
        "start_run_id": problem.data.run_ids[indices],
        "start_kind": np.where(
            np.arange(indices.size) < n_best, "best", "random"
        ),
        "start_real_loss": problem.objective(problem.data.real_h2[indices]),
        "start_predicted_loss": problem.objective(start_h2),
    }


def _distance_columns(
    problem: _RockProblem, inputs: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the distance to the nearest train run of the rock."""
    points = problem.surrogate.scaling.normalize_conditions(inputs)
    distances = diagnostics.nearest_distances(
        points,
        problem.reference,
        chunk_size=problem.config.report.distance_chunk_size,
    )
    return {
        "nearest_train_distance": distances,
        "distance_percentile": diagnostics.distance_percentile(
            distances, problem.reference_nn
        ),
    }


def _candidate_table(
    problem: _RockProblem,
    mode: str,
    rules: constraints.LinearConstraints,
    result: search.SearchResult,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build one row for each start. Return the table and H2 (B, T)."""
    inputs, h2 = _evaluate(problem, result.best_z)
    z0 = problem.space.encode(problem.data.conditions[problem.start_indices])
    columns = {
        "rock": problem.data.rock,
        "mode": mode,
        "start_index": np.arange(result.best_z.shape[0]),
        **_start_columns(problem, z0),
        **_objective_columns(problem, h2),
        **_constraint_columns(problem, rules, inputs),
        **_distance_columns(problem, inputs),
        "best_step": result.best_step,
        "first_nonfinite_step": result.first_nonfinite_step,
    }
    table = pd.DataFrame(columns)
    _add_ranks(problem, table, inputs)
    inputs_table = pd.DataFrame(inputs, columns=list(FEATURE_NAMES))
    return pd.concat([table, inputs_table], axis=1), h2


def _add_ranks(
    problem: _RockProblem, table: pd.DataFrame, inputs: np.ndarray
) -> None:
    """Add the columns usable, rank (1..k, or NA), and promising."""
    objective = table["predicted_objective"].to_numpy()
    usable = (
        np.isfinite(objective)
        & table["feasible"].to_numpy()
        & ~table["low_start"].to_numpy()
        & ~table["negative"].to_numpy()
    )
    report = problem.config.report
    kept = candidates.select_candidates(
        problem.surrogate.scaling.normalize_conditions(inputs),
        objective,
        usable,
        min_distance=report.dedup_fraction_of_nn
        * float(np.median(problem.reference_nn)),
        top_k=report.top_k,
    )
    rank = pd.array([pd.NA] * len(table), dtype="Int64")
    rank[kept] = np.arange(1, kept.size + 1)
    table["usable"] = usable
    table["rank"] = rank
    table["promising"] = (
        problem.baseline_objective - objective
    ) > problem.error_p95
