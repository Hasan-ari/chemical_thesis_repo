"""Write the results folder of one experiment for the chemistry team.

The folder is <results_dir>/<YYYYmmdd_HHMMSS>_<name>/. README.md in the
folder explains each file.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import pathlib
import platform
import shutil
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml

from conditional_input_optimization.optimization import (
    file_units,
    input_space,
    rock_optimizer,
)

if TYPE_CHECKING:
    from conditional_input_optimization.optimization import experiment

FEATURE_NAMES = rock_optimizer.FEATURE_NAMES
LEADING_COLUMNS = (
    "rock",
    "mode",
    "rank",
    "promising",
    "feasible",
    "low_start",
    "negative",
)
LIBRARIES = ("numpy", "pandas", "torch", "scikit-learn", "pyyaml", "scipy")
_SOURCE_SEPARATOR = ";"
_DIGITS = 4
# The summary columns that only a mode with a usable candidate has.
_CANDIDATE_SUMMARY_KEYS = (
    "promising",
    "distance percentile",
    "out-of-range inputs",
)
_NO_VALUE = "-"

README_TEXT = """\
# Input optimization results

WARNING: All values in this folder are predictions of the v2 surrogate model.
They are not PHREEQC results. Run PHREEQC on a candidate before you use it.

WARNING: The unbounded mode lets the free inputs leave the observed range of
the rock. Its candidates are extrapolations. The surrogate is less reliable
there.

## Files

- `summary.md`: one short table for each rock.
- `config.yaml`: a copy of the experiment config file.
- `manifest.yaml`: the git commit, the model file hashes, the library
  versions, the seeds, and the run time.
- `optimized_inputs.csv`: the top candidates of each rock and mode. The 57
  inputs are in mol/kgw.
- `optimized_inputs_file_units.csv`: the same rows. The solution inputs are
  in the units of the rock input files (mg/L for the schists).
- `all_candidates.csv`: one row for each start of each rock and mode.
- `h2_trajectories.csv`: the predicted H2 gas amount of each top candidate,
  in mol, at each time point, in days.
- `input_space.csv`: the role of each input in each rock: free, tied, or
  fixed. A tied input is a linear function of its source inputs.
- `surrogate_check.csv`: the surrogate error on the real test runs and on
  the real train runs of each rock, for the relative and the absolute
  loss (column `objective`). The search uses only `search.objective`.
- `baseline.csv`: the best real runs of each rock.
- `search_history.csv`: the min and the median penalized loss of the starts
  at each step.

## Units

- Solution inputs: mol/kgw (file units in `optimized_inputs_file_units.csv`).
- H2 amounts and absolute losses: mol.
- Relative loss: (H2(t_start) - H2(t_end)) / H2(t_start). No unit.
- Time: days.
- Distances: Euclidean distance in the normalized input space of the model.

## Columns and flags

- `rank`: 1 is the best candidate of the rock and mode. Empty: not selected.
- `predicted_objective`: the predicted objective that the search minimized.
- `start_run_id`: the real run that started the search.
- `start_real_loss`: the PHREEQC objective of the start run.
- `start_predicted_loss`: the predicted objective of the start run.
- `feasible`: True if the max scaled constraint violation is within the
  tolerance.
- `violated_constraints`: the constraints with a violation above the
  tolerance.
- `out_of_range_inputs`: the inputs outside the observed range of the rock.
- `low_start`: True if the predicted H2(t_start) is below the minimum. Then
  the relative loss is not reliable.
- `negative`: True if the predicted relative loss is below 0 (an H2 gain).
- `promising`: True if the candidate is better than the predicted-best real
  run by more than the 95th percentile of the test error.
- `distance_percentile`: 100 means that the candidate is farther from the
  train runs than all sampled train runs are from their nearest neighbor.
- `best_step`: the search step of the best point.
- `first_nonfinite_step`: the first step with a non-finite loss. -1: none.

NOTE: Only a candidate that is feasible, not low_start, and not negative can
get a rank. A small distance percentile and a resolvable rock in
`surrogate_check.csv` make a prediction more reliable.
"""


def write_results(
    result: experiment.ExperimentResult,
    config_path: pathlib.Path,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    """Write the results folder.

    Args:
        result: The experiment result.
        config_path: The config file that the run used. It is copied.
        output_dir: The results folder. The run folder goes inside it.

    Returns:
        The new run folder.

    Raises:
        FileExistsError: If the run folder exists.
    """
    stamp = result.started_at.strftime("%Y%m%d_%H%M%S")
    folder = output_dir / f"{stamp}_{result.config.name}"
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, folder / "config.yaml")
    for name, frame in result_tables(result).items():
        frame.to_csv(folder / name, index=False)
    (folder / "manifest.yaml").write_text(
        yaml.safe_dump(manifest(result), sort_keys=False)
    )
    (folder / "summary.md").write_text(summary_markdown(result))
    (folder / "README.md").write_text(README_TEXT)
    return folder


def result_tables(
    result: experiment.ExperimentResult,
) -> dict[str, pd.DataFrame]:
    """Build each CSV table of the results folder.

    Args:
        result: The experiment result.

    Returns:
        The file name and the table of each CSV file.
    """
    all_candidates = pd.concat(
        [
            mode.candidates
            for rock in result.rock_results
            for mode in rock.modes
        ],
        ignore_index=True,
    )
    optimized = _optimized_inputs(all_candidates)
    return {
        "optimized_inputs.csv": optimized,
        "optimized_inputs_file_units.csv": _file_unit_inputs(
            optimized, result.input_units
        ),
        "all_candidates.csv": all_candidates,
        "h2_trajectories.csv": _trajectory_table(result),
        "input_space.csv": _input_space_table(result),
        "surrogate_check.csv": result.surrogate_checks,
        "baseline.csv": pd.DataFrame(
            [rock.baseline for rock in result.rock_results]
        ),
        "search_history.csv": _history_table(result),
    }


def _optimized_inputs(all_candidates: pd.DataFrame) -> pd.DataFrame:
    """Return the ranked rows, with the leading columns first."""
    ranked = all_candidates[all_candidates["rank"].notna()]
    others = [c for c in ranked.columns if c not in LEADING_COLUMNS]
    return ranked[[*LEADING_COLUMNS, *others]].reset_index(drop=True)


def _file_unit_inputs(
    optimized: pd.DataFrame, input_units: dict[str, str]
) -> pd.DataFrame:
    """Return the ranked rows with the inputs in the rock file units."""
    converted = optimized.copy()
    converted.insert(3, "input_units", "")
    names = list(FEATURE_NAMES)
    for rock, rows in optimized.groupby("rock", sort=False).groups.items():
        units = input_units[rock]
        converted.loc[rows, "input_units"] = units
        converted.loc[rows, names] = file_units.to_file_units(
            optimized.loc[rows, names].to_numpy(), names, units
        )
    return converted


def _trajectory_table(result: experiment.ExperimentResult) -> pd.DataFrame:
    """Return the long table of the predicted H2 of each top candidate."""
    frames = []
    n_times = result.time_days.shape[0]
    for rock in result.rock_results:
        for mode in rock.modes:
            ranks = mode.candidates["rank"]
            selected = np.flatnonzero(ranks.notna().to_numpy())
            frames.append(
                pd.DataFrame(
                    {
                        "rock": rock.rock,
                        "mode": mode.mode,
                        "rank": np.repeat(ranks.to_numpy()[selected], n_times),
                        "time_days": np.tile(result.time_days, selected.size),
                        "h2_mol": mode.trajectories[selected].ravel(),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _input_space_table(result: experiment.ExperimentResult) -> pd.DataFrame:
    """Return the role of each input in each rock."""
    rows = input_space.role_table([rock.space for rock in result.rock_results])
    return pd.DataFrame(
        {
            "rock": [row["rock"] for row in rows],
            "input": [FEATURE_NAMES[row["column"]] for row in rows],
            "role": [row["role"] for row in rows],
            "r_squared": [row["r_squared"] for row in rows],
            "sources": [
                _SOURCE_SEPARATOR.join(FEATURE_NAMES[i] for i in row["sources"])
                for row in rows
            ],
        }
    )


def _history_table(result: experiment.ExperimentResult) -> pd.DataFrame:
    """Return the min and the median finite loss at each step."""
    frames = []
    for rock in result.rock_results:
        for mode in rock.modes:
            history = pd.DataFrame(mode.history.astype(np.float64))
            history = history.where(np.isfinite(history))
            frames.append(
                pd.DataFrame(
                    {
                        "rock": rock.rock,
                        "mode": mode.mode,
                        "step": np.arange(len(history)),
                        "min_loss": history.min(axis=1),
                        "median_loss": history.median(axis=1),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _file_sha256(path: pathlib.Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _git(repo_root: pathlib.Path, *args: str) -> str | None:
    """Return the stdout of a git command, or None if git fails."""
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def manifest(result: experiment.ExperimentResult) -> dict[str, object]:
    """Return the provenance record of the run.

    Args:
        result: The experiment result.

    Returns:
        A YAML-safe mapping.
    """
    config = result.config
    root = result.repo_root
    status = _git(root, "status", "--porcelain")
    finished_at = result.started_at + datetime.timedelta(
        seconds=result.run_seconds
    )
    return {
        "name": config.name,
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "model_files": {
            str(path): _file_sha256(root / path)
            for path in (config.model.checkpoint, config.model.preprocessors)
        },
        "library_versions": _library_versions(),
        "device": config.model.device,
        "seeds": {
            "split": config.split.seed,
            "search": config.search.seed,
            "start_draw": "search.seed + position of the rock in rock_config",
        },
        "rocks": list(config.rocks),
        "steps": config.search.steps,
        "scaler_mean_gap": result.scaler_gap,
        "started_at": result.started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "run_seconds": round(result.run_seconds, 1),
    }


def _format(value: object) -> str:
    if isinstance(value, float | np.floating):
        return f"{value:.{_DIGITS}g}"
    return str(value)


def _markdown_table(rows: list[dict[str, object]]) -> str:
    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += [
        "| " + " | ".join(_format(row[key]) for key in header) + " |"
        for row in rows
    ]
    return "\n".join(lines)


def _mode_summary(
    mode: rock_optimizer.ModeResult, baseline: dict[str, object]
) -> dict[str, object]:
    """Return one summary row: the rank-1 candidate of one mode."""
    best = mode.candidates[mode.candidates["rank"] == 1]
    row: dict[str, object] = {
        "mode": mode.mode,
        "baseline objective": baseline["predicted_best_predicted"],
    }
    if best.empty:
        # The table header comes from the first row, so each row needs
        # the same keys.
        return {
            **row,
            "best candidate objective": "no usable candidate",
            **dict.fromkeys(_CANDIDATE_SUMMARY_KEYS, _NO_VALUE),
        }
    first = best.iloc[0]
    return {
        **row,
        "best candidate objective": first["predicted_objective"],
        "promising": "yes" if first["promising"] else "no",
        "distance percentile": first["distance_percentile"],
        "out-of-range inputs": first["out_of_range_inputs"] or "none",
    }


def _check_lines(rock: str, checks: pd.DataFrame) -> list[str]:
    """Return one line for each objective: the check on the test runs."""
    test = checks[(checks["rock"] == rock) & (checks["split"] == "test")]
    return [
        f"- Surrogate check on the test runs, {check['objective']} loss: "
        f"resolvable = {'yes' if check['resolvable'] else 'no'}, Spearman = "
        f"{_format(check['spearman'])}, MAE/std = "
        f"{_format(check['mae_over_std'])}."
        for check in test.to_dict("records")
    ]


def _rock_summary(rock: rock_optimizer.RockResult, checks: pd.DataFrame) -> str:
    baseline = rock.baseline
    lines = [
        f"## {rock.rock}",
        "",
        *_check_lines(rock.rock, checks),
        "",
        f"Baseline: the predicted-best real run is "
        f"{baseline['predicted_best_run_id']} (predicted "
        f"{_format(baseline['predicted_best_predicted'])}, real "
        f"{_format(baseline['predicted_best_real'])}).",
        "",
        _markdown_table([_mode_summary(m, baseline) for m in rock.modes]),
        "",
    ]
    return "\n".join(lines)


def summary_markdown(result: experiment.ExperimentResult) -> str:
    """Return summary.md: one short section for each rock.

    Args:
        result: The experiment result.

    Returns:
        The Markdown text.
    """
    header = [
        f"# Summary: {result.config.name}",
        "",
        "WARNING: These values are surrogate predictions. PHREEQC must "
        "check them.",
        "",
        f"Objective: {result.config.search.objective} H2 loss. Smaller is "
        "better.",
        "",
    ]
    sections = [
        _rock_summary(rock, result.surrogate_checks)
        for rock in result.rock_results
    ]
    return "\n".join(header + sections)
