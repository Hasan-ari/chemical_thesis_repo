"""Tests of conditional_input_optimization.optimization.result_figures."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import pandas as pd
import pytest
import yaml

from conditional_input_optimization.optimization import (
    diagnostics,
    result_figures,
)

_CONFIG = (
    pathlib.Path(__file__).resolve().parents[1]
    / "configs"
    / "optimization_v1.yaml"
)
_ROCKS = ("Calcite", "Quartzite_schist")
_STARTS = 4
_STEPS = 3
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_ACCEPTANCE = diagnostics.AcceptanceRule(min_spearman=0.7, max_mae_over_std=0.5)


def _surrogate_check() -> pd.DataFrame:
    rows = [
        {
            "rock": rock,
            "objective": objective,
            "split": split,
            "n": 10,
            "mae": 0.1,
            "r2": -1.0,
            "spearman": 0.2 + 0.3 * index,
            "real_std": 0.05,
            "mae_over_std": 2.0 - index,
            "top_decile_precision": 0.1,
            "share_predicted_negative": 0.4,
            "abs_error_p95": 0.2,
            "optimism_gap_top": 1.0,
            "resolvable": False,
        }
        for index, rock in enumerate(_ROCKS)
        for objective in result_figures.OBJECTIVES
        for split in ("test", "train")
    ]
    return pd.DataFrame(rows)


def _baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rock": list(_ROCKS),
            "n_eligible": [100, 80],
            "predicted_best_run_id": [1, 2],
            "predicted_best_predicted": [-0.5, -0.2],
            "predicted_best_real": [0.05, 0.002],
            "real_best_run_id": [3, 4],
            "real_best_real": [1e-4, 1e-3],
            "real_best_predicted": [0.02, -0.2],
        }
    )


def _search_history() -> pd.DataFrame:
    rows = [
        {
            "rock": rock,
            "mode": mode,
            "step": step,
            "min_loss": -0.1 * (step + 1) * (10 if mode == "unbounded" else 1),
            "median_loss": 0.01,
        }
        for rock in _ROCKS
        for mode in result_figures.MODES
        for step in range(_STEPS)
    ]
    return pd.DataFrame(rows)


def _all_candidates() -> pd.DataFrame:
    rows = [
        {
            "rock": rock,
            "mode": mode,
            "start_index": start,
            "start_kind": "best" if start == 0 else "random",
            "start_real_loss": 0.01,
            "predicted_relative_loss": -0.5 * (start + 1),
            "negative": True,
            "feasible": True,
            "low_start": False,
            "nearest_train_distance": 1.0,
            "usable": False,
        }
        for rock in _ROCKS
        for mode in result_figures.MODES
        for start in range(_STARTS)
    ]
    frame = pd.DataFrame(rows)
    frame.loc[0, ["predicted_relative_loss", "negative", "usable"]] = [
        0.009,
        False,
        True,
    ]
    return frame


def _input_space() -> pd.DataFrame:
    roles = ("free", "free", "tied", "fixed")
    rows = [
        {"rock": rock, "input": f"INPUT_{index}", "role": role}
        for rock in _ROCKS
        for index, role in enumerate(roles)
    ]
    return pd.DataFrame(rows).assign(r_squared=None, sources=None)


def _h2_trajectories() -> pd.DataFrame:
    times = (0.0, 0.6, 90.0, 180.0)
    return pd.DataFrame(
        {
            "rock": "Quartzite_schist",
            "mode": "bounded",
            "rank": 1.0,
            "time_days": times,
            "h2_mol": (0.01, 0.53, 0.528, 0.529),
        }
    )


_TABLES: dict[str, Callable[[], pd.DataFrame]] = {
    "surrogate_check": _surrogate_check,
    "search_history": _search_history,
    "all_candidates": _all_candidates,
    "baseline": _baseline,
    "input_space": _input_space,
    "h2_trajectories": _h2_trajectories,
}


@pytest.fixture
def results_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    config = yaml.safe_load(_CONFIG.read_text())
    config["rocks"] = list(_ROCKS)
    (tmp_path / result_figures.CONFIG_FILE_NAME).write_text(
        yaml.safe_dump(config)
    )
    for key, build in _TABLES.items():
        name = result_figures.CSV_FILE_NAMES[key]
        build().to_csv(tmp_path / name, index=False)
    return tmp_path


def _assert_png(path: pathlib.Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 0
    assert path.read_bytes()[: len(_PNG_SIGNATURE)] == _PNG_SIGNATURE


_FIGURE_CALLS: dict[str, Callable[[pathlib.Path], pathlib.Path]] = {
    "gate": lambda path: result_figures.plot_surrogate_gate(
        _surrogate_check(), _ROCKS, _ACCEPTANCE, path
    ),
    "negative": lambda path: result_figures.plot_negative_share(
        _surrogate_check(), _baseline(), _ROCKS, path
    ),
    "search": lambda path: result_figures.plot_search_convergence(
        _search_history(), _ROCKS, path
    ),
    "candidates": lambda path: result_figures.plot_candidates(
        _all_candidates(), _baseline(), _ROCKS, path
    ),
    "roles": lambda path: result_figures.plot_input_roles(
        _input_space(), _ROCKS, path
    ),
    "h2": lambda path: result_figures.plot_usable_candidate_h2(
        _h2_trajectories(), (0.6, 180.0), path
    ),
}


@pytest.mark.parametrize("name", sorted(_FIGURE_CALLS))
def test_each_figure_writes_non_empty_png(
    name: str, tmp_path: pathlib.Path
) -> None:
    path = tmp_path / "nested" / f"{name}.png"

    written = _FIGURE_CALLS[name](path)

    assert written == path
    _assert_png(path)


def test_usable_h2_with_no_rows_writes_note_png(tmp_path: pathlib.Path):
    empty = _h2_trajectories().iloc[0:0]

    path = result_figures.plot_usable_candidate_h2(
        empty, (0.6, 180.0), tmp_path / "empty.png"
    )

    _assert_png(path)


def test_facade_returns_six_paths_in_order(results_dir: pathlib.Path):
    output_dir = results_dir / "figures"

    paths = result_figures.write_all_figures(results_dir, output_dir)

    assert len(paths) == 6
    assert [path.name for path in paths] == list(
        result_figures.FIGURE_FILE_NAMES
    )
    for path in paths:
        _assert_png(path)
    assert (output_dir / "README.md").read_text().startswith("# Figures")


def test_cli_writes_figures_into_results_folder(results_dir: pathlib.Path):
    result_figures.main(["--results", str(results_dir)])

    figures = results_dir / result_figures.FIGURES_DIR_NAME
    for name in result_figures.FIGURE_FILE_NAMES:
        _assert_png(figures / name)


@pytest.mark.parametrize("key", sorted(result_figures.CSV_FILE_NAMES))
def test_missing_csv_raises_clear_error(
    key: str, results_dir: pathlib.Path
) -> None:
    name = result_figures.CSV_FILE_NAMES[key]
    (results_dir / name).unlink()

    with pytest.raises(FileNotFoundError, match=name):
        result_figures.write_all_figures(results_dir, results_dir / "out")


def test_missing_config_raises_clear_error(results_dir: pathlib.Path):
    (results_dir / result_figures.CONFIG_FILE_NAME).unlink()

    with pytest.raises(FileNotFoundError, match="config.yaml"):
        result_figures.write_all_figures(results_dir, results_dir / "out")


def test_missing_column_raises_value_error(tmp_path: pathlib.Path):
    table = _surrogate_check().drop(columns="spearman")

    with pytest.raises(ValueError, match="spearman"):
        result_figures.plot_surrogate_gate(
            table, _ROCKS, _ACCEPTANCE, tmp_path / "gate.png"
        )


def test_missing_rock_row_raises_value_error(tmp_path: pathlib.Path):
    table = _surrogate_check()
    table = table[table["rock"] != _ROCKS[1]]

    with pytest.raises(ValueError, match=_ROCKS[1]):
        result_figures.plot_surrogate_gate(
            table, _ROCKS, _ACCEPTANCE, tmp_path / "gate.png"
        )


def test_unknown_role_raises_value_error(tmp_path: pathlib.Path):
    table = _input_space()
    table.loc[0, "role"] = "frozen"

    with pytest.raises(ValueError, match="frozen"):
        result_figures.plot_input_roles(table, _ROCKS, tmp_path / "roles.png")
