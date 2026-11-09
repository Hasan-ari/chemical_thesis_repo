"""Tests of conditional_input_optimization.optimization.report.summary."""

from __future__ import annotations

import types

import pandas as pd
import pytest

from conditional_input_optimization.optimization import report

_BASELINE = {
    "predicted_best_run_id": "Calcite:1",
    "predicted_best_predicted": 0.2,
    "predicted_best_real": 0.25,
}


def _candidates(rank: int | None) -> pd.DataFrame:
    """Return one candidate row with the given rank (None: not ranked)."""
    return pd.DataFrame(
        {
            "rank": pd.array([rank], dtype="Int64"),
            "predicted_objective": [0.1],
            "promising": [True],
            "distance_percentile": [40.0],
            "out_of_range_inputs": [""],
        }
    )


def _result(ranks: list[int | None]) -> types.SimpleNamespace:
    """Return a stub ExperimentResult with one rock and one mode per rank."""
    modes = [
        types.SimpleNamespace(mode=f"mode_{i}", candidates=_candidates(rank))
        for i, rank in enumerate(ranks)
    ]
    checks = pd.DataFrame(
        {
            "rock": ["Calcite"],
            "objective": ["relative"],
            "split": ["test"],
            "resolvable": [False],
            "spearman": [0.3],
            "mae_over_std": [0.9],
        }
    )
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            name="test", search=types.SimpleNamespace(objective="relative")
        ),
        surrogate_checks=checks,
        rock_results=[
            types.SimpleNamespace(
                rock="Calcite", baseline=_BASELINE, modes=modes
            )
        ],
    )


def _table_rows(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("| mode_")]


@pytest.mark.parametrize("ranks", [[1, None], [None, 1], [None, None], [1, 1]])
def test_summary_has_one_full_row_for_each_mode(ranks):
    text = report.summary_markdown(_result(ranks))
    header = next(line for line in text.splitlines() if line.startswith("|"))
    rows = _table_rows(text)
    assert "| promising |" in header
    assert len(rows) == len(ranks)
    for row, rank in zip(rows, ranks, strict=True):
        assert row.count("|") == header.count("|")
        assert ("no usable candidate" in row) == (rank is None)


def test_summary_shows_the_promising_flag():
    text = report.summary_markdown(_result([1]))
    assert "| promising |" in text
    assert "| yes |" in text
