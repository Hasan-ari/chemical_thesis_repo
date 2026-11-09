"""Tests of the start selection and the candidate selection."""

from __future__ import annotations

import numpy as np
import pytest

from conditional_input_optimization.optimization import candidates, starts

_SCORES = np.array([5.0, 1.0, 3.0, 0.5, 4.0, 2.0, 9.0, 0.1, 7.0, 6.0])
_ELIGIBLE = np.array([True] * 7 + [False] + [True] * 2)


def test_select_starts_puts_the_best_eligible_runs_first():
    selected = starts.select_starts(
        _SCORES, _ELIGIBLE, n_best=3, n_random=4, seed=0
    )
    assert selected.dtype == np.int64
    assert selected[:3].tolist() == [3, 1, 5]
    assert len(set(selected.tolist())) == 7
    assert _ELIGIBLE[selected].all()


def test_select_starts_is_reproducible_and_depends_on_seed():
    draws = [
        starts.select_starts(_SCORES, _ELIGIBLE, n_best=1, n_random=5, seed=s)
        for s in (0, 0, 1)
    ]
    np.testing.assert_array_equal(draws[0], draws[1])
    assert not np.array_equal(draws[0], draws[2])


def test_select_starts_can_take_all_eligible_runs():
    selected = starts.select_starts(
        _SCORES, _ELIGIBLE, n_best=0, n_random=9, seed=0
    )
    assert sorted(selected.tolist()) == np.flatnonzero(_ELIGIBLE).tolist()


def test_select_starts_ignores_nonfinite_ineligible_scores():
    scores = _SCORES.copy()
    scores[7] = np.nan
    selected = starts.select_starts(
        scores, _ELIGIBLE, n_best=2, n_random=0, seed=0
    )
    assert selected.tolist() == [3, 1]


@pytest.mark.parametrize(
    ("scores", "eligible", "n_best", "n_random", "message"),
    [
        (_SCORES, _ELIGIBLE, -1, 0, ">= 0"),
        (_SCORES, _ELIGIBLE[:5], 1, 0, "equal shapes"),
        (_SCORES, _ELIGIBLE, 5, 5, "eligible runs"),
        (np.where(_ELIGIBLE, np.nan, 0.0), _ELIGIBLE, 1, 0, "finite"),
    ],
)
def test_select_starts_rejects_bad_inputs(
    scores, eligible, n_best, n_random, message
):
    with pytest.raises(ValueError, match=message):
        starts.select_starts(
            scores, eligible, n_best=n_best, n_random=n_random, seed=0
        )


_POINTS = np.array([[0.0, 0.0], [0.05, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0]])


def test_select_candidates_removes_near_duplicates():
    kept = candidates.select_candidates(
        _POINTS,
        np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
        np.ones(5, dtype=bool),
        min_distance=0.5,
        top_k=3,
    )
    assert kept.dtype == np.int64
    assert kept.tolist() == [0, 2, 3]


def test_select_candidates_skips_unusable_and_nonfinite_points():
    points = _POINTS.copy()
    points[3, 1] = np.inf
    kept = candidates.select_candidates(
        points,
        np.array([np.nan, 0.1, 0.2, 0.3, 0.4]),
        np.array([True, True, False, True, True]),
        min_distance=0.0,
        top_k=5,
    )
    assert kept.tolist() == [1, 4]


def test_select_candidates_with_zero_top_k_keeps_nothing():
    kept = candidates.select_candidates(
        _POINTS, np.zeros(5), np.ones(5, dtype=bool), min_distance=0, top_k=0
    )
    assert kept.size == 0


@pytest.mark.parametrize(
    ("points", "usable", "min_distance", "message"),
    [
        (_POINTS[:3], np.ones(5, dtype=bool), 0.0, "points"),
        (_POINTS, np.ones(4, dtype=bool), 0.0, "usable"),
        (_POINTS, np.ones(5, dtype=bool), -1.0, ">= 0"),
    ],
)
def test_select_candidates_rejects_bad_inputs(
    points, usable, min_distance, message
):
    with pytest.raises(ValueError, match=message):
        candidates.select_candidates(
            points,
            np.zeros(5),
            usable,
            min_distance=min_distance,
            top_k=1,
        )
