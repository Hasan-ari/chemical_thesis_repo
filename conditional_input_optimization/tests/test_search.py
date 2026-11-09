"""Tests of the search loop and the projections."""

from __future__ import annotations

import functools

import numpy as np
import pytest
import torch

from conditional_input_optimization.optimization import (
    input_space,
    projections,
    search,
)
from conditional_input_optimization.tests import optimization_fakes as fakes

_TARGET = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)


def _quadratic(z):
    return ((z - _TARGET) ** 2).sum(dim=1)


def _schedule(n_steps=300, lr=0.1, optimizer=torch.optim.Adam):
    return search.SearchSchedule(
        optimizer_factory=functools.partial(optimizer, lr=lr), n_steps=n_steps
    )


def _starts(n_starts=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n_starts, 3, dtype=torch.float64, generator=generator)


def test_search_finds_the_minimum_of_a_quadratic():
    result = search.search(
        _quadratic, _starts(), projections.NoProjection(), _schedule()
    )
    np.testing.assert_allclose(
        result.best_z, np.tile(_TARGET.numpy(), (4, 1)), atol=1e-3
    )
    assert result.history.shape == (301, 4)
    assert (result.best_loss < 1e-5).all()
    np.testing.assert_array_equal(result.first_nonfinite_step, -1)


def test_box_projection_keeps_points_in_the_box():
    box = projections.BoxProjection(
        lower=torch.full((3,), -1.0, dtype=torch.float64),
        upper=torch.full((3,), 0.8, dtype=torch.float64),
    )
    result = search.search(_quadratic, _starts(), box, _schedule())
    np.testing.assert_allclose(
        result.best_z, np.tile([0.8, -1.0, 0.5], (4, 1)), atol=1e-3
    )


def test_control_mask_keeps_masked_columns_at_the_start():
    z0 = _starts()
    mask = torch.tensor([True, False, True])
    result = search.search(
        _quadratic,
        z0,
        projections.NoProjection(),
        _schedule(),
        control_mask=mask,
    )
    np.testing.assert_array_equal(result.best_z[:, 1], z0[:, 1].numpy())
    np.testing.assert_allclose(result.best_z[:, 0], 1.0, atol=1e-3)


def test_search_keeps_the_best_point_when_the_loss_grows():
    z0 = _starts()
    result = search.search(
        _quadratic,
        z0,
        projections.NoProjection(),
        _schedule(n_steps=5, lr=10.0, optimizer=torch.optim.SGD),
    )
    np.testing.assert_array_equal(result.best_step, 0)
    np.testing.assert_array_equal(result.best_z, z0.numpy())
    assert (result.history[-1] > result.history[0]).all()


def test_batch_result_equals_single_start_results():
    z0 = _starts(n_starts=3)
    batch = search.search(
        _quadratic, z0, projections.NoProjection(), _schedule(n_steps=20)
    )
    for row in range(3):
        single = search.search(
            _quadratic,
            z0[row : row + 1],
            projections.NoProjection(),
            _schedule(n_steps=20),
        )
        np.testing.assert_allclose(batch.best_z[row], single.best_z[0])
        np.testing.assert_allclose(batch.history[:, row], single.history[:, 0])


def test_nonfinite_start_does_not_move_and_does_not_affect_others():
    def loss_fn(z):
        loss = _quadratic(z)
        return torch.where(z[:, 0] > 5.0, torch.inf, loss)

    z0 = _starts(n_starts=2)
    z0[1, 0] = 10.0
    result = search.search(
        loss_fn, z0, projections.NoProjection(), _schedule(n_steps=50)
    )
    clean = search.search(
        _quadratic, z0[:1], projections.NoProjection(), _schedule(n_steps=50)
    )
    assert result.first_nonfinite_step.tolist() == [-1, 0]
    assert result.best_step[1] == -1
    assert np.isinf(result.best_loss[1])
    np.testing.assert_array_equal(result.best_z[1], z0[1].numpy())
    np.testing.assert_allclose(result.best_z[0], clean.best_z[0])


def test_first_nonfinite_step_records_a_later_nan():
    def loss_fn(z):
        loss = _quadratic(z)
        return torch.where(z[:, 0] > 0.5, torch.nan, loss)

    z0 = torch.zeros(1, 3, dtype=torch.float64)
    result = search.search(
        loss_fn, z0, projections.NoProjection(), _schedule(n_steps=50)
    )
    assert result.first_nonfinite_step[0] > 0
    assert result.best_z[0, 0] <= 0.5


def test_zero_steps_evaluates_the_projected_starts():
    box = projections.BoxProjection(
        lower=torch.zeros(3, dtype=torch.float64),
        upper=torch.ones(3, dtype=torch.float64),
    )
    z0 = torch.full((2, 3), 2.0, dtype=torch.float64)
    result = search.search(_quadratic, z0, box, _schedule(n_steps=0))
    np.testing.assert_array_equal(result.best_z, np.ones((2, 3)))
    assert result.history.shape == (1, 2)


def test_search_rejects_bad_inputs():
    with pytest.raises(ValueError, match="z0"):
        search.search(
            _quadratic,
            torch.zeros(3, dtype=torch.float64),
            projections.NoProjection(),
            _schedule(),
        )
    with pytest.raises(ValueError, match="control_mask"):
        search.search(
            _quadratic,
            _starts(),
            projections.NoProjection(),
            _schedule(),
            control_mask=torch.ones(2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="n_steps"):
        _schedule(n_steps=-1)


def test_search_result_is_read_only_and_checks_shapes():
    result = search.search(
        _quadratic, _starts(), projections.NoProjection(), _schedule(2)
    )
    with pytest.raises(ValueError, match="read-only"):
        result.best_z[0, 0] = 1.0
    with pytest.raises(ValueError, match="best_loss"):
        search.SearchResult(
            best_z=np.zeros((2, 3)),
            best_loss=np.zeros(3),
            best_step=np.zeros(2, dtype=int),
            first_nonfinite_step=np.zeros(2, dtype=int),
            history=np.zeros((1, 2)),
        )


def test_box_projection_rejects_bad_limits():
    with pytest.raises(ValueError, match="lower must be <= upper"):
        projections.BoxProjection(torch.ones(2), torch.zeros(2))
    with pytest.raises(ValueError, match="1D"):
        projections.BoxProjection(torch.zeros(2), torch.ones(3))


def test_build_projection_for_each_mode():
    space = input_space.build_input_space(
        "RockA", fakes.fake_conditions(30, seed=0), 0.99
    )
    bounded = projections.build_projection(
        projections.MODE_BOUNDED, space, torch.float64, torch.device("cpu")
    )
    lower, upper = space.free_bounds_z()
    np.testing.assert_allclose(bounded.lower.numpy(), lower)
    np.testing.assert_allclose(bounded.upper.numpy(), upper)
    unbounded = projections.build_projection(
        projections.MODE_UNBOUNDED, space, torch.float64, torch.device("cpu")
    )
    assert isinstance(unbounded, projections.NoProjection)
    with pytest.raises(ValueError, match="mode"):
        projections.build_projection(
            "free", space, torch.float64, torch.device("cpu")
        )
