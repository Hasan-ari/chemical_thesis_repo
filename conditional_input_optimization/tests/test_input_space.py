"""Tests of conditional_input_optimization.optimization.input_space."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from conditional_input_optimization.optimization import input_space
from conditional_input_optimization.tests import optimization_fakes as fakes


def _three_columns(n_runs=50, seed=0):
    """Column 0 free, column 1 = 3 * x0 + 2 (tied), column 2 fixed."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, 1.0, n_runs)
    return np.column_stack([x0, 3.0 * x0 + 2.0, np.full(n_runs, 7.0)])


def test_detect_ties_finds_free_tied_and_fixed():
    detection = input_space.detect_ties(_three_columns(), 0.99)
    assert detection.free_columns == (0,)
    assert detection.fixed_columns == (2,)
    (tie,) = detection.ties
    assert tie.column == 1
    assert tie.sources == (0,)
    assert tie.intercept == pytest.approx(2.0)
    assert tie.coefficients == pytest.approx((3.0,))
    assert tie.r_squared == pytest.approx(1.0)


def test_detect_ties_keeps_the_first_of_two_equal_inputs_free():
    conditions = _three_columns()[:, [0, 0]]
    detection = input_space.detect_ties(conditions, 0.99)
    assert detection.free_columns == (0,)
    assert [tie.column for tie in detection.ties] == [1]


def test_detect_ties_keeps_independent_inputs_free():
    rng = np.random.default_rng(1)
    detection = input_space.detect_ties(rng.normal(size=(200, 3)), 0.99)
    assert detection.free_columns == (0, 1, 2)
    assert detection.ties == ()


def test_detect_ties_on_fake_rock():
    detection = input_space.detect_ties(fakes.fake_conditions(40, seed=0), 0.99)
    names = fakes.FEATURE_NAMES
    assert [names[i] for i in detection.free_columns] == list(
        fakes.VARYING_RANGES
    )
    assert [names[tie.column] for tie in detection.ties] == ["CL"]
    assert len(detection.fixed_columns) == 57 - 6


@pytest.mark.parametrize("threshold", [0.0, 1.5])
def test_detect_ties_rejects_bad_threshold(threshold):
    with pytest.raises(ValueError, match="r2_threshold"):
        input_space.detect_ties(_three_columns(), threshold)


def test_detect_ties_rejects_one_run():
    with pytest.raises(ValueError, match="N >= 2"):
        input_space.detect_ties(_three_columns()[:1], 0.99)


def test_tie_rejects_bad_values():
    with pytest.raises(ValueError, match="coefficient"):
        input_space.Tie(1, (0,), 0.0, (1.0, 2.0), 1.0)
    with pytest.raises(ValueError, match="own source"):
        input_space.Tie(1, (1,), 0.0, (1.0,), 1.0)


def test_decode_of_encode_gives_the_runs_back():
    conditions = fakes.fake_conditions(40, seed=2)
    space = input_space.build_input_space("RockA", conditions, 0.99)
    z = space.encode(conditions)
    assert z.shape == (40, space.n_free)
    decoded = space.decode(torch.as_tensor(z)).numpy()
    np.testing.assert_allclose(decoded, conditions, rtol=1e-12, atol=1e-12)


def test_decode_is_differentiable_and_keeps_dtype():
    space = input_space.build_input_space(
        "RockA", fakes.fake_conditions(20, seed=3), 0.99
    )
    z = torch.zeros(2, space.n_free, dtype=torch.float32, requires_grad=True)
    decoded = space.decode(z)
    assert decoded.dtype == torch.float32
    decoded.sum().backward()
    assert z.grad is not None
    assert torch.all(z.grad != 0)


def test_free_bounds_contain_the_runs():
    conditions = fakes.fake_conditions(30, seed=4)
    space = input_space.build_input_space("RockA", conditions, 0.99)
    lower, upper = space.free_bounds_z()
    z = space.encode(conditions)
    np.testing.assert_allclose(z.min(axis=0), lower)
    np.testing.assert_allclose(z.max(axis=0), upper)


def test_roles_and_role_table():
    space = input_space.build_input_space("RockA", _three_columns(), 0.99)
    assert space.roles() == ["free", "tied", "fixed"]
    rows = input_space.role_table([space])
    assert [row["role"] for row in rows] == ["free", "tied", "fixed"]
    assert rows[1]["sources"] == [0]
    assert np.isnan(rows[0]["r_squared"])


def test_space_arrays_are_read_only():
    space = input_space.build_input_space("RockA", _three_columns(), 0.99)
    with pytest.raises(ValueError, match="read-only"):
        space.base[0] = 1.0


def test_space_rejects_bad_shapes_and_std():
    space = input_space.build_input_space("RockA", _three_columns(), 0.99)
    with pytest.raises(ValueError, match="weights"):
        dataclasses.replace(space, weights=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="free_std"):
        dataclasses.replace(space, free_std=np.zeros(1))
