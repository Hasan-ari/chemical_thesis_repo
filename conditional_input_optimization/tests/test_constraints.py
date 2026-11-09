"""Tests of conditional_input_optimization.optimization.constraints."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conditional_input_optimization.optimization import constraints
from conditional_input_optimization.tests import optimization_fakes as fakes

_NAMES = ("A", "B", "C")
_SCALE = np.array([1.0, 2.0, 1.0])
_SPEC = constraints.LinearConstraintSpec(
    name="a_plus_b", coefficients=(("A", 1.0), ("B", 1.0)), rocks=("all",)
)
_ONLY_TRONA = constraints.LinearConstraintSpec(
    name="trona_only", coefficients=(("C", 1.0),), rocks=("Trona",)
)
_RUNS = np.array([[1.0, 1.0, 0.0], [2.0, 0.5, 1.0], [0.0, 1.0, 3.0]])


def test_calibrated_upper_is_the_max_over_the_runs():
    rules = constraints.calibrated_constraints(
        [_SPEC, _ONLY_TRONA], "Halite", _RUNS, _NAMES, _SCALE
    )
    assert rules.names == ("a_plus_b",)
    np.testing.assert_allclose(rules.upper, [2.5])
    np.testing.assert_allclose(rules.scale, [np.hypot(1.0, 2.0)])
    violation = rules.violation(torch.as_tensor(_RUNS))
    assert torch.all(violation == 0)


def test_violation_is_scaled_and_differentiable():
    rules = constraints.calibrated_constraints(
        [_SPEC], "Halite", _RUNS, _NAMES, _SCALE
    )
    point = torch.tensor([[3.0, 1.0, 0.0]], requires_grad=True)
    violation = rules.violation(point)
    expected = (4.0 - 2.5) / np.hypot(1.0, 2.0)
    assert violation.item() == pytest.approx(expected)
    violation.sum().backward()
    assert point.grad[0, 0] > 0
    assert point.grad[0, 2] == 0


def test_calibrated_constraints_rejects_unknown_input():
    spec = constraints.LinearConstraintSpec("bad", (("Z", 1.0),), ("all",))
    with pytest.raises(ValueError, match="unknown input 'Z'"):
        constraints.calibrated_constraints(
            [spec], "Halite", _RUNS, _NAMES, _SCALE
        )


def test_no_selected_constraint_gives_an_empty_set():
    rules = constraints.calibrated_constraints(
        [_ONLY_TRONA], "Halite", _RUNS, _NAMES, _SCALE
    )
    assert rules.names == ()
    assert rules.violation(torch.as_tensor(_RUNS)).shape == (3, 0)


def test_nonnegative_constraints():
    rules = constraints.nonnegative_constraints(_NAMES, _SCALE)
    violation = rules.violation(torch.tensor([[-2.0, 1.0, -1.0]]))
    np.testing.assert_allclose(violation.numpy(), [[2.0, 0.0, 1.0]])


def test_observed_range_uses_only_varying_inputs():
    rules = constraints.observed_range_constraints(
        _NAMES, np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0, 2.0]), _SCALE
    )
    assert rules.names == (
        "A <= observed max",
        "C <= observed max",
        "A >= observed min",
        "C >= observed min",
    )
    violation = rules.violation(torch.tensor([[1.5, 9.0, -1.0]]))
    np.testing.assert_allclose(violation.numpy(), [[0.5, 0.0, 0.0, 1.0]])


def test_concat_joins_two_sets():
    first = constraints.nonnegative_constraints(_NAMES, _SCALE)
    second = constraints.calibrated_constraints(
        [_SPEC], "Halite", _RUNS, _NAMES, _SCALE
    )
    joined = first.concat(second)
    assert joined.names == (*first.names, "a_plus_b")
    assert joined.matrix.shape == (4, 3)


def test_linear_constraints_rejects_bad_values():
    with pytest.raises(ValueError, match="rows"):
        constraints.LinearConstraints(
            ("a",), np.zeros((2, 3)), np.zeros(1), np.ones(1)
        )
    with pytest.raises(ValueError, match="positive"):
        constraints.LinearConstraints(
            ("a",), np.zeros((1, 3)), np.zeros(1), np.zeros(1)
        )
    with pytest.raises(ValueError, match="needs a name"):
        constraints.LinearConstraintSpec("", (("A", 1.0),), ("all",))


def test_specs_from_config():
    specs = constraints.specs_from_config(
        fakes.raw_config()["constraints"]["linear"]
        + [{"name": "x", "coefficients": {"A": 2}, "rocks": ["Trona"]}]
    )
    assert specs[0].rocks == ("all",)
    assert specs[0].applies_to("Calcite")
    assert specs[1].coefficients == (("A", 2.0),)
    assert not specs[1].applies_to("Calcite")
    with pytest.raises(ValueError, match="exactly the keys"):
        constraints.specs_from_config([{"name": "x"}])


def test_real_runs_of_fake_rock_are_feasible():
    conditions = fakes.fake_conditions(30, seed=0)
    config = fakes.tiny_config()
    scale = np.ones(57)
    rules = constraints.calibrated_constraints(
        config.constraints.linear,
        "RockA",
        conditions,
        fakes.FEATURE_NAMES,
        scale,
    ).concat(constraints.nonnegative_constraints(fakes.FEATURE_NAMES, scale))
    assert rules.violation(torch.as_tensor(conditions)).max() == 0
