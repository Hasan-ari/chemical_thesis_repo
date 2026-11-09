"""Tests of the start-H2 hinge in the search loss of rock_optimizer."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from conditional_input_optimization.optimization import (
    constraints,
    rock_optimizer,
)

_MIN_START = 0.1
_WEIGHT = 100.0


def test_start_shortfall_values():
    h2_start = torch.tensor([0.2, 0.1, 0.05, 0.0, -0.1], dtype=torch.float64)
    shortfall = rock_optimizer.start_shortfall(h2_start, _MIN_START)
    np.testing.assert_allclose(shortfall.numpy(), [0.0, 0.0, 0.5, 1.0, 2.0])


def test_start_shortfall_gradient_pushes_the_start_up():
    h2_start = torch.tensor([0.05, 0.3], requires_grad=True)
    rock_optimizer.start_shortfall(h2_start, _MIN_START).pow(2).sum().backward()
    assert h2_start.grad[0] < 0
    assert h2_start.grad[1] == 0


@pytest.mark.parametrize("min_start", [0.0, -1.0, float("nan")])
def test_start_shortfall_rejects_a_bad_limit(min_start):
    with pytest.raises(ValueError, match="min_start_mol"):
        rock_optimizer.start_shortfall(torch.zeros(1), min_start)


class _IdentitySpace:
    def decode(self, z):
        return z


class _StartIsTheInput:
    """A stub surrogate. H2 is the first input at each time step."""

    def predict(self, inputs):
        return inputs[:, :1, None].expand(-1, 3, 1)


def _stub_problem():
    """Return the parts of a rock problem that the loss reads."""
    target = types.SimpleNamespace(min_start_mol=_MIN_START)
    penalty = types.SimpleNamespace(penalty_weight=_WEIGHT)
    return types.SimpleNamespace(
        space=_IdentitySpace(),
        surrogate=_StartIsTheInput(),
        h2_index=0,
        window=types.SimpleNamespace(start_index=0),
        objective=lambda h2: torch.zeros(h2.shape[0], dtype=h2.dtype),
        config=types.SimpleNamespace(target=target, constraints=penalty),
    )


def test_penalized_loss_adds_the_start_hinge():
    no_rules = constraints.LinearConstraints(
        (), np.zeros((0, 2)), np.zeros(0), np.ones(0)
    )
    # _PenalizedLoss is internal. The test uses it to check the formula.
    loss = rock_optimizer._PenalizedLoss(_stub_problem(), no_rules)
    z = torch.tensor([[0.2, 0.0], [0.05, 0.0], [0.0, 0.0], [-0.1, 0.0]])
    np.testing.assert_allclose(
        loss(z.double()).numpy(), [0.0, 25.0, 100.0, 400.0]
    )
