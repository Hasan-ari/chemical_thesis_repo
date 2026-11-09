"""Tests of conditional_input_optimization.optimization.objectives."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conditional_input_optimization.optimization import objectives

_WINDOW = objectives.TimeWindow(start_index=1, end_index=3)
_H2 = np.array(
    [
        [0.0, 2.0, 1.5, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.2, 1.5],
        [0.0, 0.0, 0.0, 0.5],
    ]
)


def test_relative_loss_on_numpy():
    loss = objectives.RelativeLoss(_WINDOW, eps=0.1)(_H2)
    np.testing.assert_allclose(loss, [0.5, 0.0, -0.5, -5.0])


def test_relative_loss_on_torch_matches_numpy():
    relative = objectives.RelativeLoss(_WINDOW, eps=0.1)
    tensor_loss = relative(torch.as_tensor(_H2))
    assert isinstance(tensor_loss, torch.Tensor)
    np.testing.assert_allclose(tensor_loss.numpy(), relative(_H2))


def test_absolute_loss():
    loss = objectives.AbsoluteLoss(_WINDOW)(_H2)
    np.testing.assert_allclose(loss, [1.0, 0.0, -0.5, -0.5])


def test_relative_loss_rejects_bad_eps():
    with pytest.raises(ValueError, match="eps"):
        objectives.RelativeLoss(_WINDOW, eps=0.0)


def test_time_window_from_days():
    window = objectives.TimeWindow.from_days(
        np.array([0.0, 0.6, 1.2, 180.0]), 0.6, 180.0
    )
    assert window == objectives.TimeWindow(1, 3)


def test_time_window_rejects_missing_time():
    with pytest.raises(ValueError, match="exactly one point"):
        objectives.TimeWindow.from_days(np.array([0.0, 1.0]), 0.5, 1.0)


@pytest.mark.parametrize(("start", "end"), [(-1, 2), (2, 2), (3, 1)])
def test_time_window_rejects_bad_order(start, end):
    with pytest.raises(ValueError, match="TimeWindow"):
        objectives.TimeWindow(start, end)


def test_build_objective():
    assert isinstance(
        objectives.build_objective("relative", _WINDOW, 0.1),
        objectives.RelativeLoss,
    )
    assert isinstance(
        objectives.build_objective("absolute", _WINDOW, 0.1),
        objectives.AbsoluteLoss,
    )
    with pytest.raises(ValueError, match="Unknown objective"):
        objectives.build_objective("ratio", _WINDOW, 0.1)
