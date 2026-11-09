"""Tests of the surrogate checks, the distances, and the batched predict."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conditional_input_optimization.optimization import diagnostics, predict
from conditional_input_optimization.tests import optimization_fakes as fakes

_RULE = diagnostics.AcceptanceRule(min_spearman=0.7, max_mae_over_std=0.5)


def test_regression_metrics_of_a_perfect_prediction():
    real = np.array([3.0, 1.0, 2.0, 5.0])
    metrics = diagnostics.regression_metrics(real.copy(), real)
    assert metrics == {"mae": 0.0, "r2": 1.0, "spearman": 1.0}


def test_regression_metrics_values():
    predicted = np.array([1.0, 2.0, 4.0, 3.0])
    real = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = diagnostics.regression_metrics(predicted, real)
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["r2"] == pytest.approx(1.0 - 2.0 / 5.0)
    assert metrics["spearman"] == pytest.approx(0.8)


def test_regression_metrics_r2_is_nan_for_constant_real_values():
    metrics = diagnostics.regression_metrics(
        np.array([1.0, 2.0]), np.array([1.0, 1.0])
    )
    assert np.isnan(metrics["r2"])
    assert np.isnan(metrics["spearman"])


def test_constant_prediction_is_not_resolvable():
    check = diagnostics.surrogate_check(
        np.zeros(4), np.arange(4.0), n_top=2, acceptance=_RULE
    )
    assert np.isnan(check["spearman"])
    assert not check["resolvable"]


@pytest.mark.parametrize(
    ("predicted", "real", "message"),
    [
        (np.zeros(3), np.zeros(4), "equal shapes"),
        (np.zeros(1), np.zeros(1), "at least 2"),
        (np.array([0.0, np.nan]), np.zeros(2), "finite"),
    ],
)
def test_regression_metrics_rejects_bad_inputs(predicted, real, message):
    with pytest.raises(ValueError, match=message):
        diagnostics.regression_metrics(predicted, real)


def test_surrogate_check_of_a_good_ranking():
    real = np.linspace(-1.0, 1.0, 20)
    predicted = real + 0.01
    check = diagnostics.surrogate_check(
        predicted, real, n_top=3, acceptance=_RULE
    )
    assert check["n"] == 20
    assert check["top_decile_precision"] == 1.0
    assert check["optimism_gap_top"] == pytest.approx(-0.01)
    assert check["share_predicted_negative"] == 0.5
    assert check["abs_error_p95"] == pytest.approx(0.01)
    assert check["resolvable"]


def test_surrogate_check_flags_a_bad_ranking():
    real = np.linspace(-1.0, 1.0, 20)
    check = diagnostics.surrogate_check(
        real[::-1].copy(), real, n_top=50, acceptance=_RULE
    )
    assert check["spearman"] == pytest.approx(-1.0)
    assert check["top_decile_precision"] == 0.0
    assert not check["resolvable"]


def test_surrogate_check_rejects_bad_n_top():
    with pytest.raises(ValueError, match="n_top"):
        diagnostics.surrogate_check(
            np.zeros(3), np.arange(3.0), n_top=0, acceptance=_RULE
        )


@pytest.mark.parametrize(("spearman", "ratio"), [(1.5, 0.5), (0.5, 0.0)])
def test_acceptance_rule_rejects_bad_limits(spearman, ratio):
    with pytest.raises(ValueError):
        diagnostics.AcceptanceRule(spearman, ratio)


@pytest.mark.parametrize("chunk_size", [1, 3, 100])
def test_nearest_distances_match_brute_force(chunk_size):
    rng = np.random.default_rng(0)
    points, reference = rng.normal(size=(7, 4)), rng.normal(size=(11, 4))
    expected = np.linalg.norm(
        points[:, None, :] - reference[None, :, :], axis=2
    ).min(axis=1)
    actual = diagnostics.nearest_distances(
        points, reference, chunk_size=chunk_size
    )
    np.testing.assert_allclose(actual, expected)


def test_reference_nn_distances_skip_the_point_itself():
    reference = np.array([[0.0], [1.0], [3.0], [7.0]])
    distances = diagnostics.reference_nn_distances(
        reference, sample_size=10, seed=0, chunk_size=2
    )
    np.testing.assert_allclose(np.sort(distances), [1.0, 1.0, 2.0, 4.0])


def test_reference_nn_distances_subsample():
    reference = np.random.default_rng(1).normal(size=(50, 3))
    first = diagnostics.reference_nn_distances(
        reference, sample_size=5, seed=3, chunk_size=2
    )
    second = diagnostics.reference_nn_distances(
        reference, sample_size=5, seed=3, chunk_size=4
    )
    assert first.shape == (5,)
    np.testing.assert_allclose(first, second)


def test_distance_functions_reject_bad_inputs():
    with pytest.raises(ValueError, match="m >= 2"):
        diagnostics.reference_nn_distances(
            np.zeros((1, 2)), sample_size=1, seed=0, chunk_size=1
        )
    with pytest.raises(ValueError, match="shape"):
        diagnostics.nearest_distances(
            np.zeros((2, 3)), np.zeros((2, 2)), chunk_size=1
        )
    with pytest.raises(ValueError, match="chunk_size"):
        diagnostics.nearest_distances(
            np.zeros((2, 2)), np.zeros((2, 2)), chunk_size=0
        )
    with pytest.raises(ValueError, match="empty"):
        diagnostics.distance_percentile(np.zeros(2), np.zeros(0))


def test_distance_percentile():
    percentiles = diagnostics.distance_percentile(
        np.array([0.0, 1.0, 2.5, 10.0]), np.array([1.0, 2.0, 3.0, 4.0])
    )
    np.testing.assert_allclose(percentiles, [0.0, 25.0, 50.0, 100.0])


def test_scaler_mean_gap():
    conditions = fakes.fake_conditions(20, seed=0)
    scaling = fakes.tiny_surrogate(conditions).surrogate.scaling
    assert diagnostics.scaler_mean_gap(conditions, scaling) < 1e-12
    shifted = conditions.copy()
    shifted[:, 0] += scaling.condition_scale[0]
    assert diagnostics.scaler_mean_gap(shifted, scaling) == pytest.approx(1.0)


def test_predict_output_batches_equal_one_call():
    conditions = fakes.fake_conditions(10, seed=1)
    tiny = fakes.tiny_surrogate(conditions)
    batched = predict.predict_output(
        tiny.surrogate, conditions, output_index=fakes.H2_INDEX, batch_size=3
    )
    with torch.no_grad():
        expected = tiny.surrogate.predict(
            torch.as_tensor(conditions, dtype=tiny.surrogate.dtype)
        )[:, :, fakes.H2_INDEX]
    assert batched.dtype == np.float64
    assert batched.shape == (10, len(fakes.TIME_DAYS))
    np.testing.assert_allclose(batched, expected.numpy(), rtol=1e-6)


@pytest.mark.parametrize(
    ("output_index", "batch_size", "message"),
    [(5, 0, "batch_size"), (26, 2, "out of range"), (-1, 2, "out of range")],
)
def test_predict_output_rejects_bad_arguments(
    output_index, batch_size, message
):
    conditions = fakes.fake_conditions(4, seed=2)
    tiny = fakes.tiny_surrogate(conditions)
    with pytest.raises(ValueError, match=message):
        predict.predict_output(
            tiny.surrogate,
            conditions,
            output_index=output_index,
            batch_size=batch_size,
        )
