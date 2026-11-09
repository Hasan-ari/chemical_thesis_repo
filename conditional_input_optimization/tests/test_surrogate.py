"""Tests of conditional_input_optimization.optimization.surrogate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conditional_input_optimization.optimization import surrogate
from conditional_input_optimization.tests import optimization_fakes as fakes
from conditional_model_v1 import data as v2_data


def _v2_numpy_predict(tiny, conditions):
    """Predict with the v2 numpy path: scaler, tensor builder, inverse."""
    bundle = tiny.bundle
    model_input = v2_data.build_condition_time_tensor(
        bundle.condition_scaler.transform(conditions),
        fakes.TIME_DAYS,
        time_mean=bundle.time_mean,
        time_std=bundle.time_std,
    )
    with torch.no_grad():
        normalized = tiny.model(torch.from_numpy(model_input)).numpy()
    return bundle.output_scaler.inverse_transform(normalized)


def _predict(tiny, conditions):
    with torch.no_grad():
        return tiny.surrogate.predict(
            torch.as_tensor(conditions, dtype=tiny.surrogate.dtype)
        ).numpy()


def test_predict_matches_v2_numpy_path():
    conditions = fakes.fake_conditions(12, seed=0)
    tiny = fakes.tiny_surrogate(conditions)
    actual = _predict(tiny, conditions)
    assert actual.shape == (12, len(fakes.TIME_DAYS), surrogate.N_OUTPUTS)
    expected = _v2_numpy_predict(tiny, conditions)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_predict_keeps_precision_of_a_narrow_input():
    """A float32 input loses the digits that a narrow input needs.

    The real v2 scaler has an input with mean 1.18 and std 1.4e-4. In
    float32, 1.18 has a step of 1.2e-7, which is 1e-3 of the std.
    """
    rng = np.random.default_rng(7)
    conditions = fakes.fake_conditions(12, seed=0)
    conditions[:, 12] = 1.18409 + rng.normal(0.0, 1.36e-4, size=12)
    tiny = fakes.tiny_surrogate(conditions)
    expected = _v2_numpy_predict(tiny, conditions)
    np.testing.assert_allclose(
        _predict(tiny, conditions), expected, rtol=1e-6, atol=1e-7
    )


def test_gradient_matches_finite_differences_in_float64():
    conditions = fakes.fake_conditions(6, seed=1)
    tiny = fakes.tiny_surrogate(conditions, dtype=torch.float64)
    inputs = torch.tensor(conditions[:2], requires_grad=True)

    def h2(values):
        return tiny.surrogate.predict(values)[:, :, fakes.H2_INDEX]

    assert torch.autograd.gradcheck(h2, (inputs,), eps=1e-6, atol=1e-5)


def test_model_parameters_are_frozen():
    tiny = fakes.tiny_surrogate(fakes.fake_conditions(6, seed=2))
    assert not any(p.requires_grad for p in tiny.model.parameters())
    assert not tiny.model.training


def test_scaling_from_bundle_marks_log_outputs():
    tiny = fakes.tiny_surrogate(fakes.fake_conditions(6, seed=3))
    mask = tiny.surrogate.scaling.log_output_mask
    assert mask[fakes.H2_INDEX]
    assert mask.sum() == len(fakes.LOG_INDICES)


@pytest.mark.parametrize(
    "conditions",
    [
        torch.zeros(2, 56),
        torch.zeros(57),
        torch.zeros(2, 57, dtype=torch.float32),
    ],
)
def test_predict_rejects_wrong_input(conditions):
    tiny = fakes.tiny_surrogate(fakes.fake_conditions(6, seed=4))
    with pytest.raises(ValueError, match="conditions must"):
        tiny.surrogate.predict(conditions)


def test_scaling_rejects_bad_values():
    scaling = fakes.tiny_surrogate(
        fakes.fake_conditions(6, seed=5)
    ).surrogate.scaling
    values = {
        "condition_mean": scaling.condition_mean,
        "condition_scale": scaling.condition_scale,
        "time_mean": 0.0,
        "time_std": 1.0,
        "output_mean": scaling.output_mean,
        "output_scale": scaling.output_scale,
        "log_output_mask": scaling.log_output_mask,
    }
    with pytest.raises(ValueError, match="shape"):
        surrogate.SurrogateScaling(**{**values, "output_mean": np.zeros(3)})
    with pytest.raises(ValueError, match="positive"):
        surrogate.SurrogateScaling(
            **{**values, "condition_scale": np.zeros(57)}
        )
    zero_std = surrogate.SurrogateScaling(**{**values, "time_std": 0.0})
    assert zero_std.time_std == 1.0


def test_load_surrogate_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        surrogate.load_surrogate(
            tmp_path / "best.pt",
            tmp_path / "preprocessors.pkl",
            surrogate.ModelShape(8, 1),
            fakes.TIME_DAYS,
            torch.device("cpu"),
        )


def test_load_surrogate_reads_saved_files(tmp_path):
    conditions = fakes.fake_conditions(6, seed=6)
    tiny = fakes.tiny_surrogate(conditions)
    torch.save(tiny.model.state_dict(), tmp_path / "best.pt")
    tiny.bundle.save(tmp_path / "preprocessors.pkl")
    loaded = surrogate.load_surrogate(
        tmp_path / "best.pt",
        tmp_path / "preprocessors.pkl",
        surrogate.ModelShape(fakes.HIDDEN_SIZE, fakes.NUM_LAYERS),
        fakes.TIME_DAYS,
        torch.device("cpu"),
    )
    inputs = torch.as_tensor(conditions, dtype=loaded.dtype)
    with torch.no_grad():
        torch.testing.assert_close(
            loaded.predict(inputs), tiny.surrogate.predict(inputs)
        )
