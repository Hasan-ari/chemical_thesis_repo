"""Put the frozen v2 LSTM behind one differentiable predict call.

This module imports conditional_model_v1. Reason: the v2 model class and the
v2 preprocessing objects are defined there, and the surrogate must use the
same code that trained the model.

Pattern: Adapter. FrozenSurrogate changes the v2 interface (normalized
(B, T, 58) tensors in, normalized (B, T, 26) tensors out) to a physical
interface: (B, 57) inputs in physical units in, (B, T, 26) outputs in
physical units out. All steps are torch operations, so the gradient flows
from the outputs back to the inputs.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import torch

from conditional_model_v1 import models, preprocessing

N_CONDITIONS = 57
N_OUTPUTS = 26
# The dtype of the physical inputs and outputs. v2 normalizes the inputs in
# float64 and then casts them to float32 (conditional_model_v1/cli/train.py).
# One v2 input has mean 1.18 and std 1.4e-4. A float32 input loses 1e-3 of
# that std, so the inputs must stay in float64 until they are normalized.
INPUT_DTYPE = torch.float64
_MATRIX_NDIM = 2


@dataclasses.dataclass(frozen=True, eq=False)
class SurrogateScaling:
    """The fixed scaling values of the v2 preprocessing.

    Attributes:
        condition_mean: The condition scaler mean, shape (57,).
        condition_scale: The condition scaler std, shape (57,). All values
            are positive.
        time_mean: The mean of the time axis, in days.
        time_std: The std of the time axis, in days. A value <= 0 is
            replaced by 1, as in conditional_model_v1.data.
        output_mean: The output scaler mean, shape (26,). For a log feature
            it is the mean of log1p(value).
        output_scale: The output scaler std, shape (26,).
        log_output_mask: True for each output that v2 scales as
            log1p(value), shape (26,).

    Raises:
        ValueError: If a shape is wrong, a value is not finite, or a
            condition scale is not positive.
    """

    condition_mean: np.ndarray
    condition_scale: np.ndarray
    time_mean: float
    time_std: float
    output_mean: np.ndarray
    output_scale: np.ndarray
    log_output_mask: np.ndarray

    def __post_init__(self) -> None:
        """Check the shapes and values, then make the arrays read-only."""
        expected = {
            "condition_mean": N_CONDITIONS,
            "condition_scale": N_CONDITIONS,
            "output_mean": N_OUTPUTS,
            "output_scale": N_OUTPUTS,
            "log_output_mask": N_OUTPUTS,
        }
        for name, size in expected.items():
            value = np.array(getattr(self, name), copy=True)
            if value.shape != (size,):
                raise ValueError(
                    f"{name} must have shape ({size},), got {value.shape}."
                )
            if value.dtype != bool and not np.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values.")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if not (self.condition_scale > 0).all():
            raise ValueError("condition_scale must be positive.")
        if not np.isfinite([self.time_mean, self.time_std]).all():
            raise ValueError("time_mean and time_std must be finite.")
        if self.time_std <= 0:
            object.__setattr__(self, "time_std", 1.0)

    @classmethod
    def from_bundle(
        cls, bundle: preprocessing.PreprocessorBundle
    ) -> SurrogateScaling:
        """Read the scaling arrays from a v2 PreprocessorBundle.

        Args:
            bundle: The loaded v2 bundle.

        Returns:
            The scaling values. Only the arrays mean_ and scale_ are read.
            The scikit-learn transform code is not used.
        """
        log_mask = np.zeros(N_OUTPUTS, dtype=bool)
        log_mask[list(bundle.output_scaler.log_feature_indices)] = True
        return cls(
            condition_mean=bundle.condition_scaler.scaler.mean_,
            condition_scale=bundle.condition_scaler.scaler.scale_,
            time_mean=float(bundle.time_mean),
            time_std=float(bundle.time_std),
            output_mean=bundle.output_scaler.scaler.mean_,
            output_scale=bundle.output_scaler.scaler.scale_,
            log_output_mask=log_mask,
        )

    def normalize_conditions(self, conditions: np.ndarray) -> np.ndarray:
        """Return (conditions - mean) / scale, as the v2 condition scaler.

        Args:
            conditions: Physical inputs, shape (N, 57).

        Returns:
            The normalized inputs, shape (N, 57), float64.
        """
        return (conditions - self.condition_mean) / self.condition_scale


class FrozenSurrogate:
    """Predict the 26 output trajectories from physical inputs.

    Pattern: Adapter (see the module docstring).

    The model is in eval mode and its parameters do not need a gradient.
    The gradient flows only to the input tensor. The scaling steps run in
    float64 (INPUT_DTYPE). Only the LSTM runs in the dtype of the model.

    Time: one LSTM pass, O(B * T * H^2), with H = hidden size.
    Memory: O(B * T * H) for the activations that backward needs.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scaling: SurrogateScaling,
        time_days: np.ndarray,
    ) -> None:
        """Freeze the model and keep the scaling as tensors.

        Args:
            model: A module that maps (B, T, 58) to (B, T, 26).
            scaling: The v2 scaling values.
            time_days: The time axis, shape (T,), in days.

        Raises:
            ValueError: If time_days is not 1D and finite.
        """
        time_days = np.asarray(time_days, dtype=np.float64)
        if time_days.ndim != 1 or not np.isfinite(time_days).all():
            raise ValueError("time_days must be a finite 1D array.")
        parameter = next(model.parameters())
        self._model = model.eval().requires_grad_(False)
        self._model_dtype = parameter.dtype
        self._device = parameter.device
        self.scaling = scaling
        self.time_days = time_days.copy()
        self.time_days.flags.writeable = False
        time_norm = (time_days - scaling.time_mean) / scaling.time_std
        self._time_norm = self._tensor(time_norm).to(self._model_dtype)
        self._cond_mean = self._tensor(scaling.condition_mean)
        self._cond_scale = self._tensor(scaling.condition_scale)
        self._out_mean = self._tensor(scaling.output_mean)
        self._out_scale = self._tensor(scaling.output_scale)
        self._log_mask = torch.tensor(
            scaling.log_output_mask, device=self._device
        )

    @property
    def dtype(self) -> torch.dtype:
        """The dtype that predict expects for its input (INPUT_DTYPE)."""
        return INPUT_DTYPE

    @property
    def device(self) -> torch.device:
        """The device of the model."""
        return self._device

    def predict(self, conditions: torch.Tensor) -> torch.Tensor:
        """Predict the physical output trajectories.

        Args:
            conditions: Physical inputs, shape (B, 57), dtype INPUT_DTYPE,
                on the device of the model.

        Returns:
            The physical outputs, shape (B, T, 26), dtype INPUT_DTYPE. The
            log features are expm1 of the inverse-scaled value, as in v2
            OutputScaler.

        Raises:
            ValueError: If the shape, the dtype, or the device is wrong.
        """
        self._check_input(conditions)
        batch = conditions.shape[0]
        steps = self._time_norm.shape[0]
        cond_norm = (conditions - self._cond_mean) / self._cond_scale
        cond_norm = cond_norm.to(self._model_dtype)
        cond_block = cond_norm[:, None, :].expand(batch, steps, N_CONDITIONS)
        time_block = self._time_norm[None, :, None].expand(batch, steps, 1)
        model_input = torch.cat([cond_block, time_block], dim=-1)
        # cuDNN RNN backward works only in training mode. The model is in
        # eval mode, so turn cuDNN off. This has no effect on the CPU.
        with torch.backends.cudnn.flags(enabled=False):
            normalized = self._model(model_input)
        scaled = normalized.to(INPUT_DTYPE) * self._out_scale + self._out_mean
        # The inner where keeps expm1 away from large non-log values. Without
        # it, an inf in the unused branch gives a NaN gradient.
        safe = torch.where(self._log_mask, scaled, torch.zeros_like(scaled))
        return torch.where(self._log_mask, torch.expm1(safe), scaled)

    def _tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.tensor(values, dtype=INPUT_DTYPE, device=self._device)

    def _check_input(self, conditions: torch.Tensor) -> None:
        if (
            conditions.ndim != _MATRIX_NDIM
            or conditions.shape[1] != N_CONDITIONS
        ):
            raise ValueError(
                f"conditions must have shape (B, {N_CONDITIONS}), "
                f"got {tuple(conditions.shape)}."
            )
        if conditions.dtype != INPUT_DTYPE:
            raise ValueError(
                f"conditions must have dtype {INPUT_DTYPE}, "
                f"got {conditions.dtype}."
            )
        if conditions.device != self._device:
            raise ValueError(
                f"conditions must be on {self._device}, "
                f"got {conditions.device}."
            )


@dataclasses.dataclass(frozen=True)
class ModelShape:
    """The constructor values of the v2 ConditionTimeLSTM.

    Raises:
        ValueError: If a value is not positive.
    """

    hidden_size: int
    num_layers: int

    def __post_init__(self) -> None:
        """Check that the values are positive."""
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive.")


def load_surrogate(
    checkpoint_path: pathlib.Path,
    bundle_path: pathlib.Path,
    shape: ModelShape,
    time_days: np.ndarray,
    device: torch.device,
) -> FrozenSurrogate:
    """Load the v2 checkpoint and preprocessing bundle from disk.

    WARNING: The bundle is a pickle file. Load only a file that you trust.

    Args:
        checkpoint_path: The v2 best.pt file. It holds a state_dict.
        bundle_path: The v2 preprocessors.pkl file.
        shape: The hidden size and the number of LSTM layers.
        time_days: The time axis, shape (T,), in days.
        device: The device for the model.

    Returns:
        The frozen surrogate in float32.

    Raises:
        FileNotFoundError: If a file does not exist.
    """
    for path in (checkpoint_path, bundle_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {path}. See section 9 of "
                "conditional_input_optimization/docs/optimization_v1.md."
            )
    model = models.ConditionTimeLSTM(
        input_size=N_CONDITIONS + 1,
        output_size=N_OUTPUTS,
        hidden_size=shape.hidden_size,
        num_layers=shape.num_layers,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    bundle = preprocessing.PreprocessorBundle.load(bundle_path)
    scaling = SurrogateScaling.from_bundle(bundle)
    return FrozenSurrogate(model.to(device), scaling, time_days)
