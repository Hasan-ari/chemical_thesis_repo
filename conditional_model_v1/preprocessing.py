from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


class ConditionScaler:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, conditions: np.ndarray) -> ConditionScaler:
        self.scaler.fit(conditions)
        self.fitted = True
        return self

    def transform(self, conditions: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ConditionScaler must be fitted before transform")
        return self.scaler.transform(conditions).astype(np.float32)


class OutputScaler:
    def __init__(self, log_feature_indices: tuple[int, ...]) -> None:
        self.log_feature_indices = log_feature_indices
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, trajectories: np.ndarray) -> OutputScaler:
        self.scaler.fit(self._prepare(trajectories))
        self.fitted = True
        return self

    def transform(self, trajectories: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("OutputScaler must be fitted before transform")
        original_shape = trajectories.shape
        transformed = self.scaler.transform(self._prepare(trajectories))
        return transformed.reshape(original_shape).astype(np.float32)

    def inverse_transform(self, trajectories_norm: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("OutputScaler must be fitted before inverse_transform")
        original_shape = trajectories_norm.shape
        flat = trajectories_norm.reshape(-1, original_shape[-1])
        restored = self.scaler.inverse_transform(flat)
        for index in self.log_feature_indices:
            restored[:, index] = np.expm1(restored[:, index])
        return restored.reshape(original_shape)

    def _prepare(self, trajectories: np.ndarray) -> np.ndarray:
        flat = trajectories.reshape(-1, trajectories.shape[-1]).astype(np.float64).copy()
        for index in self.log_feature_indices:
            flat[:, index] = np.log1p(np.maximum(flat[:, index], 0.0))
        return flat


@dataclass
class PreprocessorBundle:
    condition_scaler: ConditionScaler
    output_scaler: OutputScaler
    time_mean: float
    time_std: float

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file_obj:
            pickle.dump(self, file_obj)

    @classmethod
    def load(cls, path: Path | str) -> PreprocessorBundle:
        with Path(path).open("rb") as file_obj:
            return pickle.load(file_obj)
