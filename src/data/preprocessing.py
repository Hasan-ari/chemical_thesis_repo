from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.constants import LOG_TRANSFORM_COLS


class DataPreprocessor:
    """Per-variable log1p + StandardScaler normalization.

    Fit on training data only; transform both train and test.
    """

    def __init__(self, log_cols: tuple = LOG_TRANSFORM_COLS):
        self.log_cols = log_cols
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, data: np.ndarray) -> DataPreprocessor:
        """Fit on training data.

        Args:
            data: shape (n_traj, n_steps, n_feat) or (n_samples, n_feat)
        """
        flat = self._flatten(data)
        flat = self._apply_log(flat)
        self._scaler.fit(flat)
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply log1p + scaler. Must call fit() first."""
        assert self._fitted, "Call fit() before transform()"
        original_shape = data.shape
        flat = self._flatten(data)
        flat = self._apply_log(flat)
        flat = self._scaler.transform(flat)
        return flat.reshape(original_shape)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data_norm: np.ndarray) -> np.ndarray:
        """Inverse: un-scale, then expm1. Clamps negatives to 0."""
        original_shape = data_norm.shape
        flat = self._flatten(data_norm)
        flat = self._scaler.inverse_transform(flat)

        for col in self.log_cols:
            flat[:, col] = np.expm1(flat[:, col])

        # Non-negativity enforcement
        np.maximum(flat, 0, out=flat)
        return flat.reshape(original_shape)

    def _apply_log(self, data: np.ndarray) -> np.ndarray:
        data = data.copy()
        for col in self.log_cols:
            data[:, col] = np.log1p(np.maximum(data[:, col], 0))
        return data

    def _flatten(self, data: np.ndarray) -> np.ndarray:
        """Reshape 3D (n_traj, n_steps, n_feat) to 2D (n_samples, n_feat)."""
        if data.ndim == 3:
            n_traj, n_steps, n_feat = data.shape
            return data.reshape(-1, n_feat)
        return data.copy()
