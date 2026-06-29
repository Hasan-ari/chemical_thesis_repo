from __future__ import annotations

from typing import Any

import numpy as np


def regression_metrics_original_scale(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_features: tuple[str, ...],
) -> dict[str, Any]:
    """Compute report metrics after inverse-transforming to chemistry units."""
    error = y_pred - y_true
    mae = np.mean(np.abs(error), axis=(0, 1))
    rmse = np.sqrt(np.mean(error**2, axis=(0, 1)))
    final_error = error[:, -1, :]
    final_mae = np.mean(np.abs(final_error), axis=0)
    final_rmse = np.sqrt(np.mean(final_error**2, axis=0))
    return {
        "rmse_mean_original": float(np.mean(rmse)),
        "mae_mean_original": float(np.mean(mae)),
        "rmse_per_feature_original": {
            feature: float(value) for feature, value in zip(output_features, rmse, strict=True)
        },
        "mae_per_feature_original": {
            feature: float(value) for feature, value in zip(output_features, mae, strict=True)
        },
        "final_rmse_per_feature_original": {
            feature: float(value) for feature, value in zip(output_features, final_rmse, strict=True)
        },
        "final_mae_per_feature_original": {
            feature: float(value) for feature, value in zip(output_features, final_mae, strict=True)
        },
    }
