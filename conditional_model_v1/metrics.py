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


def evaluate_by_rock(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    rocks: np.ndarray,
    run_ids: list[str],
    output_features: tuple[str, ...],
    rock_order: tuple[str, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    """Compute per-rock RMSE and best/worst/mean overview trajectories."""
    if not (
        y_true.shape == y_pred.shape == y_true_norm.shape == y_pred_norm.shape
        and y_true.ndim == 3
    ):
        raise ValueError("Evaluation arrays must share one 3D shape")
    n_runs, _n_timesteps, n_outputs = y_true.shape
    if len(rocks) != n_runs or len(run_ids) != n_runs:
        raise ValueError("Evaluation metadata must align with the run axis")
    if len(output_features) != n_outputs:
        raise ValueError("Output feature names must align with the feature axis")

    per_rock: dict[str, Any] = {}
    feature_rows: list[dict[str, Any]] = []
    overviews: dict[str, dict[str, dict[str, Any]]] = {}
    for rock in rock_order:
        rock_indices = np.flatnonzero(rocks == rock)
        if len(rock_indices) == 0:
            raise ValueError(f"Evaluation split is missing configured rock: {rock}")

        original_error = (
            y_pred[rock_indices].astype(np.float64)
            - y_true[rock_indices].astype(np.float64)
        )
        original_rmse = np.sqrt(np.mean(original_error**2, axis=(0, 1)))
        del original_error
        normalized_error = (
            y_pred_norm[rock_indices].astype(np.float64)
            - y_true_norm[rock_indices].astype(np.float64)
        )
        normalized_rmse = np.sqrt(np.mean(normalized_error**2, axis=(0, 1)))
        run_rmse_normalized = np.sqrt(np.mean(normalized_error**2, axis=(1, 2)))
        del normalized_error

        best_local = int(np.argmin(run_rmse_normalized))
        worst_local = int(np.argmax(run_rmse_normalized))
        best_index = int(rock_indices[best_local])
        worst_index = int(rock_indices[worst_local])
        per_rock[rock] = {
            "n_runs": int(len(rock_indices)),
            "rmse_mean_original": float(np.mean(original_rmse)),
            "rmse_mean_normalized": float(np.mean(normalized_rmse)),
            "rmse_per_feature_original": {
                feature: float(value)
                for feature, value in zip(
                    output_features,
                    original_rmse,
                    strict=True,
                )
            },
            "rmse_per_feature_normalized": {
                feature: float(value)
                for feature, value in zip(
                    output_features,
                    normalized_rmse,
                    strict=True,
                )
            },
            "best_run_id": run_ids[best_index],
            "best_run_rmse_normalized": float(run_rmse_normalized[best_local]),
            "worst_run_id": run_ids[worst_index],
            "worst_run_rmse_normalized": float(run_rmse_normalized[worst_local]),
        }
        feature_rows.extend(
            {
                "rock": rock,
                "feature": feature,
                "n_runs": int(len(rock_indices)),
                "rmse_original": float(original_value),
                "rmse_normalized": float(normalized_value),
            }
            for feature, original_value, normalized_value in zip(
                output_features,
                original_rmse,
                normalized_rmse,
                strict=True,
            )
        )
        overviews[rock] = {
            "best": {
                "y_true": y_true[best_index],
                "y_pred": y_pred[best_index],
                "label": f"best - {run_ids[best_index]}",
            },
            "worst": {
                "y_true": y_true[worst_index],
                "y_pred": y_pred[worst_index],
                "label": f"worst - {run_ids[worst_index]}",
            },
            "mean": {
                "y_true": np.mean(y_true[rock_indices], axis=0),
                "y_pred": np.mean(y_pred[rock_indices], axis=0),
                "label": f"mean - {rock}",
            },
        }
    return per_rock, feature_rows, overviews
