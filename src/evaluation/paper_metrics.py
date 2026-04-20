from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np

from src.evaluation.paper_contracts import (
    METRIC_CONTRACT,
    METRIC_CONTRACT_VERSION,
    PAPER_FIGURE_SPECS,
    GeneralizationBin,
)


def get_paper_figure_checklist() -> list[dict[str, str]]:
    return [asdict(spec) for spec in PAPER_FIGURE_SPECS]


def get_metric_contract() -> list[dict[str, str]]:
    return [asdict(entry) for entry in METRIC_CONTRACT]


def normalize_result_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    canonical_nrmse = result.get("nrmse_total", result.get("rmse_total"))
    if canonical_nrmse is None:
        raise KeyError("Expected 'nrmse_total' or legacy 'rmse_total' in results payload.")

    normalized = dict(result)
    normalized["nrmse_total"] = float(canonical_nrmse)
    normalized["legacy_rmse_total_alias"] = float(result.get("rmse_total", canonical_nrmse))
    normalized["metric_contract_version"] = METRIC_CONTRACT_VERSION
    return normalized


def compute_initial_state_novelty(
    train_raw: np.ndarray,
    test_raw: np.ndarray,
) -> np.ndarray:
    train_initial = np.asarray(train_raw)[:, 0, :]
    test_initial = np.asarray(test_raw)[:, 0, :]

    train_mean = np.mean(train_initial, axis=0)
    train_std = np.std(train_initial, axis=0)
    train_std = np.where(train_std < 1e-12, 1.0, train_std)

    z_scores = (test_initial - train_mean) / train_std
    return np.sqrt(np.mean(z_scores ** 2, axis=1))


def summarize_generalization_bins(
    novelty_scores: np.ndarray,
    rmse_per_traj: np.ndarray,
    n_bins: int = 3,
) -> list[dict[str, float]]:
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")

    novelty_scores = np.asarray(novelty_scores, dtype=float)
    rmse_per_traj = np.asarray(rmse_per_traj, dtype=float)
    if novelty_scores.shape != rmse_per_traj.shape:
        raise ValueError("novelty_scores and rmse_per_traj must have matching shapes.")

    if n_bins == 3:
        labels = ["low_novelty", "mid_novelty", "high_novelty"]
    else:
        labels = [f"novelty_bin_{idx + 1}" for idx in range(n_bins)]

    grouped_indices = np.array_split(np.argsort(novelty_scores), n_bins)
    summaries: list[dict[str, float]] = []

    for label, indices in zip(labels, grouped_indices):
        if len(indices) == 0:
            continue

        novelty_slice = novelty_scores[indices]
        rmse_slice = rmse_per_traj[indices]
        summary = GeneralizationBin(
            bin_label=label,
            count=int(len(indices)),
            mean_novelty=float(np.mean(novelty_slice)),
            min_novelty=float(np.min(novelty_slice)),
            max_novelty=float(np.max(novelty_slice)),
            mean_rmse=float(np.mean(rmse_slice)),
            std_rmse=float(np.std(rmse_slice)),
        )
        summaries.append(asdict(summary))

    return summaries


def selected_feature_indices(
    feature_names: Sequence[str],
    selected_features: Sequence[str],
) -> list[int]:
    return [feature_names.index(name) for name in selected_features]


def percentile_indices(values: np.ndarray, percentiles: Sequence[int]) -> list[int]:
    indices: list[int] = []
    for percentile in percentiles:
        target = np.percentile(values, percentile)
        indices.append(int(np.argmin(np.abs(values - target))))
    return indices


def select_reference_trajectory_index(rmse_per_traj: np.ndarray) -> int:
    return percentile_indices(np.asarray(rmse_per_traj, dtype=float), (50,))[0]
