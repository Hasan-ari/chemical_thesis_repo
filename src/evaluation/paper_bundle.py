from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.evaluation.paper_runtime import configure_paper_runtime

configure_paper_runtime()

from src.data.loader import load_all_trajectories, load_time_axis
from src.data.preprocessing import DataPreprocessor
from src.data.split import split_trajectories
from src.evaluation.comprehensive import evaluate_all_trajectories, plot_predicted_vs_actual
from src.evaluation.paper_contracts import DEFAULT_PROFILE_FEATURES, METRIC_CONTRACT_VERSION
from src.evaluation.paper_metrics import (
    compute_initial_state_novelty,
    get_metric_contract,
    get_paper_figure_checklist,
    normalize_result_metrics,
    select_reference_trajectory_index,
)
from src.evaluation.paper_plots import (
    plot_ablation_and_sensitivity,
    plot_breakthrough_summary,
    plot_feature_time_heatmaps,
    plot_generalization_novelty,
    plot_representative_profile_panels,
)
from src.evaluation.plotting import plot_loss_curve
from src.models.lstm import PhreeqcLSTM
from src.training.config import ExperimentConfig
from src.training.trainer import setup_device


@dataclass(frozen=True)
class SavedExperimentEvaluation:
    config: ExperimentConfig
    result_payload: dict[str, Any]
    train_raw: np.ndarray
    test_raw: np.ndarray
    time_axis: np.ndarray
    evaluation: dict[str, Any]


def load_saved_experiment_evaluation(experiment_dir: Path) -> SavedExperimentEvaluation:
    config = ExperimentConfig.load(experiment_dir / "config.json")
    with open(experiment_dir / "results.json") as handle:
        result_payload = normalize_result_metrics(json.load(handle))

    raw_data = load_all_trajectories(Path(config.data_dir) / "output")
    time_axis = load_time_axis(Path(config.data_dir) / "output")
    train_raw, test_raw, _, _ = split_trajectories(
        raw_data, test_ratio=config.test_ratio, seed=config.seed
    )

    preprocessor = DataPreprocessor(log_cols=config.log_cols)
    train_norm = preprocessor.fit_transform(train_raw)
    test_norm = preprocessor.transform(test_raw)

    device = setup_device()
    model = PhreeqcLSTM(
        n_features=config.n_features,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    model.load_state_dict(
        torch.load(experiment_dir / "best_model.pt", weights_only=True, map_location=device)
    )
    model.to(device)

    evaluation = evaluate_all_trajectories(
        model,
        test_norm,
        test_raw,
        preprocessor,
        config.seq_len,
        device,
    )

    return SavedExperimentEvaluation(
        config=config,
        result_payload=result_payload,
        train_raw=train_raw,
        test_raw=test_raw,
        time_axis=time_axis,
        evaluation=evaluation,
    )


def generate_paper_figure_bundle(
    experiment_dir: str,
    output_dir: str | None = None,
    experiments_root: str | None = None,
    selected_features: Sequence[str] = DEFAULT_PROFILE_FEATURES,
) -> dict[str, Any]:
    experiment_path = Path(experiment_dir)
    figure_output_dir = Path(output_dir) if output_dir else experiment_path / "paper_figures"
    figure_output_dir.mkdir(parents=True, exist_ok=True)

    saved_eval = load_saved_experiment_evaluation(experiment_path)
    config = saved_eval.config
    evaluation = saved_eval.evaluation
    result_payload = saved_eval.result_payload

    generated_files: dict[str, str] = {}

    training_plot_path = figure_output_dir / "figure3_training_dynamics.png"
    plot_loss_curve(
        result_payload["loss_history"],
        save_path=training_plot_path,
        title=f"[{config.experiment_name}] Paper-Aligned Figure 3 - Training Dynamics",
    )
    generated_files["Figure 3"] = str(training_plot_path)

    profile_path = figure_output_dir / "figure4_5_representative_profiles.png"
    plot_representative_profile_panels(
        evaluation["all_pred_raw"],
        saved_eval.test_raw,
        evaluation["rmse_per_traj"],
        saved_eval.time_axis,
        config.seq_len,
        save_path=profile_path,
        selected_features=selected_features,
    )
    generated_files["Figures 4-5"] = str(profile_path)

    reference_idx = select_reference_trajectory_index(evaluation["rmse_per_traj"])
    heatmap_path = figure_output_dir / "figure6_truth_pred_error_heatmaps.png"
    plot_feature_time_heatmaps(
        truth=saved_eval.test_raw[reference_idx],
        pred=evaluation["all_pred_raw"][reference_idx],
        time_axis=saved_eval.time_axis,
        save_path=heatmap_path,
    )
    generated_files["Figure 6"] = str(heatmap_path)

    breakthrough_path = figure_output_dir / "figure7_breakthrough_summary.png"
    plot_breakthrough_summary(
        evaluation["all_pred_raw"],
        saved_eval.test_raw,
        evaluation["rmse_per_traj"],
        saved_eval.time_axis,
        config.seq_len,
        save_path=breakthrough_path,
        selected_features=selected_features,
    )
    generated_files["Figure 7"] = str(breakthrough_path)

    parity_path = figure_output_dir / "figure9_parity.png"
    plot_predicted_vs_actual(
        evaluation["all_pred_raw"],
        saved_eval.test_raw,
        config.seq_len,
        save_path=parity_path,
        title=f"[{config.experiment_name}] Paper-Aligned Figure 9 - Predicted vs Actual",
    )
    generated_files["Figure 9"] = str(parity_path)

    novelty_scores = compute_initial_state_novelty(saved_eval.train_raw, saved_eval.test_raw)
    generalization_path = figure_output_dir / "figure12_generalization_novelty.png"
    generalization_bins = plot_generalization_novelty(
        novelty_scores,
        evaluation["rmse_per_traj"],
        save_path=generalization_path,
    )
    generated_files["Figure 12"] = str(generalization_path)

    experiments_root_path = Path(experiments_root) if experiments_root else experiment_path.parent
    summary_path = experiments_root_path / "summary.json"
    if summary_path.exists():
        with open(summary_path) as handle:
            summary_results = json.load(handle)
        ablation_path = figure_output_dir / "figure10_11_ablation_sensitivity.png"
        plot_ablation_and_sensitivity(summary_results, experiments_root_path, save_path=ablation_path)
        generated_files["Figures 10-11"] = str(ablation_path)

    checklist_path = figure_output_dir / "paper_figure_checklist.json"
    with open(checklist_path, "w") as handle:
        json.dump(get_paper_figure_checklist(), handle, indent=2)

    metric_contract_path = figure_output_dir / "metric_contract.json"
    with open(metric_contract_path, "w") as handle:
        json.dump(get_metric_contract(), handle, indent=2)

    manifest = {
        "experiment": config.experiment_name,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "selected_features": list(selected_features),
        "reference_trajectory_index": reference_idx,
        "generated_files": generated_files,
        "paper_figure_checklist_path": str(checklist_path),
        "metric_contract_path": str(metric_contract_path),
        "canonical_metrics": {
            "nrmse_total": result_payload["nrmse_total"],
            "legacy_rmse_total_alias": result_payload["legacy_rmse_total_alias"],
            "overall_rmse_mean": float(np.mean(evaluation["rmse_per_traj"])),
        },
        "generalization_bins": generalization_bins,
    }
    manifest_path = figure_output_dir / "paper_figure_manifest.json"
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved: {manifest_path}")
    return manifest
