from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.evaluation.paper_runtime import configure_paper_runtime

configure_paper_runtime()

from src.data.constants import FEATURE_NAMES
from src.data.loader import load_all_trajectories, load_time_axis
from src.data.preprocessing import DataPreprocessor
from src.data.split import split_trajectories
from src.evaluation.comprehensive import evaluate_all_trajectories, plot_predicted_vs_actual
from src.evaluation.paper_contracts import (
    DEFAULT_PROFILE_FEATURES,
    METRIC_CONTRACT_VERSION,
    PAPER_FIGURE_SPECS,
)
from src.evaluation.paper_metrics import (
    compute_initial_state_novelty,
    get_metric_contract,
    get_paper_figure_checklist,
    normalize_result_metrics,
    select_reference_trajectory_index,
    summarize_parity_quality,
)
from src.evaluation.paper_plots import (
    plot_breakthrough_summary,
    plot_feature_time_heatmaps,
    plot_generalization_novelty,
    plot_lstm_architecture_schematic,
    plot_metric_sensitivity_summary,
    plot_novelty_profile_panels,
    plot_representative_profile_panels,
    plot_workflow_schematic,
)
from src.evaluation.plotting import plot_loss_curve
from src.evaluation.paper_writeups import (
    cleanup_legacy_figure_outputs,
    get_full_requested_figure_artifacts,
    get_requested_figure_artifacts,
    write_figure_markdown_files,
)
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


FULL_FIGURE_SPEC_OVERRIDES: dict[str, dict[str, str]] = {
    "Figure 1": {
        "repo_status": "adapted",
        "repo_equivalent": "Repo-local PHREEQC -> LSTM evaluation workflow schematic",
        "notes": "This is a workflow analog, not the paper's pyrite-column transport schematic.",
    },
    "Figure 2": {
        "repo_status": "adapted",
        "repo_equivalent": "Autoregressive LSTM architecture schematic",
        "notes": "This documents the current LSTM and explicitly excludes HRTNet PDE residual claims.",
    },
    "Figure 10": {
        "repo_status": "unsupported",
        "repo_equivalent": "No repo-native equivalent without spatial grids or retrained temporal-density ablation experiments",
        "notes": "Sequence length is not the same as the paper's spatial/temporal training-data density study.",
    },
}


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def _canonical_metrics(
    result_payload: dict[str, Any],
    rmse_per_traj: np.ndarray,
) -> dict[str, float]:
    return {
        "nrmse_total": result_payload["nrmse_total"],
        "legacy_rmse_total_alias": result_payload["legacy_rmse_total_alias"],
        "overall_rmse_mean": float(np.mean(rmse_per_traj)),
    }


def _experiment_metric_row(
    experiment_name: str,
    seq_len: int,
    hidden_size: int,
    result_payload: Mapping[str, Any],
    overall_rmse_mean: float,
) -> dict[str, Any]:
    return {
        "experiment": experiment_name,
        "seq_len": int(seq_len),
        "hidden_size": int(hidden_size),
        "nrmse_total": float(result_payload["nrmse_total"]),
        "overall_rmse_mean": float(overall_rmse_mean),
        "training_time": float(result_payload.get("training_time", 0.0)),
    }


def _full_paper_figure_checklist() -> list[dict[str, str]]:
    checklist = []
    for spec in PAPER_FIGURE_SPECS:
        entry = {
            "paper_figure": spec.paper_figure,
            "repo_status": spec.repo_status,
            "paper_role": spec.paper_role,
            "repo_equivalent": spec.repo_equivalent,
            "notes": spec.notes,
        }
        entry.update(FULL_FIGURE_SPEC_OVERRIDES.get(spec.paper_figure, {}))
        checklist.append(entry)
    return checklist


def collect_experiment_metric_rows(experiments_root: str | Path) -> list[dict[str, Any]]:
    root = Path(experiments_root)
    rows: list[dict[str, Any]] = []

    for experiment_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        results_path = experiment_dir / "results.json"
        stats_path = experiment_dir / "comprehensive" / "comprehensive_stats.json"
        if not results_path.exists() or not stats_path.exists():
            continue

        with open(results_path) as handle:
            result_payload = normalize_result_metrics(json.load(handle))
        with open(stats_path) as handle:
            stats_payload = json.load(handle)

        rows.append(
            _experiment_metric_row(
                experiment_name=experiment_dir.name,
                seq_len=result_payload["seq_len"],
                hidden_size=result_payload["hidden_size"],
                result_payload=result_payload,
                overall_rmse_mean=stats_payload["overall_rmse"]["mean"],
            )
        )

    return sorted(rows, key=lambda row: (row["seq_len"], row["hidden_size"], row["experiment"]))


def build_paper_figure_manifest(
    experiment_name: str,
    selected_features: list[str],
    canonical_metrics: dict[str, float],
    figure_artifacts: dict[str, dict[str, Any]],
    checklist_path: str,
    metric_contract_path: str,
    generalization_bins: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "experiment": experiment_name,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "selected_features": selected_features,
        "figures": figure_artifacts,
        "paper_figure_checklist_path": checklist_path,
        "metric_contract_path": metric_contract_path,
        "canonical_metrics": canonical_metrics,
        "generalization_bins": generalization_bins,
    }


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
    cleanup_legacy_figure_outputs(figure_output_dir)
    selected_features = list(selected_features)

    saved_eval = load_saved_experiment_evaluation(experiment_path)
    config = saved_eval.config
    evaluation = saved_eval.evaluation
    result_payload = saved_eval.result_payload
    novelty_scores = compute_initial_state_novelty(saved_eval.train_raw, saved_eval.test_raw)
    artifact_specs = {
        item["paper_figure"]: item for item in get_requested_figure_artifacts()
    }

    figure_artifacts: dict[str, dict[str, Any]] = {
        figure_name: {
            "image": spec["image_filename"],
        }
        for figure_name, spec in artifact_specs.items()
    }

    training_plot_path = figure_output_dir / artifact_specs["Figure 3"]["image_filename"]
    plot_loss_curve(
        result_payload["loss_history"],
        save_path=training_plot_path,
        title=f"[{config.experiment_name}] Paper-Aligned Figure 3 - Training Dynamics",
    )
    figure_artifacts["Figure 3"]["image"] = str(training_plot_path)

    figure4_path = figure_output_dir / artifact_specs["Figure 4"]["image_filename"]
    figure4_indices = plot_representative_profile_panels(
        evaluation["all_pred_raw"],
        saved_eval.test_raw,
        evaluation["rmse_per_traj"],
        saved_eval.time_axis,
        config.seq_len,
        save_path=figure4_path,
        selected_features=selected_features,
    )
    figure_artifacts["Figure 4"]["image"] = str(figure4_path)
    figure_artifacts["Figure 4"]["representative_indices"] = figure4_indices

    reference_idx = select_reference_trajectory_index(evaluation["rmse_per_traj"])
    figure5_path = figure_output_dir / artifact_specs["Figure 5"]["image_filename"]
    figure5_indices = plot_novelty_profile_panels(
        evaluation["all_pred_raw"],
        saved_eval.test_raw,
        novelty_scores,
        saved_eval.time_axis,
        config.seq_len,
        save_path=figure5_path,
        selected_features=selected_features,
    )
    figure_artifacts["Figure 5"]["image"] = str(figure5_path)
    figure_artifacts["Figure 5"]["representative_indices"] = figure5_indices

    heatmap_path = figure_output_dir / artifact_specs["Figure 6"]["image_filename"]
    plot_feature_time_heatmaps(
        truth=saved_eval.test_raw[reference_idx],
        pred=evaluation["all_pred_raw"][reference_idx],
        time_axis=saved_eval.time_axis,
        save_path=heatmap_path,
    )
    figure_artifacts["Figure 6"]["image"] = str(heatmap_path)
    figure_artifacts["Figure 6"]["reference_trajectory_index"] = reference_idx

    breakthrough_path = figure_output_dir / artifact_specs["Figure 7"]["image_filename"]
    plot_breakthrough_summary(
        evaluation["all_pred_raw"],
        saved_eval.test_raw,
        evaluation["rmse_per_traj"],
        saved_eval.time_axis,
        config.seq_len,
        save_path=breakthrough_path,
        selected_features=selected_features,
        reference_idx=reference_idx,
    )
    figure_artifacts["Figure 7"]["image"] = str(breakthrough_path)
    figure_artifacts["Figure 7"]["reference_trajectory_index"] = reference_idx

    figure_artifacts["Figure 8"]["image"] = None

    parity_path = figure_output_dir / artifact_specs["Figure 9"]["image_filename"]
    plot_predicted_vs_actual(
        evaluation["all_pred_raw"],
        saved_eval.test_raw,
        config.seq_len,
        save_path=parity_path,
        title=f"[{config.experiment_name}] Paper-Aligned Figure 9 - Predicted vs Actual",
    )
    figure_artifacts["Figure 9"]["image"] = str(parity_path)

    generalization_path = figure_output_dir / artifact_specs["Figure 12"]["image_filename"]
    generalization_bins = plot_generalization_novelty(
        novelty_scores,
        evaluation["rmse_per_traj"],
        save_path=generalization_path,
    )
    figure_artifacts["Figure 12"]["image"] = str(generalization_path)

    checklist_path = figure_output_dir / "paper_figure_checklist.json"
    _write_json(checklist_path, get_paper_figure_checklist())

    metric_contract_path = figure_output_dir / "metric_contract.json"
    _write_json(metric_contract_path, get_metric_contract())

    for figure_name in ("Figure 4", "Figure 5", "Figure 6", "Figure 7", "Figure 12"):
        figure_artifacts[figure_name]["selected_features"] = selected_features.copy()

    markdown_paths = write_figure_markdown_files(
        output_dir=figure_output_dir,
        figure_artifacts=figure_artifacts,
    )
    for figure_name, markdown_path in markdown_paths.items():
        figure_artifacts[figure_name]["markdown"] = markdown_path

    manifest = build_paper_figure_manifest(
        experiment_name=config.experiment_name,
        selected_features=selected_features,
        canonical_metrics=_canonical_metrics(result_payload, evaluation["rmse_per_traj"]),
        figure_artifacts=figure_artifacts,
        checklist_path=str(checklist_path),
        metric_contract_path=str(metric_contract_path),
        generalization_bins=generalization_bins,
    )
    manifest_path = figure_output_dir / "paper_figure_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Saved: {manifest_path}")
    return manifest


def generate_full_paper_figure_bundle(
    experiment_dir: str,
    output_dir: str | None = None,
    experiments_root: str | None = None,
    selected_features: Sequence[str] = DEFAULT_PROFILE_FEATURES,
) -> dict[str, Any]:
    experiment_path = Path(experiment_dir)
    figure_output_dir = Path(output_dir) if output_dir else experiment_path / "paper_figures_full"
    figure_output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_legacy_figure_outputs(figure_output_dir)
    selected_features = list(selected_features)

    saved_eval = load_saved_experiment_evaluation(experiment_path)
    config = saved_eval.config
    evaluation = saved_eval.evaluation
    result_payload = saved_eval.result_payload
    novelty_scores = compute_initial_state_novelty(saved_eval.train_raw, saved_eval.test_raw)
    artifact_specs = {
        item["paper_figure"]: item for item in get_full_requested_figure_artifacts()
    }
    full_checklist = {
        item["paper_figure"]: item for item in _full_paper_figure_checklist()
    }

    figure_artifacts: dict[str, dict[str, Any]] = {
        figure_name: {"image": None, **full_checklist.get(figure_name, {})}
        for figure_name in artifact_specs
    }

    workflow_path = figure_output_dir / artifact_specs["Figure 1"]["image_filename"]
    plot_workflow_schematic(workflow_path)
    figure_artifacts["Figure 1"]["image"] = str(workflow_path)

    architecture_path = figure_output_dir / artifact_specs["Figure 2"]["image_filename"]
    plot_lstm_architecture_schematic(architecture_path)
    figure_artifacts["Figure 2"]["image"] = str(architecture_path)

    training_plot_path = figure_output_dir / artifact_specs["Figure 3"]["image_filename"]
    plot_loss_curve(
        result_payload["loss_history"],
        save_path=training_plot_path,
        title=f"[{config.experiment_name}] Paper-Aligned Figure 3 - Training Dynamics",
    )
    figure_artifacts["Figure 3"]["image"] = str(training_plot_path)

    figure4_path = figure_output_dir / artifact_specs["Figure 4"]["image_filename"]
    figure4_indices = plot_representative_profile_panels(
        evaluation["all_pred_raw"],
        saved_eval.test_raw,
        evaluation["rmse_per_traj"],
        saved_eval.time_axis,
        config.seq_len,
        save_path=figure4_path,
        selected_features=selected_features,
    )
    figure_artifacts["Figure 4"].update(
        {
            "image": str(figure4_path),
            "representative_indices": figure4_indices,
            "selected_features": selected_features.copy(),
        }
    )

    reference_idx = select_reference_trajectory_index(evaluation["rmse_per_traj"])
    figure5_path = figure_output_dir / artifact_specs["Figure 5"]["image_filename"]
    figure5_indices = plot_novelty_profile_panels(
        evaluation["all_pred_raw"],
        saved_eval.test_raw,
        novelty_scores,
        saved_eval.time_axis,
        config.seq_len,
        save_path=figure5_path,
        selected_features=selected_features,
    )
    figure_artifacts["Figure 5"].update(
        {
            "image": str(figure5_path),
            "representative_indices": figure5_indices,
            "selected_features": selected_features.copy(),
        }
    )

    heatmap_path = figure_output_dir / artifact_specs["Figure 6"]["image_filename"]
    plot_feature_time_heatmaps(
        truth=saved_eval.test_raw[reference_idx],
        pred=evaluation["all_pred_raw"][reference_idx],
        time_axis=saved_eval.time_axis,
        save_path=heatmap_path,
    )
    figure_artifacts["Figure 6"].update(
        {
            "image": str(heatmap_path),
            "reference_trajectory_index": reference_idx,
            "selected_features": selected_features.copy(),
        }
    )

    breakthrough_path = figure_output_dir / artifact_specs["Figure 7"]["image_filename"]
    plot_breakthrough_summary(
        evaluation["all_pred_raw"],
        saved_eval.test_raw,
        evaluation["rmse_per_traj"],
        saved_eval.time_axis,
        config.seq_len,
        save_path=breakthrough_path,
        selected_features=selected_features,
        reference_idx=reference_idx,
    )
    figure_artifacts["Figure 7"].update(
        {
            "image": str(breakthrough_path),
            "reference_trajectory_index": reference_idx,
            "selected_features": selected_features.copy(),
        }
    )

    figure_artifacts["Figure 8"]["image"] = None

    parity_path = figure_output_dir / artifact_specs["Figure 9"]["image_filename"]
    plot_predicted_vs_actual(
        evaluation["all_pred_raw"],
        saved_eval.test_raw,
        config.seq_len,
        save_path=parity_path,
        title=f"[{config.experiment_name}] Paper-Aligned Figure 9 - Predicted vs Actual",
    )
    parity_quality = summarize_parity_quality(
        evaluation["all_pred_raw"],
        saved_eval.test_raw,
        config.seq_len,
        feature_names=FEATURE_NAMES,
    )
    figure_artifacts["Figure 9"].update(
        {
            "image": str(parity_path),
            "parity_quality": parity_quality,
        }
    )

    metric_rows = collect_experiment_metric_rows(experiments_root or experiment_path.parent)
    if not metric_rows:
        metric_rows = [
            _experiment_metric_row(
                experiment_name=config.experiment_name,
                seq_len=config.seq_len,
                hidden_size=config.hidden_size,
                result_payload=result_payload,
                overall_rmse_mean=float(np.mean(evaluation["rmse_per_traj"])),
            )
        ]

    figure_artifacts["Figure 10"].update(
        {
            "image": None,
            "experiment_rows": metric_rows,
        }
    )

    metric_path = figure_output_dir / artifact_specs["Figure 11"]["image_filename"]
    plot_metric_sensitivity_summary(metric_rows, metric_path)
    figure_artifacts["Figure 11"].update(
        {
            "image": str(metric_path),
            "experiment_rows": metric_rows,
        }
    )

    generalization_path = figure_output_dir / artifact_specs["Figure 12"]["image_filename"]
    generalization_bins = plot_generalization_novelty(
        novelty_scores,
        evaluation["rmse_per_traj"],
        save_path=generalization_path,
    )
    figure_artifacts["Figure 12"].update(
        {
            "image": str(generalization_path),
            "selected_features": selected_features.copy(),
        }
    )

    checklist_path = figure_output_dir / "paper_figure_checklist.json"
    _write_json(checklist_path, list(full_checklist.values()))

    metric_contract_path = figure_output_dir / "metric_contract.json"
    _write_json(metric_contract_path, get_metric_contract())

    markdown_paths = write_figure_markdown_files(
        output_dir=figure_output_dir,
        figure_artifacts=figure_artifacts,
        full_package=True,
    )
    for figure_name, markdown_path in markdown_paths.items():
        figure_artifacts[figure_name]["markdown"] = markdown_path

    manifest = build_paper_figure_manifest(
        experiment_name=config.experiment_name,
        selected_features=selected_features,
        canonical_metrics=_canonical_metrics(result_payload, evaluation["rmse_per_traj"]),
        figure_artifacts=figure_artifacts,
        checklist_path=str(checklist_path),
        metric_contract_path=str(metric_contract_path),
        generalization_bins=generalization_bins,
    )
    manifest["experiment_metric_rows"] = metric_rows
    manifest["parity_quality"] = parity_quality

    manifest_path = figure_output_dir / "paper_figure_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Saved: {manifest_path}")
    return manifest
