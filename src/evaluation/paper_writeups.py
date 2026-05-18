from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from src.evaluation.paper_contracts import PAPER_FIGURE_SPECS


@dataclass(frozen=True)
class RequestedFigureArtifact:
    paper_figure: str
    image_filename: str | None
    markdown_filename: str


REQUESTED_FIGURE_ARTIFACTS: tuple[RequestedFigureArtifact, ...] = (
    RequestedFigureArtifact("Figure 3", "figure03_training_dynamics.png", "figure03.md"),
    RequestedFigureArtifact("Figure 4", "figure04_representative_profiles_a.png", "figure04.md"),
    RequestedFigureArtifact("Figure 5", "figure05_representative_profiles_b.png", "figure05.md"),
    RequestedFigureArtifact("Figure 6", "figure06_truth_pred_error_heatmaps.png", "figure06.md"),
    RequestedFigureArtifact("Figure 7", "figure07_breakthrough_summary.png", "figure07.md"),
    RequestedFigureArtifact("Figure 8", None, "figure08.md"),
    RequestedFigureArtifact("Figure 9", "figure09_parity.png", "figure09.md"),
    RequestedFigureArtifact("Figure 12", "figure12_generalization.png", "figure12.md"),
)

FULL_REQUESTED_FIGURE_ARTIFACTS: tuple[RequestedFigureArtifact, ...] = (
    RequestedFigureArtifact("Figure 1", "figure01_workflow_schematic.png", "figure01.md"),
    RequestedFigureArtifact("Figure 2", "figure02_lstm_architecture.png", "figure02.md"),
    RequestedFigureArtifact("Figure 3", "figure03_training_dynamics.png", "figure03.md"),
    RequestedFigureArtifact("Figure 4", "figure04_representative_profiles_a.png", "figure04.md"),
    RequestedFigureArtifact("Figure 5", "figure05_representative_profiles_b.png", "figure05.md"),
    RequestedFigureArtifact("Figure 6", "figure06_truth_pred_error_heatmaps.png", "figure06.md"),
    RequestedFigureArtifact("Figure 7", "figure07_breakthrough_summary.png", "figure07.md"),
    RequestedFigureArtifact("Figure 8", None, "figure08.md"),
    RequestedFigureArtifact("Figure 9", "figure09_parity.png", "figure09.md"),
    RequestedFigureArtifact("Figure 10", None, "figure10.md"),
    RequestedFigureArtifact("Figure 11", "figure11_metric_sensitivity.png", "figure11.md"),
    RequestedFigureArtifact("Figure 12", "figure12_generalization.png", "figure12.md"),
)

LEGACY_FIGURE_OUTPUTS: tuple[str, ...] = (
    "figure3_training_dynamics.png",
    "figure4_5_representative_profiles.png",
    "figure6_truth_pred_error_heatmaps.png",
    "figure7_breakthrough_summary.png",
    "figure9_parity.png",
    "figure12_generalization_novelty.png",
    "figure10_11_ablation_sensitivity.png",
    "figure10_sequence_sensitivity.png",
)

_SPEC_BY_FIGURE = {spec.paper_figure: spec for spec in PAPER_FIGURE_SPECS}

FIGURE_METHOD_SUMMARIES: dict[str, tuple[str, str]] = {
    "Figure 1": (
        "This figure shows the end-to-end surrogate modeling workflow.",
        "The repo analog summarizes PHREEQC output loading, train/test splitting, normalization, LSTM rollout, and paper-aligned evaluation outputs.",
    ),
    "Figure 2": (
        "This figure shows the neural-network architecture used by the surrogate.",
        "The repo analog documents the autoregressive LSTM architecture and explicitly contrasts it with the paper's two-network HRTNet/PDE design.",
    ),
    "Figure 3": (
        "This figure shows how the training loss evolves over epochs.",
        "It uses the saved `loss_history` from the experiment results and plots total training loss over time.",
    ),
    "Figure 4": (
        "This figure shows representative rollout profiles chosen by RMSE percentile.",
        "The repo selects P10, P50, and P90 trajectories from rollout RMSE and overlays ground truth vs prediction across selected variables.",
    ),
    "Figure 5": (
        "This figure shows a second representative profile family, now grouped by novelty level instead of RMSE percentile.",
        "The repo bins test trajectories into low, mid, and high initial-state novelty groups and plots one representative trajectory from each group.",
    ),
    "Figure 6": (
        "This figure shows truth, prediction, and error as aligned heatmaps for one representative rollout.",
        "The repo chooses one reference trajectory and plots feature-by-time heatmaps for truth, prediction, and normalized absolute error.",
    ),
    "Figure 7": (
        "This figure shows key output curves for one representative rollout.",
        "The repo uses the same representative trajectory idea to compare predicted vs true chemistry variables over rollout time.",
    ),
    "Figure 8": (
        "This figure is explanation-only in the repo workflow.",
        "The paper uses latent solid-state or reaction-rate style quantities, but the current dataset does not expose pyrite-mass or reaction-rate outputs.",
    ),
    "Figure 9": (
        "This figure shows pooled parity between predictions and ground truth.",
        "The repo pools all post-context predictions and renders predicted-vs-actual scatter with the same agreement question as the paper.",
    ),
    "Figure 10": (
        "This figure is explanation-only in the repo workflow.",
        "The paper varies spatial and temporal training-data density, but the current saved experiments do not retrain the LSTM on spatial grids or thinned temporal snapshots.",
    ),
    "Figure 11": (
        "This figure shows sensitivity to a modeling hyperparameter.",
        "The repo analog compares normalized training/evaluation RMSE against original-scale rollout RMSE because the LSTM has no physics-loss weighting factor.",
    ),
    "Figure 12": (
        "This figure shows generalization performance as a function of how unusual the initial condition is.",
        "The repo computes initial-state novelty and compares it against rollout RMSE; this is an adapted generalization test, not a spatial-zone relocation test.",
    ),
}

MEASUREMENT_SUMMARIES: dict[str, str] = {
    "Figure 1": "It measures reproducibility of the analysis path: which data, model, rollout, and output artifacts are connected.",
    "Figure 2": "It measures architectural scope: what the current LSTM can and cannot claim relative to HRTNet.",
    "Figure 3": "It measures optimization behavior through the saved MSE loss history.",
    "Figure 4": "It measures rollout quality at low, median, and high trajectory-level RMSE.",
    "Figure 5": "It measures whether initially unusual test states show visibly different rollout behavior.",
    "Figure 6": "It measures where prediction errors concentrate across variables and time.",
    "Figure 7": "It measures representative chemistry-curve timing, shape, and magnitude agreement.",
    "Figure 8": "It documents an unavailable diagnostic: pyrite mass and reaction-rate behavior are not output by this dataset.",
    "Figure 9": "It measures pooled agreement with R2, RMSE, and visual distance from the 1:1 line.",
    "Figure 10": "It documents that the paper's spatial/temporal data-density ablation cannot be reproduced from the current saved LSTM experiments.",
    "Figure 11": "It measures sensitivity of conclusions to normalized versus original-scale quality metrics.",
    "Figure 12": "It measures whether rollout RMSE increases as test initial states move away from the training manifold.",
}


def get_requested_figure_artifacts() -> list[dict[str, str | None]]:
    return [asdict(artifact) for artifact in REQUESTED_FIGURE_ARTIFACTS]


def get_full_requested_figure_artifacts() -> list[dict[str, str | None]]:
    return [asdict(artifact) for artifact in FULL_REQUESTED_FIGURE_ARTIFACTS]


def cleanup_legacy_figure_outputs(output_dir: Path) -> None:
    for filename in LEGACY_FIGURE_OUTPUTS:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def _format_image_reference(image_path: str | None) -> str:
    if image_path is None:
        return "No PNG is produced for this figure in the current repo workflow."
    return f"Rendered asset: `{image_path}`"


def _figure_method_summary(paper_figure: str) -> tuple[str, str]:
    try:
        return FIGURE_METHOD_SUMMARIES[paper_figure]
    except KeyError as exc:
        raise KeyError(f"Unsupported writeup figure: {paper_figure}") from exc


def _measurement_summary(paper_figure: str) -> str:
    return MEASUREMENT_SUMMARIES[paper_figure]


def build_figure_writeup(
    paper_figure: str,
    image_path: str | None,
    metadata: Mapping[str, Any] | None = None,
    full_package: bool = False,
) -> str:
    metadata = metadata or {}
    figure_spec = _SPEC_BY_FIGURE[paper_figure]
    summary, method = _figure_method_summary(paper_figure)
    paper_role = str(metadata.get("paper_role", figure_spec.paper_role))
    repo_equivalent = str(metadata.get("repo_equivalent", figure_spec.repo_equivalent))
    limitation_note = str(metadata.get("notes", figure_spec.notes))

    how_to_read = "Compare the shape, timing, and spread of the prediction against the ground truth."
    if paper_figure in {"Figure 8", "Figure 10"}:
        how_to_read = "Read this as a limitation note: the paper used a diagnostic that is not available from the current repo outputs."
    elif paper_figure == "Figure 12":
        how_to_read = (
            "Higher novelty means the test initial state is farther from the training manifold. "
            "This is not a spatial-zone relocation test; it is an adapted OOD-style generalization view."
        )

    metadata_lines: list[str] = []
    if metadata.get("representative_indices") is not None:
        metadata_lines.append(
            f"- Representative trajectories: `{metadata['representative_indices']}`"
        )
    if metadata.get("reference_trajectory_index") is not None:
        metadata_lines.append(
            f"- Reference trajectory index: `{metadata['reference_trajectory_index']}`"
        )
    if metadata.get("selected_features") is not None:
        metadata_lines.append(
            f"- Selected features: `{metadata['selected_features']}`"
        )
    if not metadata_lines:
        metadata_lines.append("- No extra run-specific metadata recorded for this figure.")

    if not full_package:
        return "\n".join(
            [
                f"# {paper_figure}",
                "",
                "## What This Figure Shows",
                summary,
                "",
                "## How It Is Computed In This Repo",
                method,
                _format_image_reference(image_path),
                *metadata_lines,
                "",
                "## How To Read It",
                how_to_read,
                "",
                "## Limitation Vs The Original Paper",
                limitation_note,
                "",
            ]
        )

    return "\n".join(
        [
            f"# {paper_figure}",
            "",
            "## Original Paper Role",
            paper_role,
            "",
            "## Repo Analog",
            repo_equivalent,
            "",
            "## What This Figure Shows",
            summary,
            "",
            "## What It Measures In This Study",
            _measurement_summary(paper_figure),
            "",
            "## How It Is Computed In This Repo",
            method,
            _format_image_reference(image_path),
            *metadata_lines,
            "",
            "## How To Read It",
            how_to_read,
            "",
            "## Limitation Vs The Original Paper",
            limitation_note,
            "",
        ]
    )


def write_figure_markdown_files(
    output_dir: Path,
    figure_artifacts: Mapping[str, Mapping[str, Any]],
    full_package: bool = False,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_paths: dict[str, str] = {}

    requested_artifacts = (
        FULL_REQUESTED_FIGURE_ARTIFACTS if full_package else REQUESTED_FIGURE_ARTIFACTS
    )
    for artifact in requested_artifacts:
        artifact_payload = figure_artifacts.get(artifact.paper_figure, {})
        image_path = artifact_payload.get("image")
        markdown_path = output_dir / artifact.markdown_filename
        markdown_path.write_text(
            build_figure_writeup(
                artifact.paper_figure,
                image_path=image_path,
                metadata=artifact_payload,
                full_package=full_package,
            )
        )
        markdown_paths[artifact.paper_figure] = str(markdown_path)

    return markdown_paths
