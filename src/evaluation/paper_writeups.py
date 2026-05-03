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

LEGACY_FIGURE_OUTPUTS: tuple[str, ...] = (
    "figure3_training_dynamics.png",
    "figure4_5_representative_profiles.png",
    "figure6_truth_pred_error_heatmaps.png",
    "figure7_breakthrough_summary.png",
    "figure9_parity.png",
    "figure12_generalization_novelty.png",
    "figure10_11_ablation_sensitivity.png",
)

_SPEC_BY_FIGURE = {spec.paper_figure: spec for spec in PAPER_FIGURE_SPECS}


def get_requested_figure_artifacts() -> list[dict[str, str | None]]:
    return [asdict(artifact) for artifact in REQUESTED_FIGURE_ARTIFACTS]


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
    if paper_figure == "Figure 3":
        return (
            "This figure shows how the training loss evolves over epochs.",
            "It uses the saved `loss_history` from the experiment results and plots total training loss over time.",
        )
    if paper_figure == "Figure 4":
        return (
            "This figure shows representative rollout profiles chosen by RMSE percentile.",
            "The repo selects P10, P50, and P90 trajectories from rollout RMSE and overlays ground truth vs prediction across selected variables.",
        )
    if paper_figure == "Figure 5":
        return (
            "This figure shows a second representative profile family, now grouped by novelty level instead of RMSE percentile.",
            "The repo bins test trajectories into low, mid, and high initial-state novelty groups and plots one representative trajectory from each group.",
        )
    if paper_figure == "Figure 6":
        return (
            "This figure shows truth, prediction, and error as aligned heatmaps for one representative rollout.",
            "The repo chooses one reference trajectory and plots feature-by-time heatmaps for truth, prediction, and normalized absolute error.",
        )
    if paper_figure == "Figure 7":
        return (
            "This figure shows key output curves for one representative rollout.",
            "The repo uses the same representative trajectory idea to compare predicted vs true chemistry variables over rollout time.",
        )
    if paper_figure == "Figure 8":
        return (
            "This figure is explanation-only in the repo workflow.",
            "The paper uses latent solid-state or reaction-rate style quantities, but the current dataset does not expose pyrite-mass or reaction-rate outputs.",
        )
    if paper_figure == "Figure 9":
        return (
            "This figure shows pooled parity between predictions and ground truth.",
            "The repo pools all post-context predictions and renders predicted-vs-actual scatter with the same agreement question as the paper.",
        )
    if paper_figure == "Figure 12":
        return (
            "This figure shows generalization performance as a function of how unusual the initial condition is.",
            "The repo computes initial-state novelty and compares it against rollout RMSE; this is an adapted generalization test, not a spatial-zone relocation test.",
        )
    raise KeyError(f"Unsupported writeup figure: {paper_figure}")


def build_figure_writeup(
    paper_figure: str,
    image_path: str | None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    metadata = metadata or {}
    figure_spec = _SPEC_BY_FIGURE[paper_figure]
    summary, method = _figure_method_summary(paper_figure)

    how_to_read = "Compare the shape, timing, and spread of the prediction against the ground truth."
    if paper_figure == "Figure 8":
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
            figure_spec.notes,
            "",
        ]
    )


def write_figure_markdown_files(
    output_dir: Path,
    figure_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_paths: dict[str, str] = {}

    for artifact in REQUESTED_FIGURE_ARTIFACTS:
        artifact_payload = figure_artifacts.get(artifact.paper_figure, {})
        image_path = artifact_payload.get("image")
        markdown_path = output_dir / artifact.markdown_filename
        markdown_path.write_text(
            build_figure_writeup(
                artifact.paper_figure,
                image_path=image_path,
                metadata=artifact_payload,
            )
        )
        markdown_paths[artifact.paper_figure] = str(markdown_path)

    return markdown_paths
