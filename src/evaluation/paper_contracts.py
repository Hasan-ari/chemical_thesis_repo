from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RepoFigureStatus = Literal["direct", "adapted", "unsupported"]

DEFAULT_PROFILE_FEATURES: tuple[str, ...] = (
    "pH2_atm",
    "pCH4_atm",
    "CH4_g_mol",
    "SO4",
    "Formate",
    "Acetate",
)
DEFAULT_BREAKTHROUGH_FEATURES: tuple[str, ...] = DEFAULT_PROFILE_FEATURES
METRIC_CONTRACT_VERSION = "2026-04-hidden-rtnn-paper-figures"


@dataclass(frozen=True)
class PaperFigureSpec:
    paper_figure: str
    repo_status: RepoFigureStatus
    paper_role: str
    repo_equivalent: str
    notes: str


@dataclass(frozen=True)
class MetricContractEntry:
    storage_key: str
    canonical_name: str
    scale: str
    description: str


@dataclass(frozen=True)
class GeneralizationBin:
    bin_label: str
    count: int
    mean_novelty: float
    min_novelty: float
    max_novelty: float
    mean_rmse: float
    std_rmse: float


PAPER_FIGURE_SPECS: tuple[PaperFigureSpec, ...] = (
    PaperFigureSpec(
        paper_figure="Figure 1",
        repo_status="unsupported",
        paper_role="transport workflow schematic",
        repo_equivalent="No repo-native equivalent without new transport-state inputs",
        notes="Keep documented as unavailable under current repo-only constraints.",
    ),
    PaperFigureSpec(
        paper_figure="Figure 2",
        repo_status="unsupported",
        paper_role="HRTNet architecture schematic",
        repo_equivalent="No repo-native equivalent without a transport-informed model",
        notes="Current repository evaluates an autoregressive LSTM instead of HRTNet.",
    ),
    PaperFigureSpec(
        paper_figure="Figure 3",
        repo_status="direct",
        paper_role="training dynamics",
        repo_equivalent="Loss history from saved LSTM experiments",
        notes="This is a direct evaluation analog for convergence behavior.",
    ),
    PaperFigureSpec(
        paper_figure="Figure 4",
        repo_status="adapted",
        paper_role="representative profile panels",
        repo_equivalent="Representative time-series profiles for selected test trajectories",
        notes="Time replaces depth because the current data have no spatial coordinate.",
    ),
    PaperFigureSpec(
        paper_figure="Figure 5",
        repo_status="adapted",
        paper_role="second representative profile panel family",
        repo_equivalent="A second representative profile panel family on repo-native variables",
        notes="Keeps the paper's comparison role while staying within available data.",
    ),
    PaperFigureSpec(
        paper_figure="Figure 6",
        repo_status="adapted",
        paper_role="truth/prediction/error fields",
        repo_equivalent="Feature-time heatmaps for one representative rollout",
        notes="A representative rollout stands in for the paper's space-time field.",
    ),
    PaperFigureSpec(
        paper_figure="Figure 7",
        repo_status="adapted",
        paper_role="breakthrough curves",
        repo_equivalent="Representative chemistry output curves for one reference rollout",
        notes="Uses a median-RMSE rollout instead of an outlet transport experiment.",
    ),
    PaperFigureSpec(
        paper_figure="Figure 8",
        repo_status="unsupported",
        paper_role="derived solid-state diagnostics",
        repo_equivalent="No repo-native equivalent without pyrite-mass or reaction-rate state",
        notes="The current dataset does not expose the paper's mechanistic latent variables.",
    ),
    PaperFigureSpec(
        paper_figure="Figure 9",
        repo_status="direct",
        paper_role="parity and correlation plots",
        repo_equivalent="Predicted-vs-actual scatter across all variables",
        notes="This is a direct evaluation analog for agreement assessment.",
    ),
    PaperFigureSpec(
        paper_figure="Figure 10",
        repo_status="adapted",
        paper_role="ablation / data density",
        repo_equivalent="Sequence-length and hidden-size ablation heatmap",
        notes="Uses the saved experiment matrix already present in the repository.",
    ),
    PaperFigureSpec(
        paper_figure="Figure 11",
        repo_status="adapted",
        paper_role="sensitivity analysis",
        repo_equivalent="Normalized-vs-original-scale sensitivity heatmaps",
        notes="Reuses saved experiments to show metric sensitivity rather than physics-loss sensitivity.",
    ),
    PaperFigureSpec(
        paper_figure="Figure 12",
        repo_status="adapted",
        paper_role="generalization study",
        repo_equivalent="Initial-state novelty vs rollout RMSE",
        notes="Measures how far test initial conditions drift from the training manifold.",
    ),
)


METRIC_CONTRACT: tuple[MetricContractEntry, ...] = (
    MetricContractEntry(
        storage_key="results.json:rmse_total",
        canonical_name="nrmse_total",
        scale="normalized",
        description="Legacy key in saved experiment results; value is the normalized autoregressive total RMSE.",
    ),
    MetricContractEntry(
        storage_key="results.json:nrmse_total",
        canonical_name="nrmse_total",
        scale="normalized",
        description="Canonical key for normalized total RMSE in new tooling.",
    ),
    MetricContractEntry(
        storage_key="results.json:rmse_per_var.*",
        canonical_name="rmse_per_var",
        scale="original",
        description="Per-variable RMSE on the original chemistry scale.",
    ),
    MetricContractEntry(
        storage_key="comprehensive_stats.json:overall_rmse.mean",
        canonical_name="rmse_per_trajectory_mean",
        scale="original",
        description="Mean original-scale RMSE computed across test trajectories.",
    ),
    MetricContractEntry(
        storage_key="comprehensive_stats.json:per_variable_rmse_mean.*",
        canonical_name="rmse_per_traj_per_var_mean",
        scale="original",
        description="Mean per-trajectory RMSE for each variable on the original scale.",
    ),
)
