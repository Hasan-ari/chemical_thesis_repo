from __future__ import annotations

from src.evaluation.paper_bundle import (
    generate_paper_figure_bundle,
    load_saved_experiment_evaluation,
)
from src.evaluation.paper_contracts import (
    DEFAULT_BREAKTHROUGH_FEATURES,
    DEFAULT_PROFILE_FEATURES,
    METRIC_CONTRACT_VERSION,
)
from src.evaluation.paper_metrics import (
    compute_initial_state_novelty,
    get_metric_contract,
    get_paper_figure_checklist,
    normalize_result_metrics,
    percentile_indices,
    select_reference_trajectory_index,
    selected_feature_indices,
    summarize_generalization_bins,
)
from src.evaluation.paper_plots import (
    plot_ablation_and_sensitivity,
    plot_breakthrough_summary,
    plot_feature_time_heatmaps,
    plot_generalization_novelty,
    plot_representative_profile_panels,
)
from src.evaluation.paper_runtime import validate_repo_env_python

__all__ = [
    "DEFAULT_BREAKTHROUGH_FEATURES",
    "DEFAULT_PROFILE_FEATURES",
    "METRIC_CONTRACT_VERSION",
    "compute_initial_state_novelty",
    "generate_paper_figure_bundle",
    "get_metric_contract",
    "get_paper_figure_checklist",
    "load_saved_experiment_evaluation",
    "normalize_result_metrics",
    "percentile_indices",
    "plot_ablation_and_sensitivity",
    "plot_breakthrough_summary",
    "plot_feature_time_heatmaps",
    "plot_generalization_novelty",
    "plot_representative_profile_panels",
    "select_reference_trajectory_index",
    "selected_feature_indices",
    "summarize_generalization_bins",
    "validate_repo_env_python",
]
