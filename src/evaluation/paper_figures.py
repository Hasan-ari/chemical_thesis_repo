from __future__ import annotations

from src.evaluation.paper_bundle import (
    build_paper_figure_manifest,
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
    select_novelty_representative_indices,
    select_reference_trajectory_index,
    selected_feature_indices,
    summarize_generalization_bins,
)
from src.evaluation.paper_plots import (
    plot_breakthrough_summary,
    plot_feature_time_heatmaps,
    plot_generalization_novelty,
    plot_novelty_profile_panels,
    plot_profile_panel_family,
    plot_representative_profile_panels,
)
from src.evaluation.paper_runtime import validate_repo_env_python
from src.evaluation.paper_writeups import (
    build_figure_writeup,
    cleanup_legacy_figure_outputs,
    get_requested_figure_artifacts,
    write_figure_markdown_files,
)

__all__ = [
    "DEFAULT_BREAKTHROUGH_FEATURES",
    "DEFAULT_PROFILE_FEATURES",
    "METRIC_CONTRACT_VERSION",
    "build_figure_writeup",
    "build_paper_figure_manifest",
    "cleanup_legacy_figure_outputs",
    "compute_initial_state_novelty",
    "generate_paper_figure_bundle",
    "get_metric_contract",
    "get_paper_figure_checklist",
    "get_requested_figure_artifacts",
    "load_saved_experiment_evaluation",
    "normalize_result_metrics",
    "percentile_indices",
    "plot_breakthrough_summary",
    "plot_feature_time_heatmaps",
    "plot_generalization_novelty",
    "plot_novelty_profile_panels",
    "plot_profile_panel_family",
    "plot_representative_profile_panels",
    "select_novelty_representative_indices",
    "select_reference_trajectory_index",
    "selected_feature_indices",
    "summarize_generalization_bins",
    "validate_repo_env_python",
    "write_figure_markdown_files",
]
