from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.evaluation.paper_figures import (
    compute_initial_state_novelty,
    get_metric_contract,
    get_paper_figure_checklist,
    normalize_result_metrics,
    select_reference_trajectory_index,
    summarize_generalization_bins,
    validate_repo_env_python,
)


class PaperFigureChecklistTests(unittest.TestCase):
    def test_checklist_uses_expected_repo_statuses(self) -> None:
        checklist = get_paper_figure_checklist()
        allowed_statuses = {"direct", "adapted", "unsupported"}
        status_by_figure = {
            item["paper_figure"]: item["repo_status"] for item in checklist
        }

        self.assertTrue(checklist)
        self.assertTrue(
            all(item["repo_status"] in allowed_statuses for item in checklist)
        )
        self.assertEqual(status_by_figure["Figure 3"], "direct")
        self.assertEqual(status_by_figure["Figure 6"], "adapted")
        self.assertEqual(status_by_figure["Figure 8"], "unsupported")
        self.assertEqual(status_by_figure["Figure 1"], "unsupported")


class MetricContractTests(unittest.TestCase):
    def test_metric_contract_explains_legacy_and_canonical_names(self) -> None:
        contract = {item["storage_key"]: item for item in get_metric_contract()}

        self.assertEqual(contract["results.json:rmse_total"]["canonical_name"], "nrmse_total")
        self.assertEqual(contract["results.json:rmse_total"]["scale"], "normalized")
        self.assertEqual(
            contract["comprehensive_stats.json:overall_rmse.mean"]["scale"],
            "original",
        )

    def test_normalize_result_metrics_preserves_legacy_alias(self) -> None:
        normalized = normalize_result_metrics(
            {
                "rmse_total": 0.25,
                "rmse_per_var": {"pH": 0.01},
            }
        )

        self.assertEqual(normalized["nrmse_total"], 0.25)
        self.assertEqual(normalized["legacy_rmse_total_alias"], 0.25)
        self.assertEqual(normalized["rmse_per_var"]["pH"], 0.01)


class GeneralizationHelperTests(unittest.TestCase):
    def test_compute_initial_state_novelty_ranks_farther_samples_higher(self) -> None:
        train_raw = np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ]
        )
        test_raw = np.array(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[4.0, 4.0], [4.0, 4.0]],
            ]
        )

        novelty_scores = compute_initial_state_novelty(train_raw, test_raw)

        self.assertEqual(novelty_scores.shape, (2,))
        self.assertLess(novelty_scores[0], novelty_scores[1])

    def test_summarize_generalization_bins_groups_scores_by_quantiles(self) -> None:
        novelty_scores = np.array([0.1, 0.2, 1.0, 1.2, 2.0, 2.2])
        rmse_per_traj = np.array([0.11, 0.12, 0.2, 0.24, 0.35, 0.4])

        bins = summarize_generalization_bins(novelty_scores, rmse_per_traj, n_bins=3)

        self.assertEqual([item["bin_label"] for item in bins], [
            "low_novelty",
            "mid_novelty",
            "high_novelty",
        ])
        self.assertEqual([item["count"] for item in bins], [2, 2, 2])
        self.assertLess(bins[0]["mean_novelty"], bins[-1]["mean_novelty"])
        self.assertLess(bins[0]["mean_rmse"], bins[-1]["mean_rmse"])

    def test_select_reference_trajectory_index_uses_median_rmse(self) -> None:
        rmse_per_traj = np.array([0.05, 0.12, 0.25, 0.35, 0.6])

        selected_idx = select_reference_trajectory_index(rmse_per_traj)

        self.assertEqual(selected_idx, 2)


class EnvValidationTests(unittest.TestCase):
    def test_validate_repo_env_python_accepts_repo_env_interpreter(self) -> None:
        repo_root = "/tmp/demo-repo"
        executable = "/tmp/demo-repo/env/bin/python"

        validated = validate_repo_env_python(repo_root, executable=executable)

        self.assertEqual(validated, Path(executable).resolve())

    def test_validate_repo_env_python_rejects_non_env_interpreter(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_repo_env_python("/tmp/demo-repo", executable="/usr/bin/python3")


if __name__ == "__main__":
    unittest.main()
