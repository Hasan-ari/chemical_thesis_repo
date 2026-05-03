from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.paper_figures import (
    build_paper_figure_manifest,
    cleanup_legacy_figure_outputs,
    compute_initial_state_novelty,
    get_requested_figure_artifacts,
    get_metric_contract,
    get_paper_figure_checklist,
    normalize_result_metrics,
    select_reference_trajectory_index,
    select_novelty_representative_indices,
    summarize_generalization_bins,
    validate_repo_env_python,
    write_figure_markdown_files,
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

    def test_requested_artifacts_match_target_figure_set(self) -> None:
        artifacts = {
            item["paper_figure"]: item for item in get_requested_figure_artifacts()
        }

        self.assertEqual(
            set(artifacts),
            {
                "Figure 3",
                "Figure 4",
                "Figure 5",
                "Figure 6",
                "Figure 7",
                "Figure 8",
                "Figure 9",
                "Figure 12",
            },
        )
        self.assertEqual(
            artifacts["Figure 4"]["image_filename"],
            "figure04_representative_profiles_a.png",
        )
        self.assertEqual(
            artifacts["Figure 5"]["image_filename"],
            "figure05_representative_profiles_b.png",
        )
        self.assertIsNone(artifacts["Figure 8"]["image_filename"])
        self.assertEqual(artifacts["Figure 12"]["markdown_filename"], "figure12.md")


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

    def test_select_novelty_representative_indices_uses_three_bins(self) -> None:
        novelty_scores = np.array([0.1, 0.2, 1.0, 1.2, 2.0, 2.2])

        indices = select_novelty_representative_indices(novelty_scores)

        self.assertEqual(indices, [1, 3, 5])


class WriteupTests(unittest.TestCase):
    def test_cleanup_legacy_figure_outputs_removes_old_combined_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="paper-figure-cleanup-") as temp_dir:
            output_dir = Path(temp_dir)
            legacy_paths = [
                output_dir / "figure4_5_representative_profiles.png",
                output_dir / "figure10_11_ablation_sensitivity.png",
                output_dir / "figure12_generalization_novelty.png",
            ]
            for path in legacy_paths:
                path.write_text("legacy")

            cleanup_legacy_figure_outputs(output_dir)

            self.assertTrue(all(not path.exists() for path in legacy_paths))

    def test_write_figure_markdown_files_creates_figure8_md_without_png(self) -> None:
        with tempfile.TemporaryDirectory(prefix="paper-figure-writeups-") as temp_dir:
            output_dir = Path(temp_dir)
            figure_artifacts = {
                "Figure 3": {
                    "image": "figure03_training_dynamics.png",
                },
                "Figure 8": {
                    "image": None,
                },
                "Figure 12": {
                    "image": "figure12_generalization.png",
                },
            }

            markdown_paths = write_figure_markdown_files(
                output_dir=output_dir,
                figure_artifacts=figure_artifacts,
            )

            figure8_path = output_dir / "figure08.md"
            figure12_text = (output_dir / "figure12.md").read_text()
            figure8_text = figure8_path.read_text()

            self.assertEqual(markdown_paths["Figure 8"], str(figure8_path))
            self.assertIn("No PNG is produced", figure8_text)
            self.assertIn("pyrite-mass or reaction-rate", figure8_text)
            self.assertIn("initial-state novelty", figure12_text)
            self.assertIn("not a spatial-zone relocation test", figure12_text)


class ManifestTests(unittest.TestCase):
    def test_build_manifest_includes_markdown_and_reference_metadata(self) -> None:
        manifest = build_paper_figure_manifest(
            experiment_name="demo_exp",
            selected_features=["pH2_atm", "SO4"],
            canonical_metrics={
                "nrmse_total": 0.1,
                "legacy_rmse_total_alias": 0.1,
                "overall_rmse_mean": 0.2,
            },
            figure_artifacts={
                "Figure 4": {
                    "image": "figure04_representative_profiles_a.png",
                    "markdown": "figure04.md",
                    "representative_indices": [1, 2, 3],
                },
                "Figure 8": {
                    "image": None,
                    "markdown": "figure08.md",
                },
            },
            checklist_path="paper_figure_checklist.json",
            metric_contract_path="metric_contract.json",
            generalization_bins=[{"bin_label": "low_novelty"}],
        )

        self.assertEqual(manifest["figures"]["Figure 4"]["image"], "figure04_representative_profiles_a.png")
        self.assertEqual(manifest["figures"]["Figure 4"]["representative_indices"], [1, 2, 3])
        self.assertIsNone(manifest["figures"]["Figure 8"]["image"])
        self.assertEqual(manifest["figures"]["Figure 8"]["markdown"], "figure08.md")
        self.assertEqual(manifest["canonical_metrics"]["overall_rmse_mean"], 0.2)


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
