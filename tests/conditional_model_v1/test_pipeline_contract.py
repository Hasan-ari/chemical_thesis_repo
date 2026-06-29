from __future__ import annotations

import tempfile
import unittest
import csv
from pathlib import Path

import numpy as np
import torch

from conditional_model_v1.config import parse_config
from conditional_model_v1.data import (
    CONDITION_FEATURES,
    FullTrajectoryDataset,
    DatasetSpec,
    build_condition_time_tensor,
    load_input_parameters,
    load_output_trajectory,
)
from conditional_model_v1.models import ConditionTimeLSTM
from conditional_model_v1.plotting import plot_trajectory_examples
from conditional_model_v1.preprocessing import ConditionScaler, OutputScaler
from conditional_model_v1.splitting import rock_aware_split
from conditional_model_v1.tracking import ExperimentTracker


class PipelineContractTests(unittest.TestCase):
    def test_input_parser_maps_rock_specific_mineral_fields_to_generic_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "1_Input.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "{DATABASE} C:\\phreeqc\\database\\phreeqc.dat",
                        "{TEMPERATURE} 43.13",
                        "{POROSITY} 0.16",
                        "{WATER_VOLUME} 32.15",
                        "{GAS_VOLUME} 60.65",
                        "{SOLID_MASS} 2349.61",
                        "{PORE_VOLUME} 0.16",
                        "{ALKALINITY} 0.0126",
                        "{NA} 0.3801",
                        "{MG} 0.000104",
                        "{CL} 0.3801",
                        "{CA} 0.00232",
                        "{S6} 0.00232",
                        "{H2} 1.184",
                        "{CH4} 41.385",
                        "{CO2} 9.492",
                        "{N2} 7.933",
                        "{H2S} 0.0061",
                        "{DOLOMITE_MOLES} 12.74",
                        "{DOLOMITE_AREA} 50.35",
                    ]
                )
            )

            row = load_input_parameters(input_path, DatasetSpec(name="dolo", rock="Dolomite", path=tmp_dir))

        self.assertEqual(row["run_id"], "1")
        self.assertEqual(row["rock"], "Dolomite")
        self.assertEqual(row["mineral_moles"], 12.74)
        self.assertEqual(row["mineral_area"], 50.35)
        for feature in CONDITION_FEATURES:
            self.assertIn(feature, row)

    def test_output_parser_reads_time_axis_and_feature_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "1_Output.txt"
            output_path.write_text(
                "\n".join(
                    [
                        "time_d pH Ptot_atm pH2_atm",
                        "0.0 7.0 4.0 0.0",
                        "0.6 6.5 5.0 1.0",
                    ]
                )
            )

            frame = load_output_trajectory(output_path)

        self.assertEqual(frame["run_id"].iloc[0], "1")
        self.assertEqual(frame["time_d"].tolist(), [0.0, 0.6])
        self.assertEqual(frame["pH"].tolist(), [7.0, 6.5])

    def test_rock_aware_split_keeps_runs_disjoint_and_each_rock_represented(self) -> None:
        rocks = np.array(["Calcite"] * 10 + ["Dolomite"] * 10, dtype=object)

        split = rock_aware_split(rocks=rocks, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

        train = set(split.train.tolist())
        val = set(split.val.tolist())
        test = set(split.test.tolist())
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertEqual(train | val | test, set(range(20)))
        self.assertEqual(set(rocks[split.val]), {"Calcite", "Dolomite"})
        self.assertEqual(set(rocks[split.test]), {"Calcite", "Dolomite"})

    def test_scaling_and_dataset_contract_do_not_include_rock_label_in_x(self) -> None:
        conditions = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        time_axis = np.array([0.0, 0.6, 1.2], dtype=np.float64)
        trajectories = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)

        condition_scaler = ConditionScaler().fit(conditions[:1])
        output_scaler = OutputScaler(log_feature_indices=()).fit(trajectories[:1])

        x = build_condition_time_tensor(
            condition_scaler.transform(conditions),
            time_axis,
            time_mean=float(time_axis.mean()),
            time_std=float(time_axis.std()),
        )
        y = output_scaler.transform(trajectories)
        dataset = FullTrajectoryDataset(x=x, y=y, run_ids=["c:1", "d:1"], rocks=["Calcite", "Dolomite"])

        sample_x, sample_y = dataset[0]
        self.assertEqual(tuple(sample_x.shape), (3, 3))
        self.assertEqual(tuple(sample_y.shape), (3, 4))
        self.assertEqual(dataset.rocks[0], "Calcite")

    def test_condition_time_lstm_returns_full_trajectory_predictions(self) -> None:
        model = ConditionTimeLSTM(input_size=20, output_size=32, hidden_size=16, num_layers=1)
        x = torch.zeros((2, 301, 20), dtype=torch.float32)

        y = model(x)

        self.assertEqual(tuple(y.shape), (2, 301, 32))

    def test_plot_config_allows_null_max_runs_to_mean_every_eval_run(self) -> None:
        config = parse_config(
            {
                "experiment": {"name": "plot_all_runs", "run_root": "/tmp/runs"},
                "data": {
                    "processed_root": "/tmp/processed",
                    "datasets": [
                        {"name": "calcite", "rock": "Calcite", "path": "/tmp/calcite"},
                    ],
                },
                "plots": {"max_runs": None, "features": "all"},
            }
        )

        self.assertIsNone(config.plots.max_runs)
        self.assertEqual(config.plots.features, "all")

    def test_plotting_can_render_every_run_and_every_feature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            time_axis = np.array([0.0, 0.6, 1.2], dtype=np.float64)
            y_true = np.zeros((2, 3, 2), dtype=np.float64)
            y_pred = np.ones((2, 3, 2), dtype=np.float64)

            plot_trajectory_examples(
                time_axis=time_axis,
                y_true=y_true,
                y_pred=y_pred,
                output_features=("pH", "Ptot_atm"),
                run_ids=["calcite:1", "dolomite:1"],
                output_dir=Path(tmp_dir),
                max_runs=None,
                feature_names=None,
            )

            png_names = sorted(path.name for path in Path(tmp_dir).rglob("*.png"))

        self.assertIn("calcite_1_pH.png", png_names)
        self.assertIn("dolomite_1_Ptot_atm.png", png_names)
        self.assertIn("calcite_1_all_outputs.png", png_names)
        self.assertIn("dolomite_1_all_outputs.png", png_names)

    def test_tracker_writes_per_feature_metrics_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = parse_config(
                {
                    "experiment": {"name": "metrics_csv", "run_root": tmp_dir},
                    "data": {
                        "processed_root": "/tmp/processed",
                        "datasets": [
                            {"name": "calcite", "rock": "Calcite", "path": "/tmp/calcite"},
                        ],
                    },
                }
            )
            tracker = ExperimentTracker(config)

            tracker.write_feature_metrics(
                {
                    "rmse_per_feature_original": {"pH": 0.1, "Ptot_atm": 0.2},
                    "mae_per_feature_original": {"pH": 0.01, "Ptot_atm": 0.02},
                    "final_rmse_per_feature_original": {"pH": 0.3, "Ptot_atm": 0.4},
                    "final_mae_per_feature_original": {"pH": 0.03, "Ptot_atm": 0.04},
                }
            )

            with (tracker.run_dir / "feature_metrics.csv").open() as file_obj:
                rows = list(csv.DictReader(file_obj))

        self.assertEqual(rows[0]["feature"], "pH")
        self.assertEqual(rows[1]["feature"], "Ptot_atm")
        self.assertEqual(rows[0]["rmse_original"], "0.1")
        self.assertEqual(rows[1]["final_mae_original"], "0.04")


if __name__ == "__main__":
    unittest.main()
