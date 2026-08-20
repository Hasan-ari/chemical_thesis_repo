from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from conditional_model_v1.data import (
    CONDITION_FEATURES,
    OUTPUT_FEATURES,
    SCALAR_CONDITION_FEATURES,
)
from scripts.four_rock_visuals import generate as visual_module
from scripts.four_rock_visuals.generate import (
    DatasetLayout,
    ReportArtifacts,
    RockSamples,
    default_dataset_layouts,
    generate_report,
    load_rock_samples,
    main,
)

ROCKS = ("Calcite", "Dolomite", "Halite", "Trona")


class FourRockVisualTests(unittest.TestCase):
    def test_default_layouts_point_to_the_exact_four_datasets(self) -> None:
        layouts = default_dataset_layouts(Path("/data"))

        self.assertEqual(
            [(layout.rock, layout.dataset_name) for layout in layouts],
            [
                ("Calcite", "Calcite_wat_sat_data_3"),
                ("Dolomite", "Dolomite_wat_sat_data_2"),
                ("Halite", "Halite_wat_sat_data_2"),
                ("Trona", "Trona_par_sat_data_3"),
            ],
        )
        self.assertEqual(
            layouts[-1].dataset_root,
            Path("/data/wat_sat/Trona/Trona_par_sat_data_3"),
        )

    def test_raw_loader_reads_only_the_first_two_timestamps_and_aligns_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "Calcite_wat_sat_data_3"
            self._write_run(dataset_root, rock="Calcite", run_id="10", offset=100.0)
            self._write_run(
                dataset_root,
                rock="Calcite",
                run_id="2",
                offset=20.0,
                invalid_third_output_row=True,
            )

            samples = load_rock_samples(
                DatasetLayout(
                    rock="Calcite",
                    dataset_name="Calcite_wat_sat_data_3",
                    dataset_root=dataset_root,
                ),
                progress_every=0,
            )

        self.assertEqual(
            samples.run_ids,
            (
                "Calcite_wat_sat_data_3:2",
                "Calcite_wat_sat_data_3:10",
            ),
        )
        self.assertEqual(samples.conditions.shape, (2, len(CONDITION_FEATURES)))
        self.assertEqual(samples.outputs.shape, (2, 2, len(OUTPUT_FEATURES)))
        np.testing.assert_allclose(samples.time_values, [0.0, 0.6])
        self.assertAlmostEqual(samples.outputs[0, 0, 0], 20.0)
        self.assertAlmostEqual(samples.outputs[0, 1, 0], 21.0)
        self.assertAlmostEqual(samples.outputs[1, 0, 0], 100.0)
        self.assertAlmostEqual(samples.outputs[1, 1, 0], 101.0)
        self.assertAlmostEqual(samples.conditions[0, 0], 21.0)
        self.assertAlmostEqual(samples.conditions[1, 0], 101.0)

    def test_raw_loader_rejects_unmatched_runs_and_inconsistent_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "Calcite_wat_sat_data_3"
            self._write_run(dataset_root, rock="Calcite", run_id="1", offset=1.0)
            self._write_run(
                dataset_root,
                rock="Calcite",
                run_id="2",
                offset=2.0,
                time_values=(0.0, 0.7),
            )
            layout = DatasetLayout(
                rock="Calcite",
                dataset_name="Calcite_wat_sat_data_3",
                dataset_root=dataset_root,
            )

            with self.assertRaisesRegex(ValueError, "time indices 0 and 1"):
                load_rock_samples(layout, progress_every=0)

            (dataset_root / "output" / "2_Output.txt").unlink()
            with self.assertRaisesRegex(ValueError, "input/output run ids do not match"):
                load_rock_samples(layout, progress_every=0)

    def test_report_uses_every_paired_run_and_writes_exactly_five_pngs(self) -> None:
        samples = self._synthetic_four_rocks()
        original_change_counts = visual_module._change_counts
        change_calls: list[tuple[np.ndarray, np.ndarray]] = []

        def record_change_counts(at_t0, at_t1):
            change_calls.append((np.asarray(at_t0).copy(), np.asarray(at_t1).copy()))
            return original_change_counts(at_t0, at_t1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            with mock.patch.object(
                visual_module,
                "_change_counts",
                new=record_change_counts,
            ):
                artifacts = generate_report(samples, output_dir)

            png_paths = sorted(output_dir.glob("*.png"))
            self.assertEqual(len(png_paths), 5)
            self.assertEqual(set(png_paths), set(artifacts.png_paths))
            self.assertEqual(
                [path.name for path in artifacts.output_png_paths],
                [
                    "calcite_t0_t1_output_transition.png",
                    "dolomite_t0_t1_output_transition.png",
                    "halite_t0_t1_output_transition.png",
                    "trona_t0_t1_output_transition.png",
                ],
            )
            self.assertEqual(
                artifacts.condition_png_path.name,
                "condition_distributions_all_rocks.png",
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in png_paths))

        self.assertEqual(len(change_calls), 4 * len(OUTPUT_FEATURES))
        for rock_index, rock_samples in enumerate(samples):
            for feature_index in range(len(OUTPUT_FEATURES)):
                call_index = rock_index * len(OUTPUT_FEATURES) + feature_index
                actual_t0, actual_t1 = change_calls[call_index]
                np.testing.assert_array_equal(
                    actual_t0,
                    rock_samples.outputs[:, 0, feature_index],
                )
                np.testing.assert_array_equal(
                    actual_t1,
                    rock_samples.outputs[:, 1, feature_index],
                )

    def test_change_summary_counts_direction_for_each_paired_run(self) -> None:
        at_t0 = np.asarray([0.0, 1.0, 2.0, 3.0])
        at_t1 = np.asarray([1.0, 1.0, 1.0, 2.0])

        self.assertEqual(visual_module._change_counts(at_t0, at_t1), (2, 1, 1))
        self.assertEqual(
            visual_module._format_distribution(np.asarray([7.0, 7.0, 7.0])),
            "fixed 7",
        )

    def test_cli_wires_local_data_and_output_paths_without_training(self) -> None:
        samples = self._synthetic_four_rocks()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            artifacts = ReportArtifacts(
                output_png_paths=tuple(root / f"{rock}.png" for rock in ROCKS),
                condition_png_path=root / "conditions.png",
            )
            with (
                mock.patch(
                    "scripts.four_rock_visuals.generate.load_all_rocks",
                    return_value=samples,
                ) as load_mock,
                mock.patch(
                    "scripts.four_rock_visuals.generate.generate_report",
                    return_value=artifacts,
                ) as report_mock,
            ):
                main(
                    [
                        "--data-root",
                        str(root / "data"),
                        "--output-dir",
                        str(root / "visuals"),
                    ]
                )

        load_mock.assert_called_once_with(root / "data")
        report_mock.assert_called_once_with(samples, root / "visuals")

    @staticmethod
    def _write_run(
        dataset_root: Path,
        *,
        rock: str,
        run_id: str,
        offset: float,
        time_values: tuple[float, float] = (0.0, 0.6),
        invalid_third_output_row: bool = False,
    ) -> None:
        input_dir = dataset_root / "input"
        output_dir = dataset_root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_lines = [
            f"{{{feature}}} {offset + index + 1.0}"
            for index, feature in enumerate(SCALAR_CONDITION_FEATURES)
        ]
        input_lines.extend(
            [
                f"{{{rock.upper()}_MOLES}} {offset + 100.0}",
                f"{{{rock.upper()}_AREA}} {offset + 200.0}",
            ]
        )
        (input_dir / f"{run_id}_Input.txt").write_text("\n".join(input_lines) + "\n")

        header = " ".join(("time_d", *OUTPUT_FEATURES))
        rows = []
        for time_index, time_value in enumerate(time_values):
            feature_values = [
                offset + feature_index + time_index
                for feature_index in range(len(OUTPUT_FEATURES))
            ]
            rows.append(" ".join(str(value) for value in (time_value, *feature_values)))
        third_row = "this row must never be parsed" if invalid_third_output_row else rows[-1]
        (output_dir / f"{run_id}_Output.txt").write_text(
            "\n".join((header, *rows, third_row)) + "\n"
        )

    @staticmethod
    def _synthetic_four_rocks() -> tuple[RockSamples, ...]:
        samples = []
        for rock_index, rock in enumerate(ROCKS):
            n_runs = 4
            conditions = np.arange(
                n_runs * len(CONDITION_FEATURES), dtype=np.float64
            ).reshape(n_runs, len(CONDITION_FEATURES))
            conditions += rock_index * 100.0
            outputs = np.arange(
                n_runs * 2 * len(OUTPUT_FEATURES), dtype=np.float64
            ).reshape(n_runs, 2, len(OUTPUT_FEATURES))
            outputs += rock_index * 1000.0
            samples.append(
                RockSamples(
                    rock=rock,
                    dataset_name=f"{rock.lower()}_dataset",
                    run_ids=tuple(f"{rock}:{index}" for index in range(n_runs)),
                    conditions=conditions,
                    outputs=outputs,
                    time_values=np.asarray([0.0, 0.6], dtype=np.float64),
                )
            )
        return tuple(samples)


if __name__ == "__main__":
    unittest.main()
