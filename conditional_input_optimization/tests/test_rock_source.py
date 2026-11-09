"""Tests of conditional_input_optimization.run_data.rock_source."""

from __future__ import annotations

import pathlib

import pytest

from conditional_input_optimization.run_data import rock_source
from conditional_input_optimization.tests import synthetic_runs

_OUTPUT = {"time_d": [0.0], "H2_g_mol": [1.0]}


def _source(folder: pathlib.Path) -> rock_source.RockSource:
    return rock_source.RockSource(
        name="Calcite", folder=folder, input_units="mol_kgw"
    )


class TestRockSource:
    def test_paths_follow_the_folder_layout(self, tmp_path):
        source = _source(tmp_path)

        assert source.input_path("7") == tmp_path / "input" / "7_Input.txt"
        assert source.output_path("7") == tmp_path / "output" / "7_Output.txt"

    def test_empty_name_raises(self, tmp_path):
        with pytest.raises(ValueError, match="name"):
            rock_source.RockSource(
                name="", folder=tmp_path, input_units="mol_kgw"
            )

    def test_unsupported_units_raise(self, tmp_path):
        with pytest.raises(ValueError, match="input_units"):
            rock_source.RockSource(
                name="Calcite", folder=tmp_path, input_units="ppm"
            )

    def test_string_folder_raises(self):
        with pytest.raises(TypeError, match="pathlib.Path"):
            rock_source.RockSource(
                name="Calcite", folder="data", input_units="mol_kgw"
            )

    def test_is_immutable(self, tmp_path):
        source = _source(tmp_path)

        with pytest.raises(AttributeError):
            source.name = "Halite"


class TestFindCompleteRunIds:
    def test_returns_numeric_order_and_skips_failed_runs(self, tmp_path):
        values = synthetic_runs.required_input_values()
        for run_id in ("10", "2", "1", "3"):
            synthetic_runs.write_input(tmp_path, run_id, values)
        for run_id in ("10", "2", "1"):  # run 3 failed: no output file
            synthetic_runs.write_output(tmp_path, run_id, _OUTPUT)

        run_ids = rock_source.find_complete_run_ids(_source(tmp_path))

        assert run_ids == ("1", "2", "10")

    def test_output_without_input_is_skipped(self, tmp_path):
        synthetic_runs.write_input(
            tmp_path, "1", synthetic_runs.required_input_values()
        )
        synthetic_runs.write_output(tmp_path, "1", _OUTPUT)
        synthetic_runs.write_output(tmp_path, "2", _OUTPUT)

        assert rock_source.find_complete_run_ids(_source(tmp_path)) == ("1",)

    def test_missing_folder_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="input"):
            rock_source.find_complete_run_ids(_source(tmp_path))


class TestLoadRockSources:
    def _write_config(self, tmp_path, text):
        path = tmp_path / "config.yaml"
        path.write_text(text)
        return path

    def test_reads_rocks_in_file_order(self, tmp_path):
        config = self._write_config(
            tmp_path,
            "data_root: data/rocks\n"
            "rocks:\n"
            "  - {name: Halite, input_units: mol_kgw}\n"
            "  - {name: Mica_schist, input_units: mg_L}\n",
        )

        sources = rock_source.load_rock_sources(config, base_dir=tmp_path)

        assert [source.name for source in sources] == ["Halite", "Mica_schist"]
        assert sources[0].folder == tmp_path / "data" / "rocks" / "Halite"
        assert sources[1].input_units == "mg_L"

    def test_missing_key_raises(self, tmp_path):
        config = self._write_config(
            tmp_path, "data_root: d\nrocks:\n  - {name: Halite}\n"
        )

        with pytest.raises(ValueError, match="input_units"):
            rock_source.load_rock_sources(config, base_dir=tmp_path)

    def test_empty_rock_list_raises(self, tmp_path):
        config = self._write_config(tmp_path, "data_root: d\nrocks: []\n")

        with pytest.raises(ValueError, match="empty"):
            rock_source.load_rock_sources(config, base_dir=tmp_path)

    def test_duplicate_rock_names_raise(self, tmp_path):
        config = self._write_config(
            tmp_path,
            "data_root: d\n"
            "rocks:\n"
            "  - {name: Halite, input_units: mol_kgw}\n"
            "  - {name: Halite, input_units: mol_kgw}\n",
        )

        with pytest.raises(ValueError, match="duplicate"):
            rock_source.load_rock_sources(config, base_dir=tmp_path)

    def test_non_mapping_file_raises(self, tmp_path):
        config = self._write_config(tmp_path, "- just a list\n")

        with pytest.raises(ValueError, match="mapping"):
            rock_source.load_rock_sources(config, base_dir=tmp_path)

    def test_project_config_matches_the_data_folders(self):
        work_stream = pathlib.Path(__file__).resolve().parents[1]
        repo_root = work_stream.parent
        config = work_stream / "configs" / "eda_eight_rocks.yaml"

        sources = rock_source.load_rock_sources(config, base_dir=repo_root)

        assert len(sources) == 8
        assert {s.input_units for s in sources} == {"mol_kgw", "mg_L"}
