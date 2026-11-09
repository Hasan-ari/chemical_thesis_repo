"""Tests of conditional_input_optimization.run_data.run_table."""

from __future__ import annotations

import concurrent.futures

import numpy as np
import pytest

from conditional_input_optimization.run_data import rock_source, run_table
from conditional_input_optimization.tests import synthetic_runs

_N_FEATURES = len(run_table.OUTPUT_FEATURES)


def _sources(tmp_path, runs_per_rock=None):
    """Write two synthetic rocks. The run value scale is the run id."""
    runs_per_rock = runs_per_rock or {"Halite": ("10", "2"), "Trona": ("1",)}
    sources = []
    for name, run_ids in runs_per_rock.items():
        folder = tmp_path / name
        for run_id in run_ids:
            synthetic_runs.write_run(folder, run_id, scale=float(run_id))
        sources.append(
            rock_source.RockSource(
                name=name, folder=folder, input_units="mol_kgw"
            )
        )
    return sources


def _build(sources, chunk_size=1):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        return run_table.build_run_table(sources, executor, chunk_size)


def _read_only(array):
    array = np.array(array)
    array.setflags(write=False)
    return array


def _table(**changes):
    fields = {
        "rocks": ["Halite", "Trona"],
        "run_ids": ["1", "2"],
        "conditions": np.zeros((2, 57)),
        "time_days": np.zeros(3),
        "outputs": np.zeros((_N_FEATURES, 2, 3)),
    }
    fields.update(changes)
    return run_table.RunTable(
        **{name: _read_only(value) for name, value in fields.items()}
    )


class TestBuildRunTable:
    def test_rows_follow_source_order_then_numeric_run_id(self, tmp_path):
        table = _build(_sources(tmp_path))

        assert table.rocks.tolist() == ["Halite", "Halite", "Trona"]
        assert table.run_ids.tolist() == ["2", "10", "1"]

    def test_outputs_have_feature_run_time_layout(self, tmp_path):
        table = _build(_sources(tmp_path))

        assert table.outputs.shape == (_N_FEATURES, 3, 3)
        expected = synthetic_runs.output_columns(scale=10.0)
        np.testing.assert_array_equal(
            table.output("H2_g_mol")[1], expected["H2_g_mol"]
        )
        np.testing.assert_allclose(table.time_days, [0.0, 0.6, 1.2])

    def test_conditions_have_one_row_for_each_run(self, tmp_path):
        table = _build(_sources(tmp_path))

        assert table.conditions.shape == (3, 57)
        np.testing.assert_array_equal(table.conditions[0], table.conditions[2])

    def test_chunk_size_does_not_change_the_result(self, tmp_path):
        sources = _sources(tmp_path)

        small = _build(sources, chunk_size=1)
        large = _build(sources, chunk_size=100)

        np.testing.assert_array_equal(small.outputs, large.outputs)
        np.testing.assert_array_equal(small.run_ids, large.run_ids)

    def test_all_arrays_are_read_only(self, tmp_path):
        table = _build(_sources(tmp_path))

        with pytest.raises(ValueError, match="read-only"):
            table.outputs[0, 0, 0] = 1.0

    def test_different_time_axis_raises(self, tmp_path):
        sources = _sources(tmp_path)
        synthetic_runs.write_run(sources[1].folder, "5", n_steps=4)

        with pytest.raises(ValueError, match="different time axis"):
            _build(sources, chunk_size=100)

    def test_no_complete_run_raises(self, tmp_path):
        (tmp_path / "Empty" / "input").mkdir(parents=True)
        (tmp_path / "Empty" / "output").mkdir()
        source = rock_source.RockSource(
            name="Empty", folder=tmp_path / "Empty", input_units="mol_kgw"
        )

        with pytest.raises(ValueError, match="no complete run"):
            _build([source])

    def test_non_positive_chunk_size_raises(self, tmp_path):
        with pytest.raises(ValueError, match="chunk_size"):
            _build(_sources(tmp_path), chunk_size=0)

    def test_process_pool_gives_the_same_table(self, tmp_path):
        sources = _sources(tmp_path)

        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
            table = run_table.build_run_table(sources, pool, chunk_size=1)

        np.testing.assert_array_equal(table.outputs, _build(sources).outputs)


class TestRunTable:
    def test_output_returns_a_view_of_one_feature(self):
        table = _table()

        h2 = table.output("H2_g_mol")

        assert h2.shape == (2, 3)
        assert np.shares_memory(h2, table.outputs)

    def test_unknown_output_raises(self):
        with pytest.raises(KeyError, match="pH3"):
            _table().output("pH3")

    def test_writable_array_raises(self):
        with pytest.raises(ValueError, match="read-only"):
            run_table.RunTable(
                rocks=_read_only(["Halite"]),
                run_ids=_read_only(["1"]),
                conditions=np.zeros((1, 57)),
                time_days=_read_only(np.zeros(3)),
                outputs=_read_only(np.zeros((_N_FEATURES, 1, 3))),
            )

    def test_wrong_outputs_shape_raises(self):
        with pytest.raises(ValueError, match="outputs shape"):
            _table(outputs=np.zeros((_N_FEATURES, 2, 4)))

    def test_wrong_run_ids_length_raises(self):
        with pytest.raises(ValueError, match="run_ids shape"):
            _table(run_ids=["1"])
