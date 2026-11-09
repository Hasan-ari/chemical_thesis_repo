"""Tests of conditional_input_optimization.run_data.run_reader."""

from __future__ import annotations

import numpy as np
import pytest

from conditional_input_optimization.run_data import rock_source, run_reader
from conditional_input_optimization.tests import synthetic_runs
from conditional_model_v1 import units

_FEATURES = run_reader.CONDITION_FEATURES


_OUTPUTS = run_reader.OUTPUT_FEATURES


def _write_run(folder, run_id="1", values=None, output=None):
    values = values or synthetic_runs.required_input_values()
    output = output or synthetic_runs.output_columns()
    synthetic_runs.write_input(folder, run_id, values)
    synthetic_runs.write_output(folder, run_id, output)


def _reader(folder, input_units="mol_kgw"):
    source = rock_source.RockSource(
        name="Calcite", folder=folder, input_units=input_units
    )
    return run_reader.RunFileReader(source)


def _record(**changes):
    fields = {
        "rock": "Calcite",
        "run_id": "1",
        "conditions": np.zeros(len(_FEATURES)),
        "time_days": np.array([0.0, 0.6]),
        "outputs": np.zeros((2, len(_OUTPUTS))),
    }
    fields.update(changes)
    return run_reader.RunRecord(**fields)


class TestRunFileReader:
    def test_conditions_follow_the_v2_feature_order(self, tmp_path):
        values = synthetic_runs.required_input_values()
        _write_run(tmp_path, values=values)

        record = _reader(tmp_path).read("1")

        assert record.conditions.shape == (57,)
        for key, value in values.items():
            assert record.conditions[_FEATURES.index(key)] == value

    def test_absent_features_are_zero(self, tmp_path):
        values = synthetic_runs.required_input_values()
        _write_run(tmp_path, values=values)

        record = _reader(tmp_path).read("1")

        absent = [i for i, name in enumerate(_FEATURES) if name not in values]
        assert absent
        assert np.all(record.conditions[absent] == 0.0)

    def test_reads_time_and_all_output_columns(self, tmp_path):
        _write_run(tmp_path)

        record = _reader(tmp_path).read("1")

        assert record.rock == "Calcite"
        assert record.run_id == "1"
        np.testing.assert_allclose(record.time_days, [0.0, 0.6, 1.2])
        assert record.outputs.shape == (3, 26)
        h2 = _OUTPUTS.index("H2_g_mol")
        np.testing.assert_array_equal(
            record.outputs[:, h2], [100.0 * h2, 100.0 * h2 + 1, 100.0 * h2 + 2]
        )

    def test_columns_are_found_by_name_not_position(self, tmp_path):
        columns = synthetic_runs.output_columns()
        shuffled = {"Extra_col": [9.0, 9.0, 9.0]}
        shuffled.update(reversed(list(columns.items())))
        _write_run(tmp_path, output=shuffled)

        record = _reader(tmp_path).read("1")

        expected = np.array([columns[name] for name in _OUTPUTS]).T
        np.testing.assert_array_equal(record.outputs, expected)

    def test_mg_l_values_are_converted(self, tmp_path):
        values = synthetic_runs.required_input_values()
        _write_run(tmp_path, values=values)

        record = _reader(tmp_path, input_units="mg_L").read("1")

        expected = units.convert_solution_units(
            {"NA": values["NA"]}, input_units="mg_L"
        )["NA"]
        assert record.conditions[_FEATURES.index("NA")] == expected
        assert expected != values["NA"]

    def test_missing_h2_column_raises(self, tmp_path):
        output = synthetic_runs.output_columns()
        del output["H2_g_mol"]
        _write_run(tmp_path, output=output)

        with pytest.raises(ValueError, match="H2_g_mol"):
            _reader(tmp_path).read("1")

    def test_missing_output_file_raises(self, tmp_path):
        synthetic_runs.write_input(
            tmp_path, "1", synthetic_runs.required_input_values()
        )

        with pytest.raises(FileNotFoundError):
            _reader(tmp_path).read("1")


class TestRunRecord:
    def test_arrays_are_read_only(self):
        record = _record()

        with pytest.raises(ValueError, match="read-only"):
            record.outputs[0, 0] = 1.0

    def test_stores_a_copy_of_the_caller_arrays(self):
        outputs = np.zeros((2, len(_OUTPUTS)))
        record = _record(outputs=outputs)

        outputs[0, 0] = 99.0

        assert record.outputs[0, 0] == 0.0
        assert outputs.flags.writeable

    def test_wrong_conditions_shape_raises(self):
        with pytest.raises(ValueError, match="conditions shape"):
            _record(conditions=np.zeros(3))

    def test_unequal_time_and_output_lengths_raise(self):
        with pytest.raises(ValueError, match="do not agree"):
            _record(outputs=np.zeros((3, len(_OUTPUTS))))

    def test_wrong_output_feature_count_raises(self):
        with pytest.raises(ValueError, match="do not agree"):
            _record(outputs=np.zeros((2, 5)))

    def test_non_finite_value_raises(self):
        outputs = np.zeros((2, len(_OUTPUTS)))
        outputs[1, 3] = np.nan

        with pytest.raises(ValueError, match="non-finite"):
            _record(outputs=outputs)
