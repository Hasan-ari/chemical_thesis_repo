"""Tests of conditional_input_optimization.run_data.run_table_cache."""

from __future__ import annotations

import concurrent.futures

import numpy as np
import pytest

from conditional_input_optimization.run_data import (
    rock_source,
    run_table,
    run_table_cache,
)
from conditional_input_optimization.tests import synthetic_runs


def _source(tmp_path, run_ids=("1", "2")):
    folder = tmp_path / "Halite"
    for run_id in run_ids:
        synthetic_runs.write_run(folder, run_id, scale=float(run_id))
    return rock_source.RockSource(
        name="Halite", folder=folder, input_units="mol_kgw"
    )


def _build(sources):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return run_table.build_run_table(sources, executor)


class TestSourceFingerprint:
    def test_is_stable_for_the_same_files(self, tmp_path):
        sources = [_source(tmp_path)]

        first = run_table_cache.source_fingerprint(sources)

        assert first == run_table_cache.source_fingerprint(sources)
        assert len(first) == 64

    def test_changes_when_a_file_changes(self, tmp_path):
        sources = [_source(tmp_path)]
        before = run_table_cache.source_fingerprint(sources)

        synthetic_runs.write_run(sources[0].folder, "2", n_steps=4)

        assert run_table_cache.source_fingerprint(sources) != before

    def test_changes_when_a_run_is_added(self, tmp_path):
        sources = [_source(tmp_path)]
        before = run_table_cache.source_fingerprint(sources)

        synthetic_runs.write_run(sources[0].folder, "3")

        assert run_table_cache.source_fingerprint(sources) != before

    def test_changes_when_the_units_change(self, tmp_path):
        source = _source(tmp_path)
        other = rock_source.RockSource(
            name=source.name, folder=source.folder, input_units="mg_L"
        )

        assert run_table_cache.source_fingerprint(
            [source]
        ) != run_table_cache.source_fingerprint([other])

    def test_missing_folder_raises(self, tmp_path):
        source = rock_source.RockSource(
            name="None", folder=tmp_path / "None", input_units="mol_kgw"
        )

        with pytest.raises(FileNotFoundError):
            run_table_cache.source_fingerprint([source])


class TestSaveAndLoad:
    def test_round_trip_keeps_all_values(self, tmp_path):
        sources = [_source(tmp_path)]
        table = _build(sources)
        cache_dir = tmp_path / "cache"

        run_table_cache.save_run_table(table, cache_dir, "abc")
        loaded = run_table_cache.load_run_table(cache_dir, "abc")

        for name in ("rocks", "run_ids", "conditions", "time_days", "outputs"):
            np.testing.assert_array_equal(
                getattr(loaded, name), getattr(table, name)
            )

    def test_outputs_are_memory_mapped_and_read_only(self, tmp_path):
        cache_dir = tmp_path / "cache"
        run_table_cache.save_run_table(
            _build([_source(tmp_path)]), cache_dir, "abc"
        )

        loaded = run_table_cache.load_run_table(cache_dir, "abc")

        assert isinstance(loaded.outputs, np.memmap)
        with pytest.raises(ValueError, match="read-only"):
            loaded.conditions[0, 0] = 1.0

    def test_other_fingerprint_raises(self, tmp_path):
        cache_dir = tmp_path / "cache"
        run_table_cache.save_run_table(
            _build([_source(tmp_path)]), cache_dir, "abc"
        )

        assert not run_table_cache.is_cache_valid(cache_dir, "xyz")
        with pytest.raises(run_table_cache.CacheMismatchError):
            run_table_cache.load_run_table(cache_dir, "xyz")

    def test_absent_cache_is_not_valid(self, tmp_path):
        assert not run_table_cache.is_cache_valid(tmp_path / "none", "abc")


class TestLoadOrBuildRunTable:
    def test_builds_one_time_then_loads(self, tmp_path, monkeypatch):
        sources = [_source(tmp_path)]
        cache_dir = tmp_path / "cache"
        calls = []
        real_build = run_table.build_run_table

        def counting_build(*args, **kwargs):
            calls.append(1)
            return real_build(*args, **kwargs)

        monkeypatch.setattr(run_table, "build_run_table", counting_build)

        first = run_table_cache.load_or_build_run_table(
            sources, cache_dir, max_workers=1
        )
        second = run_table_cache.load_or_build_run_table(
            sources, cache_dir, max_workers=1
        )

        assert len(calls) == 1
        np.testing.assert_array_equal(first.outputs, second.outputs)

    def test_rebuilds_when_a_source_file_changes(self, tmp_path):
        sources = [_source(tmp_path)]
        cache_dir = tmp_path / "cache"
        run_table_cache.load_or_build_run_table(
            sources, cache_dir, max_workers=1
        )

        synthetic_runs.write_run(sources[0].folder, "3", scale=3.0)
        table = run_table_cache.load_or_build_run_table(
            sources, cache_dir, max_workers=1
        )

        assert table.run_ids.tolist() == ["1", "2", "3"]
