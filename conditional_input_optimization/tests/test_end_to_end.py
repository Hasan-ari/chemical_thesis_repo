"""End-to-end tests of the optimization with a tiny model and fake runs."""

from __future__ import annotations

import dataclasses
import datetime
import math

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from conditional_input_optimization.optimization import (
    experiment,
    objectives,
    report,
    rock_optimizer,
)
from conditional_input_optimization.run_data import (
    rock_source,
    run_table_cache,
)
from conditional_input_optimization.tests import optimization_fakes as fakes
from conditional_input_optimization.tests import synthetic_runs

_N_RUNS = 30
_N_STEPS_ON_DISK = 301
_RESULT_FILES = {
    "README.md",
    "all_candidates.csv",
    "baseline.csv",
    "config.yaml",
    "h2_trajectories.csv",
    "input_space.csv",
    "manifest.yaml",
    "optimized_inputs.csv",
    "optimized_inputs_file_units.csv",
    "search_history.csv",
    "summary.md",
    "surrogate_check.csv",
}


@pytest.fixture(name="rock")
def _rock():
    conditions = fakes.fake_conditions(_N_RUNS, seed=0)
    tiny = fakes.tiny_surrogate(conditions)
    return tiny, fakes.fake_rock_data(tiny.surrogate, conditions)


def _optimize(rock, config=None):
    tiny, data = rock
    return rock_optimizer.optimize_rock(
        data,
        tiny.surrogate,
        config or fakes.tiny_config(),
        rock_index=0,
        error_p95=0.01,
    )


def test_optimize_rock_returns_one_table_for_each_mode(rock):
    result = _optimize(rock)
    config = fakes.tiny_config()
    n_starts = config.search.n_best_starts + config.search.n_random_starts
    assert [mode.mode for mode in result.modes] == ["bounded", "unbounded"]
    for mode in result.modes:
        table = mode.candidates
        assert len(table) == n_starts
        assert list(table.columns[-57:]) == list(fakes.FEATURE_NAMES)
        assert table["start_kind"].tolist() == ["best"] * 2 + ["random"] * 3
        ranked = table[table["rank"].notna()]
        assert sorted(ranked["rank"]) == list(range(1, len(ranked) + 1))
        assert ranked["usable"].all()
        assert mode.trajectories.shape == (n_starts, len(fakes.TIME_DAYS))
        assert mode.history.shape == (config.search.steps + 1, n_starts)
    assert result.baseline["n_eligible"] == _N_RUNS


def test_best_starts_have_the_smallest_real_loss(rock):
    _, data = rock
    config = fakes.tiny_config()
    target = config.target
    window = objectives.TimeWindow.from_days(
        fakes.TIME_DAYS, target.start_days, target.end_days
    )
    objective = objectives.build_objective(
        config.search.objective, window, target.eps_mol
    )
    real = objective(data.real_h2)
    real[data.real_h2[:, window.start_index] < target.min_start_mol] = np.inf
    expected = np.sort(real)[: config.search.n_best_starts]
    for mode in _optimize(rock).modes:
        best = mode.candidates[mode.candidates["start_kind"] == "best"]
        np.testing.assert_allclose(best["start_real_loss"], expected)


def test_candidates_keep_ties_and_bounded_mode_keeps_range(rock):
    result = _optimize(rock)
    names = list(fakes.FEATURE_NAMES)
    for mode in result.modes:
        np.testing.assert_allclose(mode.candidates["CL"], mode.candidates["NA"])
    bounded = result.modes[0].candidates[names].to_numpy()
    free = list(result.space.detection.free_columns)
    span = result.space.observed_max[free] - result.space.observed_min[free]
    slack = 1e-5 * span
    assert (bounded[:, free] >= result.space.observed_min[free] - slack).all()
    assert (bounded[:, free] <= result.space.observed_max[free] + slack).all()


def test_optimize_rock_is_reproducible(rock):
    first = _optimize(rock).modes[1].candidates
    second = _optimize(rock).modes[1].candidates
    pd.testing.assert_frame_equal(first, second)


def test_context_input_stays_at_the_start_value(rock):
    config = fakes.tiny_config()
    config = dataclasses.replace(
        config,
        input_space=dataclasses.replace(
            config.input_space, context_inputs=("TEMPERATURE",)
        ),
    )
    _, data = rock
    for mode in _optimize(rock, config).modes:
        table = mode.candidates
        starts = dict(zip(data.run_ids, data.conditions[:, 0], strict=True))
        expected = table["start_run_id"].map(starts).to_numpy()
        np.testing.assert_allclose(table["TEMPERATURE"], expected, rtol=1e-6)


def test_context_control_mask(rock):
    tiny, data = rock
    space = _optimize(rock).space
    mask = rock_optimizer.context_control_mask(space, ["TEMPERATURE"])
    assert mask.tolist() == [False, True, True, True, True]
    fixed = rock_optimizer.context_control_mask(space, ["SOLID_MASS"])
    assert fixed.all()
    with pytest.raises(ValueError, match="tied in rock 'RockA'"):
        rock_optimizer.context_control_mask(space, ["CL"])


def test_rock_data_rejects_bad_shapes(rock):
    _, data = rock
    with pytest.raises(ValueError, match="one row for each run"):
        dataclasses.replace(data, is_train=data.is_train[:3])
    with pytest.raises(ValueError, match="57"):
        dataclasses.replace(data, conditions=data.conditions[:, :5])


def _labels(n_runs):
    labels = np.full(n_runs, experiment.SPLIT_VAL, dtype=object)
    labels[: int(0.7 * n_runs)] = experiment.SPLIT_TRAIN
    labels[int(0.7 * n_runs) : int(0.7 * n_runs) + 5] = experiment.SPLIT_TEST
    return labels


def _write_model_files(tmp_path, tiny):
    folder = tmp_path / "model"
    folder.mkdir()
    torch.save(tiny.model.state_dict(), folder / "best.pt")
    tiny.bundle.save(folder / "preprocessors.pkl")


def test_write_results_writes_each_file(rock, tmp_path):
    tiny, data = rock
    config = fakes.tiny_config()
    _write_model_files(tmp_path, tiny)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(fakes.raw_config()))
    checks = experiment.check_rock(
        data, _labels(_N_RUNS), config, fakes.TIME_DAYS
    )
    result = experiment.ExperimentResult(
        config=config,
        repo_root=tmp_path,
        rock_results=(_optimize(rock),),
        surrogate_checks=pd.DataFrame(checks),
        input_units={"RockA": "mg_L"},
        time_days=fakes.TIME_DAYS,
        scaler_gap=0.0,
        started_at=datetime.datetime(2026, 1, 2, 3, 4, 5),
        run_seconds=1.5,
    )
    folder = report.write_results(result, config_path, tmp_path / "results")
    assert folder.name == "20260102_030405_tiny"
    assert {path.name for path in folder.iterdir()} == _RESULT_FILES
    _check_tables(folder)
    manifest = yaml.safe_load((folder / "manifest.yaml").read_text())
    assert manifest["git_commit"] is None
    assert manifest["finished_at"] == "2026-01-02T03:04:06.500000"
    with pytest.raises(FileExistsError):
        report.write_results(result, config_path, tmp_path / "results")


def _check_tables(folder):
    optimized = pd.read_csv(folder / "optimized_inputs.csv")
    assert tuple(optimized.columns[: len(report.LEADING_COLUMNS)]) == (
        report.LEADING_COLUMNS
    )
    assert optimized["rank"].notna().all()
    file_units = pd.read_csv(folder / "optimized_inputs_file_units.csv")
    assert set(file_units["input_units"]) == {"mg_L"}
    assert (file_units["NA"] > optimized["NA"]).all()
    trajectories = pd.read_csv(folder / "h2_trajectories.csv")
    assert len(trajectories) == len(optimized) * len(fakes.TIME_DAYS)
    checks = pd.read_csv(folder / "surrogate_check.csv")
    assert checks["objective"].tolist() == ["relative"] * 2 + ["absolute"] * 2
    assert checks["split"].tolist() == ["test", "train"] * 2
    summary = (folder / "summary.md").read_text()
    assert "test runs, absolute loss: resolvable" in summary
    roles = pd.read_csv(folder / "input_space.csv")
    tied = roles.loc[roles["input"] == "CL"].squeeze()
    assert tied["role"] == "tied"
    assert "NA" in tied["sources"].split(";")


def _write_rock_runs(folder, n_runs, seed):
    """Write n_runs synthetic runs. Five inputs vary and CL = NA."""
    rng = np.random.default_rng(seed)
    time_days = 0.6 * np.arange(_N_STEPS_ON_DISK)
    for index in range(n_runs):
        values = synthetic_runs.required_input_values()
        for name, (low, high) in fakes.VARYING_RANGES.items():
            values[name] = float(rng.uniform(low, high))
        values["CL"] = values["NA"]
        run_id = str(index + 1)
        synthetic_runs.write_input(folder, run_id, values)
        columns = synthetic_runs.output_columns(_N_STEPS_ON_DISK)
        start, rate = rng.uniform(0.5, 1.0), rng.uniform(0.05, 0.5)
        columns["H2_g_mol"] = list(start * np.exp(-rate * time_days / 180))
        synthetic_runs.write_output(folder, run_id, columns)


def _prepare_repo(tmp_path):
    """Write two rocks, a rock config, a run config, and model files."""
    rock_units = {"RockA": "mol_kgw", "RockB": "mg_L"}
    for seed, rock in enumerate(rock_units):
        _write_rock_runs(tmp_path / "data" / rock, 20, seed)
    (tmp_path / "rocks.yaml").write_text(
        yaml.safe_dump(
            {
                "data_root": "data",
                "rocks": [
                    {"name": name, "input_units": units}
                    for name, units in rock_units.items()
                ],
            }
        )
    )
    raw = fakes.raw_config()
    raw["rocks"] = ["RockB", "RockA"]
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    config = fakes.tiny_config()
    sources = rock_source.load_rock_sources(tmp_path / "rocks.yaml", tmp_path)
    table = run_table_cache.load_or_build_run_table(
        sources, tmp_path / "cache", max_workers=1
    )
    labels = experiment.split_labels(table.rocks, config.split)
    train = np.asarray(table.conditions[labels == experiment.SPLIT_TRAIN])
    bundle = fakes.tiny_bundle(train, np.asarray(table.time_days))
    tiny = fakes.TinySurrogate(
        surrogate=None, model=fakes.tiny_model(), bundle=bundle
    )
    _write_model_files(tmp_path, tiny)
    return config_path


def test_main_runs_on_synthetic_run_files(tmp_path, monkeypatch):
    config_path = _prepare_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    experiment.main(["--config", str(config_path), "--steps", "2"])
    (folder,) = (tmp_path / "results").iterdir()
    assert {path.name for path in folder.iterdir()} == _RESULT_FILES
    manifest = yaml.safe_load((folder / "manifest.yaml").read_text())
    assert manifest["steps"] == 2
    assert manifest["rocks"] == ["RockB", "RockA"]
    assert manifest["scaler_mean_gap"] < 1e-6
    candidates = pd.read_csv(folder / "all_candidates.csv")
    assert candidates["rock"].unique().tolist() == ["RockB", "RockA"]
    checks = pd.read_csv(folder / "surrogate_check.csv")
    assert len(checks) == 8
    assert all(math.isfinite(value) for value in checks["mae"])


def test_run_experiment_stops_on_a_wrong_scaler(tmp_path):
    config_path = _prepare_repo(tmp_path)
    bundle_path = tmp_path / "model" / "preprocessors.pkl"
    conditions = fakes.fake_conditions(10, seed=9)
    fakes.tiny_bundle(conditions, fakes.TIME_DAYS).save(bundle_path)
    config = experiment.apply_overrides(
        fakes.tiny_config(), ["RockA", "RockB"], 1
    )
    assert config_path.is_file()
    with pytest.raises(ValueError, match="does not match the v2 scaler"):
        experiment.run_experiment(config, tmp_path)


def test_run_experiment_rejects_an_unknown_rock(tmp_path):
    _prepare_repo(tmp_path)
    config = experiment.apply_overrides(fakes.tiny_config(), ["RockC"], 1)
    with pytest.raises(ValueError, match="no runs of the rocks"):
        experiment.run_experiment(config, tmp_path)
