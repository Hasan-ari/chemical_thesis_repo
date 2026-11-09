"""Tests of conditional_input_optimization.optimization.experiment_config."""

from __future__ import annotations

import pathlib

import pytest
import yaml

from conditional_input_optimization.optimization import (
    experiment,
    experiment_config,
)
from conditional_input_optimization.tests import optimization_fakes as fakes

_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "configs"
    / "optimization_v1.yaml"
)


def test_the_v1_config_file_loads():
    config = experiment_config.load_config(_CONFIG_PATH)
    assert config.name == "optimization_v1"
    assert config.target.output == "H2_g_mol"
    assert config.search.modes == ("bounded", "unbounded")
    assert config.search.n_best_starts + config.search.n_random_starts == 64
    assert len(config.rocks) == 8
    assert len(config.constraints.linear) == 2
    assert config.target.eps_mol == config.target.min_start_mol
    assert config.model.checkpoint == pathlib.Path(
        "conditional_input_optimization/models/v2/best.pt"
    )


def test_parse_config_of_the_tiny_config():
    config = fakes.tiny_config()
    assert config.input_space.context_inputs == ()
    assert config.acceptance.min_spearman == 0.7


@pytest.mark.parametrize(
    ("section", "key"), [(None, "colour"), ("search", "momentum")]
)
def test_unknown_key_raises(section, key):
    raw = fakes.raw_config()
    (raw if section is None else raw[section])[key] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        experiment_config.parse_config(raw)


def test_missing_key_raises():
    raw = fakes.raw_config()
    del raw["report"]["top_k"]
    with pytest.raises(ValueError, match="missing keys"):
        experiment_config.parse_config(raw)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("split", "test", 0.2, "must be 1"),
        ("search", "modes", ["free"], "search.modes"),
        ("search", "modes", [], "must not be empty"),
        ("search", "objective", "ratio", "search.objective"),
        ("search", "learning_rate", 0.0, "learning_rate"),
        ("target", "output", "H2_mol", "target.output"),
        ("target", "start_days", 200.0, "start_days"),
        ("input_space", "context_inputs", ["SALT"], "context_inputs"),
        ("report", "top_k", 0, "top_k"),
        ("model", "hidden_size", 0, "hidden_size"),
        ("acceptance", "max_mae_over_std", -1.0, "max_mae_over_std"),
    ],
)
def test_invalid_value_raises(section, key, value, message):
    raw = fakes.raw_config()
    raw[section][key] = value
    with pytest.raises(ValueError, match=message):
        experiment_config.parse_config(raw)


def test_eps_above_min_start_raises():
    raw = fakes.raw_config()
    raw["target"]["eps_mol"] = 2 * raw["target"]["min_start_mol"]
    with pytest.raises(ValueError, match="eps_mol must be <="):
        experiment_config.parse_config(raw)


def test_duplicate_rocks_raise():
    raw = fakes.raw_config()
    raw["rocks"] = ["RockA", "RockA"]
    with pytest.raises(ValueError, match="duplicate"):
        experiment_config.parse_config(raw)


def test_load_config_rejects_a_list(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump([1, 2]))
    with pytest.raises(ValueError, match="mapping"):
        experiment_config.load_config(path)


def test_apply_overrides():
    config = fakes.tiny_config()
    changed = experiment.apply_overrides(config, ["RockB"], 7)
    assert changed.rocks == ("RockB",)
    assert changed.search.steps == 7
    assert experiment.apply_overrides(config, None, None) == config
    with pytest.raises(ValueError, match="steps"):
        experiment.apply_overrides(config, None, -1)
