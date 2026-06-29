from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    """One raw PHREEQC dataset folder in the experiment registry."""

    name: str
    rock: str
    path: str
    max_runs: int | None = None


@dataclass(frozen=True)
class SplitConfig:
    """Run-level split ratios. Runs never cross train/val/test boundaries."""

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42
    strategy: Literal["rock_aware_run_level"] = "rock_aware_run_level"


@dataclass(frozen=True)
class DataConfig:
    """Data paths and cache/export behavior.

    `data_root`, `processed_root`, and `run_root` should come from environment
    variables in Colab so committed configs do not leak private Drive paths.
    """

    datasets: tuple[DatasetConfig, ...]
    processed_root: str
    use_cache: bool = True
    rebuild_cache: bool = False
    write_outputs_csv: bool = False
    split: SplitConfig = field(default_factory=SplitConfig)


@dataclass(frozen=True)
class ModelConfig:
    """Many-to-many LSTM model settings."""

    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.0


@dataclass(frozen=True)
class SchedulerConfig:
    """Learning-rate scheduler settings."""

    type: Literal["none", "reduce_on_plateau", "cosine", "step"] = "none"
    factor: float = 0.5
    patience: int = 10
    min_lr: float = 1e-6
    step_size: int = 50
    gamma: float = 0.5


@dataclass(frozen=True)
class TrainingConfig:
    """Plain PyTorch training-loop settings."""

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    num_workers: int = 0
    seed: int = 42
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass(frozen=True)
class PlotConfig:
    """Prediction plot settings.

    Full numeric predictions are always saved separately. These settings only
    control how many PNG files we render for human inspection.
    Use `max_runs=None` to render every run in the evaluation split.
    """

    max_runs: int | None = 3
    features: tuple[str, ...] | Literal["all"] = "all"


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment config loaded from YAML."""

    name: str
    run_root: str
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)


def load_config(path: Path | str) -> ExperimentConfig:
    """Load a YAML config and expand `${ENV_VAR}` placeholders."""
    raw_text = os.path.expandvars(Path(path).read_text())
    payload = yaml.safe_load(raw_text)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return parse_config(payload)


def parse_config(payload: dict[str, Any]) -> ExperimentConfig:
    """Convert a plain YAML mapping into typed config dataclasses."""
    data_payload = payload["data"]
    split_payload = data_payload.get("split", {})
    datasets = tuple(DatasetConfig(**dataset) for dataset in data_payload["datasets"])
    data_config = DataConfig(
        datasets=datasets,
        processed_root=data_payload["processed_root"],
        use_cache=bool(data_payload.get("use_cache", True)),
        rebuild_cache=bool(data_payload.get("rebuild_cache", False)),
        write_outputs_csv=bool(data_payload.get("write_outputs_csv", False)),
        split=SplitConfig(**split_payload),
    )
    training_payload = payload.get("training", {})
    scheduler_payload = training_payload.get("scheduler", {})
    training_config = TrainingConfig(
        epochs=int(training_payload.get("epochs", 100)),
        batch_size=int(training_payload.get("batch_size", 64)),
        learning_rate=float(training_payload.get("learning_rate", 1e-3)),
        weight_decay=float(training_payload.get("weight_decay", 0.0)),
        grad_clip=training_payload.get("grad_clip", 1.0),
        num_workers=int(training_payload.get("num_workers", 0)),
        seed=int(training_payload.get("seed", 42)),
        scheduler=SchedulerConfig(**scheduler_payload),
    )
    return ExperimentConfig(
        name=payload["experiment"]["name"],
        run_root=payload["experiment"]["run_root"],
        data=data_config,
        model=ModelConfig(**payload.get("model", {})),
        training=training_config,
        plots=_parse_plot_config(payload.get("plots", {})),
    )


def _parse_plot_config(payload: dict[str, Any]) -> PlotConfig:
    """Parse plot settings while keeping `features: all` ergonomic in YAML."""
    features = payload.get("features", "all")
    if features != "all":
        features = tuple(features)
    max_runs = payload.get("max_runs", 3)
    return PlotConfig(
        max_runs=None if max_runs is None else int(max_runs),
        features=features,
    )
