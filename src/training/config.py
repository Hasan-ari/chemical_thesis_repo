from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

from src.data.constants import LOG_TRANSFORM_COLS, N_FEATURES


@dataclass
class ExperimentConfig:
    """Complete configuration for one training run."""

    # Data
    data_dir: str = "data/phreeqc_v23"
    test_ratio: float = 0.1

    # Architecture
    n_features: int = N_FEATURES
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    seq_len: int = 10

    # Training
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # Preprocessing
    log_cols: tuple = LOG_TRANSFORM_COLS

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "experiments"
    experiment_name: str = ""

    def __post_init__(self):
        if not self.experiment_name:
            self.experiment_name = f"seq{self.seq_len}_h{self.hidden_size}"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["log_cols"] = list(data["log_cols"])
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        with open(path) as f:
            data = json.load(f)
        data["log_cols"] = tuple(data["log_cols"])
        return cls(**data)


def build_experiment_matrix() -> List[ExperimentConfig]:
    """Generate configs for the full experiment grid.

    Grid: seq_len x hidden_size
    """
    configs = []
    for seq_len in [3, 5, 10, 20]:
        for hidden_size in [64, 128, 256]:
            configs.append(
                ExperimentConfig(
                    seq_len=seq_len,
                    hidden_size=hidden_size,
                )
            )
    return configs
