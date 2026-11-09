"""Small fake models, rock data, and configs for the optimization tests.

The fakes do not need the real model file (best.pt) or the real run data.
The tiny model uses the real v2 classes, so the tests check the real code.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import torch

from conditional_input_optimization.optimization import (
    experiment_config,
    rock_optimizer,
)
from conditional_input_optimization.optimization import (
    surrogate as surrogate_lib,
)
from conditional_model_v1 import data as v2_data
from conditional_model_v1 import models, preprocessing

FEATURE_NAMES = v2_data.CONDITION_FEATURES
H2_INDEX = v2_data.OUTPUT_FEATURES.index("H2_g_mol")
TIME_DAYS = np.array([0.0, 0.6, 1.2, 90.0, 180.0])
LOG_INDICES = tuple(
    v2_data.OUTPUT_FEATURES.index(name) for name in v2_data.LOG_OUTPUT_FEATURES
)
HIDDEN_SIZE = 8
NUM_LAYERS = 1
# Column ranges of the inputs that vary in the fake rock. CL is tied: CL = NA.
VARYING_RANGES = {
    "TEMPERATURE": (20.0, 80.0),
    "POROSITY": (0.1, 0.3),
    "WATER_VOLUME": (10.0, 50.0),
    "GAS_VOLUME": (10.0, 50.0),
    "NA": (0.1, 1.0),
}
TIED_TO = {"CL": "NA"}

_RAW_CONFIG = {
    "name": "tiny",
    "data": {"rock_config": "rocks.yaml", "cache_dir": "cache"},
    "model": {
        "checkpoint": "model/best.pt",
        "preprocessors": "model/preprocessors.pkl",
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "device": "cpu",
    },
    "split": {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1,
        "seed": 42,
        "scaler_mean_tolerance": 1.0e-6,
    },
    "target": {
        "output": "H2_g_mol",
        "start_days": 0.6,
        "end_days": 180.0,
        "eps_mol": 1.0e-5,
        "min_start_mol": 0.1,
    },
    "input_space": {"tie_r2_threshold": 0.99, "context_inputs": []},
    "constraints": {
        "penalty_weight": 100.0,
        "feasibility_tolerance": 0.01,
        "linear": [
            {
                "name": "gas_and_water_fit_in_pore_space",
                "coefficients": {
                    "GAS_VOLUME": 1.0,
                    "WATER_VOLUME": 1.0,
                    "POROSITY": -1000.0,
                },
                "rocks": "all",
            }
        ],
    },
    "search": {
        "objective": "relative",
        "modes": ["bounded", "unbounded"],
        "n_best_starts": 2,
        "n_random_starts": 3,
        "steps": 4,
        "learning_rate": 0.05,
        "seed": 0,
    },
    "report": {
        "top_k": 2,
        "dedup_fraction_of_nn": 0.25,
        "nn_sample_size": 10,
        "batch_size": 7,
        "distance_chunk_size": 4,
    },
    "acceptance": {"min_spearman": 0.7, "max_mae_over_std": 0.5},
    "rocks": ["RockA"],
    "results_dir": "results",
}


@dataclasses.dataclass(frozen=True, eq=False)
class TinySurrogate:
    """A tiny frozen surrogate and the v2 objects that built it."""

    surrogate: surrogate_lib.FrozenSurrogate
    model: models.ConditionTimeLSTM
    bundle: preprocessing.PreprocessorBundle


def raw_config() -> dict:
    """Return a deep copy of the raw tiny config. A test can change it."""
    return copy.deepcopy(_RAW_CONFIG)


def tiny_config() -> experiment_config.ExperimentConfig:
    """Return the parsed tiny config."""
    return experiment_config.parse_config(raw_config())


def fake_conditions(n_runs: int, seed: int) -> np.ndarray:
    """Return the inputs of one fake rock, shape (n_runs, 57), all > 0.

    The inputs in VARYING_RANGES are random. CL is equal to NA. All other
    inputs are fixed.
    """
    rng = np.random.default_rng(seed)
    conditions = np.tile(1.0 + 0.1 * np.arange(len(FEATURE_NAMES)), (n_runs, 1))
    for name, (low, high) in VARYING_RANGES.items():
        column = FEATURE_NAMES.index(name)
        conditions[:, column] = rng.uniform(low, high, n_runs)
    for tied, source in TIED_TO.items():
        conditions[:, FEATURE_NAMES.index(tied)] = conditions[
            :, FEATURE_NAMES.index(source)
        ]
    return conditions


def fake_h2(n_runs: int, time_days: np.ndarray, seed: int) -> np.ndarray:
    """Return a decreasing H2 amount, shape (n_runs, T), in mol, all > 0.1."""
    rng = np.random.default_rng(seed)
    start = rng.uniform(0.5, 1.0, n_runs)
    rate = rng.uniform(0.05, 0.5, n_runs)
    return start[:, None] * np.exp(-rate[:, None] * time_days / 180.0)


def tiny_bundle(
    conditions: np.ndarray, time_days: np.ndarray, seed: int = 0
) -> preprocessing.PreprocessorBundle:
    """Fit the real v2 scalers on the inputs and on random positive outputs."""
    rng = np.random.default_rng(seed)
    outputs = rng.uniform(
        0.5, 2.0, (conditions.shape[0], len(time_days), surrogate_lib.N_OUTPUTS)
    )
    return preprocessing.PreprocessorBundle(
        condition_scaler=preprocessing.ConditionScaler().fit(conditions),
        output_scaler=preprocessing.OutputScaler(LOG_INDICES).fit(outputs),
        time_mean=float(time_days.mean()),
        time_std=float(time_days.std()),
    )


def tiny_model(
    seed: int = 0, dtype: torch.dtype = torch.float32
) -> models.ConditionTimeLSTM:
    """Return a random v2 ConditionTimeLSTM with hidden size 8, 1 layer."""
    torch.manual_seed(seed)
    model = models.ConditionTimeLSTM(
        input_size=surrogate_lib.N_CONDITIONS + 1,
        output_size=surrogate_lib.N_OUTPUTS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )
    return model.to(dtype)


def tiny_surrogate(
    conditions: np.ndarray,
    time_days: np.ndarray = TIME_DAYS,
    *,
    dtype: torch.dtype = torch.float32,
) -> TinySurrogate:
    """Build a FrozenSurrogate from a tiny model and fitted v2 scalers."""
    bundle = tiny_bundle(conditions, time_days)
    model = tiny_model(dtype=dtype)
    surrogate = surrogate_lib.FrozenSurrogate(
        model, surrogate_lib.SurrogateScaling.from_bundle(bundle), time_days
    )
    return TinySurrogate(surrogate=surrogate, model=model, bundle=bundle)


def fake_rock_data(
    surrogate: surrogate_lib.FrozenSurrogate,
    conditions: np.ndarray,
    rock: str = "RockA",
) -> rock_optimizer.RockData:
    """Return the RockData of the fake runs. The first 70% are train runs."""
    n_runs = conditions.shape[0]
    with torch.no_grad():
        predicted = surrogate.predict(
            torch.as_tensor(conditions, dtype=surrogate.dtype)
        )[:, :, H2_INDEX]
    return rock_optimizer.RockData(
        rock=rock,
        run_ids=np.array([f"{rock}_{i}" for i in range(n_runs)]),
        conditions=conditions,
        real_h2=fake_h2(n_runs, surrogate.time_days, seed=1),
        predicted_h2=predicted.numpy().astype(np.float64),
        is_train=np.arange(n_runs) < int(0.7 * n_runs),
    )
