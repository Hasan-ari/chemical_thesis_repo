from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SCALAR_CONDITION_FEATURES: tuple[str, ...] = (
    "TEMPERATURE",
    "POROSITY",
    "WATER_VOLUME",
    "GAS_VOLUME",
    "SOLID_MASS",
    "PORE_VOLUME",
    "ALKALINITY",
    "NA",
    "MG",
    "CL",
    "CA",
    "S6",
    "H2",
    "CH4",
    "CO2",
    "N2",
    "H2S",
)

# Fixed mineral dictionary shared by every dataset, single-mineral or not.
# A run gets one slot per vocabulary mineral; minerals absent from that run's
# input file are encoded as 0.0 instead of being renamed to a generic feature.
MINERAL_VOCAB: tuple[str, ...] = (
    "ALBITE",
    "ANDALUSITE",
    "BARITE",
    "CALCITE",
    "DOLOMITE",
    "EPIDOTE",
    "HALITE",
    "HEMATITE",
    "ILLITE",
    "KAOLINITE",
    "MONTMOR_CA",
    "MONTMOR_NA",
    "MUSCOVITE",
    "PARAGONITE",
    "PYRITE",
    "QUARTZ",
    "TRONA",
)

# Deterministic block layout: all `<MINERAL>_MOLES` in MINERAL_VOCAB order first,
# then all `<MINERAL>_AREA` in MINERAL_VOCAB order (not interleaved).
MINERAL_CONDITION_FEATURES: tuple[str, ...] = (
    *(f"{mineral}_MOLES" for mineral in MINERAL_VOCAB),
    *(f"{mineral}_AREA" for mineral in MINERAL_VOCAB),
)

# 17 scalar conditions + 34 mineral-dictionary slots = 51 condition features.
# `build_condition_time_tensor` appends the normalized time channel on top (52).
CONDITION_FEATURES: tuple[str, ...] = (
    *SCALAR_CONDITION_FEATURES,
    *MINERAL_CONDITION_FEATURES,
)

OUTPUT_FEATURES: tuple[str, ...] = (
    "pH",
    "Ptot_atm",
    "pH2_atm",
    "pCO2_atm",
    "pCH4_atm",
    "H2_g_mol",
    "CO2_g_mol",
    "CH4_g_mol",
    "SO4_mol",
    "Formate",
    "Acetate",
    "Ca_mol",
    "Fe_mol",
    "X_SRB_mol",
    "X_IRB_mol",
    "X_SRB_mol_per_L",
    "Fe(OH)3",
    "SR_H2",
    "SR_FOR",
    "SR_AC",
    "IR_H2",
    "IR_AC",
    "IR_FOR",
    "Water_VOL",
    "Gas_VOL",
    "HS-_mol",
    "HCO3_mol",
    "Na_tot",
    "Mg_tot",
    "Cl_tot",
    "Ca_tot",
    "S6_tot",
)

LOG_OUTPUT_FEATURES: tuple[str, ...] = (
    "pH2_atm",
    "pCH4_atm",
    "H2_g_mol",
    "CO2_g_mol",
    "CH4_g_mol",
    "Formate",
    "Acetate",
    "Fe_mol",
    "X_SRB_mol",
    "X_IRB_mol",
    "X_SRB_mol_per_L",
    "Fe(OH)3",
    "SR_H2",
    "SR_FOR",
    "SR_AC",
    "IR_H2",
    "IR_AC",
    "IR_FOR",
    "HS-_mol",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    rock: str
    path: str | Path
    max_runs: int | None = None


@dataclass(frozen=True)
class TrajectoryBundle:
    conditions: np.ndarray
    trajectories: np.ndarray
    time_axis: np.ndarray
    run_ids: list[str]
    rocks: np.ndarray
    condition_features: tuple[str, ...]
    output_features: tuple[str, ...]


_INPUT_LINE = re.compile(r"^\{(?P<key>[^}]+)\}\s*(?P<value>.*)$")
_MINERAL_KEY = re.compile(r"([A-Z0-9_]+)_(?:MOLES|AREA)")


def run_id_from_path(path: Path) -> str:
    """Extract numeric run id from names like `1_Input.txt` or `1_Output.txt`."""
    stem = path.stem
    return stem.removesuffix("_Input").removesuffix("_Output")


def load_input_parameters(path: Path | str, spec: DatasetSpec) -> dict[str, Any]:
    """Parse one professor-generated `*_Input.txt` file into generic numeric conditions.

    Mineral fields such as `CALCITE_MOLES` or `QUARTZ_AREA` are projected onto the
    fixed `MINERAL_VOCAB` dictionary: every vocabulary mineral always gets a
    `<MINERAL>_MOLES` / `<MINERAL>_AREA` slot, set to 0.0 when the run does not
    contain that mineral. Single-mineral and multi-mineral rocks therefore share
    one condition vector. The rock label remains metadata only; it is not a model
    input feature.
    """
    path = Path(path)
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _INPUT_LINE.match(line)
        if match:
            values[match.group("key")] = match.group("value").strip()

    row: dict[str, Any] = {
        "dataset": spec.name,
        "rock": spec.rock,
        "run_id": run_id_from_path(path),
        "input_path": str(path),
    }
    for key, value in values.items():
        row[key] = _coerce_value(value)

    unknown_minerals = sorted(
        {
            match.group(1)
            for key in values
            if (match := _MINERAL_KEY.fullmatch(key)) and match.group(1) not in MINERAL_VOCAB
        }
    )
    if unknown_minerals:
        raise ValueError(f"{path} declares minerals outside MINERAL_VOCAB: {unknown_minerals}")
    for feature in MINERAL_CONDITION_FEATURES:
        raw_value = row.get(feature, 0.0)
        try:
            row[feature] = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} has a non-numeric {feature} value: {raw_value!r}") from exc
    if not any(row[f"{mineral}_MOLES"] > 0.0 for mineral in MINERAL_VOCAB):
        raise ValueError(f"{path} declares no mineral with positive moles")

    missing = [feature for feature in CONDITION_FEATURES if feature not in row]
    if missing:
        raise ValueError(f"{path} is missing condition features: {missing}")
    return row


def load_output_trajectory(path: Path | str) -> pd.DataFrame:
    """Read one `*_Output.txt` PHREEQC trajectory as a wide dataframe.

    Every column present in the file is returned. Callers select the fixed
    `OUTPUT_FEATURES` contract, so dataset-specific extra columns are ignored.
    """
    path = Path(path)
    frame = pd.read_csv(path, sep=r"\s+", engine="python")
    frame.insert(0, "run_id", run_id_from_path(path))
    frame.insert(0, "output_path", str(path))
    return frame


def build_condition_time_tensor(
    conditions_norm: np.ndarray,
    time_axis: np.ndarray,
    *,
    time_mean: float,
    time_std: float,
) -> np.ndarray:
    """Create model input tensor `(runs, timesteps, normalized_conditions + time)`.

    The same normalized condition vector is repeated for each timestep, then a
    normalized time column is appended. No output rows and no rock labels enter X.
    """
    if conditions_norm.ndim != 2:
        raise ValueError(f"conditions_norm must be 2D, got {conditions_norm.shape}")
    if time_axis.ndim != 1:
        raise ValueError(f"time_axis must be 1D, got {time_axis.shape}")
    safe_time_std = time_std if time_std > 0 else 1.0
    time_norm = ((time_axis - time_mean) / safe_time_std).astype(np.float32)
    condition_block = np.repeat(conditions_norm[:, None, :], len(time_axis), axis=1)
    time_block = np.repeat(time_norm[None, :, None], conditions_norm.shape[0], axis=0)
    return np.concatenate([condition_block, time_block], axis=-1).astype(np.float32)


class FullTrajectoryDataset(Dataset):
    def __init__(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        run_ids: Sequence[str],
        rocks: Sequence[str],
    ) -> None:
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError("x and y must be 3D arrays")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of runs")
        if len(run_ids) != x.shape[0] or len(rocks) != x.shape[0]:
            raise ValueError("metadata length must match number of runs")
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.run_ids = list(run_ids)
        self.rocks = list(rocks)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class IndexedTrajectoryDataset(Dataset):
    """Build one condition+time trajectory sample from a disk-backed target file."""

    def __init__(
        self,
        *,
        conditions_norm: np.ndarray,
        time_norm: np.ndarray,
        targets_path: Path | str,
        indices: np.ndarray,
    ) -> None:
        if conditions_norm.ndim != 2:
            raise ValueError(
                f"conditions_norm must be 2D, got {conditions_norm.shape}"
            )
        if conditions_norm.dtype != np.dtype(np.float32):
            raise ValueError("conditions_norm must be float32")
        if conditions_norm.shape[1] == 0:
            raise ValueError("conditions_norm must contain at least one condition feature")
        if not np.isfinite(conditions_norm).all():
            raise ValueError("conditions_norm must be finite")
        if time_norm.ndim != 1:
            raise ValueError(f"time_norm must be 1D, got {time_norm.shape}")
        if time_norm.dtype != np.dtype(np.float32):
            raise ValueError("time_norm must be float32")
        if len(time_norm) == 0:
            raise ValueError("time_norm must contain at least one timestep")
        if not np.isfinite(time_norm).all():
            raise ValueError("time_norm must be finite")

        selected = _validated_dataset_indices(indices, conditions_norm.shape[0])
        self.conditions_norm = np.array(
            conditions_norm,
            dtype=np.float32,
            order="C",
            copy=True,
        )
        self.time_norm = np.array(
            time_norm,
            dtype=np.float32,
            order="C",
            copy=True,
        )
        self.indices = selected
        self.targets_path = Path(targets_path)
        self._expected_target_shape: tuple[int, int, int]
        self._targets: np.memmap | None = None
        self._target_pid: int | None = None

        header_map = _load_target_memmap(self.targets_path)
        try:
            self._expected_target_shape = _validate_target_contract(
                header_map,
                n_runs=self.conditions_norm.shape[0],
                n_timesteps=len(self.time_norm),
            )
        finally:
            header_map._mmap.close()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index,
            (int, np.integer),
        ):
            raise TypeError("dataset index must be an integer")
        if index < 0 or index >= len(self):
            raise IndexError("dataset index is outside the split range")

        global_index = int(self.indices[index])
        targets = self._targets_for_current_process()
        x = np.empty(
            (len(self.time_norm), self.conditions_norm.shape[1] + 1),
            dtype=np.float32,
        )
        x[:, :-1] = self.conditions_norm[global_index]
        x[:, -1] = self.time_norm
        y = np.array(
            targets[global_index],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        return torch.from_numpy(x), torch.from_numpy(y)

    def close(self) -> None:
        """Close this process's target mapping; the next item reopens it lazily."""
        targets = getattr(self, "_targets", None)
        if targets is not None:
            targets._mmap.close()
        self._targets = None
        self._target_pid = None

    def __getstate__(self) -> dict[str, Any]:
        """Do not pickle an open mmap into spawned DataLoader workers."""
        state = self.__dict__.copy()
        state["_targets"] = None
        state["_target_pid"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _targets_for_current_process(self) -> np.memmap:
        current_pid = os.getpid()
        if self._targets is not None and self._target_pid == current_pid:
            return self._targets

        self.close()
        candidate = _load_target_memmap(self.targets_path)
        try:
            shape = _validate_target_contract(
                candidate,
                n_runs=self.conditions_norm.shape[0],
                n_timesteps=len(self.time_norm),
            )
            if shape != self._expected_target_shape:
                raise ValueError("targets shape changed after dataset construction")
        except BaseException:
            candidate._mmap.close()
            raise
        self._targets = candidate
        self._target_pid = current_pid
        return candidate


def _load_target_memmap(path: Path) -> np.memmap:
    targets = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(targets, np.memmap):
        raise ValueError(f"targets must be a memory-mappable NPY file: {path}")
    return targets


def _validate_target_contract(
    targets: np.memmap,
    *,
    n_runs: int,
    n_timesteps: int,
) -> tuple[int, int, int]:
    if targets.ndim != 3:
        raise ValueError(f"targets must be 3D, got {targets.shape}")
    if targets.dtype != np.dtype(np.float32):
        raise ValueError("targets must be float32")
    if targets.shape[0] != n_runs:
        raise ValueError("target run count must match conditions_norm")
    if targets.shape[1] != n_timesteps:
        raise ValueError("target timestep count must match time_norm")
    if targets.shape[2] <= 0:
        raise ValueError("targets must contain at least one output feature")
    return tuple(int(size) for size in targets.shape)


def _validated_dataset_indices(indices: np.ndarray, n_runs: int) -> np.ndarray:
    selected = np.asarray(indices)
    if selected.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if np.issubdtype(selected.dtype, np.bool_) or not np.issubdtype(
        selected.dtype,
        np.integer,
    ):
        raise ValueError("indices must contain integer values")
    if len(selected) == 0:
        raise ValueError("indices must be non-empty")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("indices must be unique")
    if selected.min() < 0 or selected.max() >= n_runs:
        raise ValueError("indices contain values outside the valid run range")
    return selected.astype(np.int64, copy=True)


def build_bundle(
    specs: Sequence[DatasetSpec],
    *,
    keep_outputs_frame: bool = False,
) -> tuple[TrajectoryBundle, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse all configured datasets and keep only matched successful runs.

    The returned inventory reports missing input/output runs, but tensors are
    built only from complete runs with valid input and output files.
    """
    input_rows: list[dict[str, Any]] = []
    output_frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, Any]] = []
    trajectory_arrays: list[np.ndarray] = []
    condition_arrays: list[np.ndarray] = []
    run_ids: list[str] = []
    rocks: list[str] = []
    time_axis: np.ndarray | None = None

    for spec in specs:
        root = Path(spec.path)
        input_dir = root / "input"
        output_dir = root / "output"
        input_paths = {run_id_from_path(path): path for path in sorted(input_dir.glob("*_Input.txt"))}
        output_paths = {run_id_from_path(path): path for path in sorted(output_dir.glob("*_Output.txt"))}
        all_run_ids = sorted(set(input_paths) | set(output_paths), key=_numeric_sort_key)
        if spec.max_runs is not None:
            all_run_ids = all_run_ids[: spec.max_runs]

        for run_id in all_run_ids:
            status = "matched"
            if run_id not in input_paths:
                status = "missing_input"
            elif run_id not in output_paths:
                status = "missing_output"

            inventory_rows.append(
                {
                    "dataset": spec.name,
                    "rock": spec.rock,
                    "run_id": run_id,
                    "status": status,
                    "input_path": str(input_paths.get(run_id, "")),
                    "output_path": str(output_paths.get(run_id, "")),
                }
            )
            if status != "matched":
                continue

            input_row = load_input_parameters(input_paths[run_id], spec)
            output_frame = load_output_trajectory(output_paths[run_id])
            missing_outputs = [name for name in OUTPUT_FEATURES if name not in output_frame.columns]
            if missing_outputs:
                raise ValueError(f"{output_paths[run_id]} missing outputs: {missing_outputs}")
            run_time = output_frame["time_d"].to_numpy(dtype=np.float64)
            if time_axis is None:
                time_axis = run_time
            elif not np.allclose(time_axis, run_time):
                raise ValueError(f"{output_paths[run_id]} has a different time axis")

            input_rows.append(input_row)
            output_frame.insert(0, "rock", spec.rock)
            output_frame.insert(0, "dataset", spec.name)
            if keep_outputs_frame:
                output_frames.append(output_frame)
            condition_arrays.append(np.array([input_row[name] for name in CONDITION_FEATURES], dtype=np.float64))
            trajectory_arrays.append(output_frame[list(OUTPUT_FEATURES)].to_numpy(dtype=np.float64))
            run_ids.append(f"{spec.name}:{run_id}")
            rocks.append(spec.rock)

    if not trajectory_arrays or time_axis is None:
        raise ValueError("No matched successful runs found")

    bundle = TrajectoryBundle(
        conditions=np.stack(condition_arrays, axis=0),
        trajectories=np.stack(trajectory_arrays, axis=0),
        time_axis=time_axis,
        run_ids=run_ids,
        rocks=np.array(rocks, dtype=object),
        condition_features=CONDITION_FEATURES,
        output_features=OUTPUT_FEATURES,
    )
    return (
        bundle,
        pd.DataFrame(input_rows),
        pd.concat(output_frames, ignore_index=True) if output_frames else pd.DataFrame(),
        pd.DataFrame(inventory_rows),
    )


def write_processed_bundle(
    bundle: TrajectoryBundle,
    inputs: pd.DataFrame,
    outputs: pd.DataFrame,
    inventory: pd.DataFrame,
    processed_dir: Path | str,
) -> Path:
    """Write inspection CSVs plus a fast NPZ cache for training."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    inputs.to_csv(processed_dir / "inputs.csv", index=False)
    inventory.to_csv(processed_dir / "run_inventory.csv", index=False)
    outputs_path = processed_dir / "outputs.csv"
    if not outputs.empty:
        outputs.to_csv(outputs_path, index=False)
    else:
        outputs_path.unlink(missing_ok=True)
    np.savez_compressed(
        processed_dir / "bundle.npz",
        conditions=bundle.conditions,
        trajectories=bundle.trajectories,
        time_axis=bundle.time_axis,
        run_ids=np.asarray(bundle.run_ids, dtype=np.str_),
        rocks=np.asarray(bundle.rocks, dtype=np.str_),
        condition_features=np.asarray(bundle.condition_features, dtype=np.str_),
        output_features=np.asarray(bundle.output_features, dtype=np.str_),
    )
    (processed_dir / "schema.json").write_text(
        json.dumps(
            {
                "condition_features": list(bundle.condition_features),
                "output_features": list(bundle.output_features),
                "n_runs": int(bundle.conditions.shape[0]),
                "n_timesteps": int(bundle.trajectories.shape[1]),
                "n_condition_features": int(bundle.conditions.shape[1]),
                "n_output_features": int(bundle.trajectories.shape[2]),
            },
            indent=2,
        )
    )
    return processed_dir / "bundle.npz"


def load_cached_bundle(cache_path: Path | str) -> TrajectoryBundle:
    """Load the fast training arrays produced by `write_processed_bundle`."""
    cache_path = Path(cache_path)
    required_keys = {
        "conditions",
        "trajectories",
        "time_axis",
        "run_ids",
        "rocks",
        "condition_features",
        "output_features",
    }
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            missing = sorted(required_keys - set(payload.files))
            if missing:
                raise ValueError(f"Data cache is missing required arrays: {missing}")
            conditions = payload["conditions"]
            trajectories = payload["trajectories"]
            time_axis = payload["time_axis"]
            run_ids_array = payload["run_ids"]
            rocks_array = payload["rocks"]
            condition_features_array = payload["condition_features"]
            output_features_array = payload["output_features"]
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise ValueError(
                f"Data cache must use pickle-free Unicode metadata: {cache_path}"
            ) from exc
        raise

    metadata_arrays = (
        run_ids_array,
        rocks_array,
        condition_features_array,
        output_features_array,
    )
    if any(array.ndim != 1 for array in metadata_arrays):
        raise ValueError("Cached string arrays must use one-dimensional metadata")
    run_ids = run_ids_array.astype(str).tolist()
    rocks = rocks_array.astype(str)
    condition_features = tuple(condition_features_array.astype(str).tolist())
    output_features = tuple(output_features_array.astype(str).tolist())

    if conditions.ndim != 2 or trajectories.ndim != 3 or time_axis.ndim != 1:
        raise ValueError("Cached conditions/time/trajectories have invalid dimensions")
    n_runs = conditions.shape[0]
    if n_runs == 0 or trajectories.shape[1] == 0:
        raise ValueError("Cached data arrays must not be empty")
    if trajectories.shape[0] != n_runs or len(run_ids) != n_runs or len(rocks) != n_runs:
        raise ValueError("Cached run arrays and metadata have inconsistent lengths")
    if trajectories.shape[1] != len(time_axis):
        raise ValueError("Cached trajectory length does not match the time axis")
    if conditions.shape[1] != len(condition_features):
        raise ValueError("Cached condition feature count does not match conditions")
    if trajectories.shape[2] != len(output_features):
        raise ValueError("Cached output feature count does not match trajectories")
    for name, array in (
        ("conditions", conditions),
        ("trajectories", trajectories),
        ("time_axis", time_axis),
    ):
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype, np.complexfloating
        ):
            raise ValueError(f"Cached {name} must contain real numeric values")
        if not _all_finite(array):
            raise ValueError(f"Cached {name} must contain only finite values")
    if not np.all(np.diff(time_axis) > 0):
        raise ValueError("Cached time axis must be strictly increasing")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Cached run ids must be unique")
    if condition_features != CONDITION_FEATURES:
        raise ValueError("Cached condition feature order does not match the model contract")
    if output_features != OUTPUT_FEATURES:
        raise ValueError("Cached output feature order does not match the model contract")

    return TrajectoryBundle(
        conditions=conditions,
        trajectories=trajectories,
        time_axis=time_axis,
        run_ids=run_ids,
        rocks=rocks,
        condition_features=condition_features,
        output_features=output_features,
    )


def _all_finite(array: np.ndarray, *, chunk_size: int = 256) -> bool:
    """Check large arrays without allocating one full-size boolean array."""
    for start in range(0, array.shape[0], chunk_size):
        if not np.isfinite(array[start : start + chunk_size]).all():
            return False
    return True


def _coerce_value(value: str) -> str | float:
    """Convert numeric strings to floats while preserving non-numeric metadata."""
    try:
        return float(value)
    except ValueError:
        return value


def _numeric_sort_key(run_id: str) -> tuple[int, str]:
    """Sort numeric run ids naturally instead of lexicographically."""
    return (int(run_id), run_id) if run_id.isdigit() else (10**12, run_id)
