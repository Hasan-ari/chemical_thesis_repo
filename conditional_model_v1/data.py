from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CONDITION_FEATURES: tuple[str, ...] = (
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
    "mineral_moles",
    "mineral_area",
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


def run_id_from_path(path: Path) -> str:
    """Extract numeric run id from names like `1_Input.txt` or `1_Output.txt`."""
    stem = path.stem
    return stem.removesuffix("_Input").removesuffix("_Output")


def load_input_parameters(path: Path | str, spec: DatasetSpec) -> dict[str, Any]:
    """Parse one professor-generated `*_Input.txt` file into generic numeric conditions.

    Rock-specific mineral fields such as `CALCITE_MOLES` and `DOLOMITE_MOLES`
    are mapped into the shared `mineral_moles` / `mineral_area` names. The rock
    label remains metadata only; it is not a model input feature.
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

    mineral_moles_key = f"{spec.rock.upper()}_MOLES"
    mineral_area_key = f"{spec.rock.upper()}_AREA"
    if mineral_moles_key not in row or mineral_area_key not in row:
        raise ValueError(
            f"{path} is missing {mineral_moles_key}/{mineral_area_key} for {spec.rock}"
        )
    row["mineral_moles"] = float(row[mineral_moles_key])
    row["mineral_area"] = float(row[mineral_area_key])

    missing = [feature for feature in CONDITION_FEATURES if feature not in row]
    if missing:
        raise ValueError(f"{path} is missing condition features: {missing}")
    return row


def load_output_trajectory(path: Path | str) -> pd.DataFrame:
    """Read one `*_Output.txt` PHREEQC trajectory as a wide dataframe."""
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
    outputs.to_csv(processed_dir / "outputs.csv", index=False)
    inventory.to_csv(processed_dir / "run_inventory.csv", index=False)
    if not outputs.empty:
        outputs.to_csv(processed_dir / "outputs.csv", index=False)
    np.savez_compressed(
        processed_dir / "bundle.npz",
        conditions=bundle.conditions,
        trajectories=bundle.trajectories,
        time_axis=bundle.time_axis,
        run_ids=np.array(bundle.run_ids, dtype=object),
        rocks=bundle.rocks,
        condition_features=np.array(bundle.condition_features, dtype=object),
        output_features=np.array(bundle.output_features, dtype=object),
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
    payload = np.load(cache_path, allow_pickle=True)
    return TrajectoryBundle(
        conditions=payload["conditions"],
        trajectories=payload["trajectories"],
        time_axis=payload["time_axis"],
        run_ids=payload["run_ids"].astype(str).tolist(),
        rocks=payload["rocks"],
        condition_features=tuple(payload["condition_features"].astype(str).tolist()),
        output_features=tuple(payload["output_features"].astype(str).tolist()),
    )


def _coerce_value(value: str) -> str | float:
    """Convert numeric strings to floats while preserving non-numeric metadata."""
    try:
        return float(value)
    except ValueError:
        return value


def _numeric_sort_key(run_id: str) -> tuple[int, str]:
    """Sort numeric run ids naturally instead of lexicographically."""
    return (int(run_id), run_id) if run_id.isdigit() else (10**12, run_id)
