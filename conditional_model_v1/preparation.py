from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TextIO

import numpy as np
import pandas as pd

from conditional_model_v1.units import SUPPORTED_INPUT_UNITS, conversion_table
from conditional_model_v1.data import (
    CONDITION_FEATURES,
    OUTPUT_FEATURES,
    DatasetSpec,
    load_input_parameters,
    load_output_trajectory,
    run_id_from_path,
)

# v2: 23 scalar conditions (8-rock union), 26 shared outputs, per-dataset input_units.
CACHE_SCHEMA_VERSION = 2
_STATUSES = ("matched", "failed", "missing_input", "missing_output")


@dataclass(frozen=True)
class _RunRecord:
    spec: DatasetSpec
    run_id: str
    status: str
    input_path: Path | None
    output_path: Path | None


@dataclass(frozen=True)
class _Discovery:
    records: tuple[_RunRecord, ...]
    counts_by_dataset: dict[str, dict[str, int]]
    source_inventory_signature: str


def prepare_cache(
    specs: Sequence[DatasetSpec],
    cache_dir: Path | str,
    *,
    progress_every: int = 250,
) -> Path:
    """Parse raw TXT once and atomically publish a reusable training cache.

    Numeric arrays are staged in disk-backed memmaps, so preparation holds only
    one trajectory dataframe in memory at a time. Output inspection CSVs are
    sharded by dataset and written incrementally through persistent gzip handles.
    Existing caches are immutable: a valid cache is reused; an invalid one must
    be given a new versioned directory name.
    """
    normalized_specs = tuple(specs)
    _validate_specs(normalized_specs)
    cache_dir = Path(cache_dir)

    if cache_dir.exists():
        validate_cache_manifest(cache_dir, normalized_specs)
        return cache_dir / "bundle.npz"

    discovery = _discover_runs(normalized_specs)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{cache_dir.name}.building-",
            dir=cache_dir.parent,
        )
    )
    try:
        _write_staged_cache(
            specs=normalized_specs,
            discovery=discovery,
            staging_dir=staging_dir,
            progress_every=progress_every,
        )
        (staging_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
        _publish_cache(staging_dir, cache_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return cache_dir / "bundle.npz"


def validate_cache_manifest(
    cache_dir: Path | str,
    specs: Sequence[DatasetSpec],
    *,
    verify_artifact_hashes: bool = False,
) -> dict[str, Any]:
    """Validate a cache without consulting raw TXT paths.

    This is intentionally cheap enough to run after copying a cache from Drive.
    The NPZ's numeric contents are validated separately by ``load_cached_bundle``.
    """
    normalized_specs = tuple(specs)
    _validate_specs(normalized_specs)
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise ValueError(f"Cache directory does not exist: {cache_dir}")
    if not (cache_dir / "_SUCCESS").is_file():
        raise ValueError(f"Cache is incomplete because _SUCCESS is missing: {cache_dir}")

    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Cache manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cache manifest is not readable JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Cache manifest must contain a JSON object: {manifest_path}")
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            "Cache schema version does not match this code: "
            f"expected {CACHE_SCHEMA_VERSION}, got {manifest.get('schema_version')}"
        )

    actual_datasets = manifest.get("datasets")
    if not isinstance(actual_datasets, list):
        raise ValueError("Cache manifest datasets must be a list")
    actual_contract = [
        {
            "name": item.get("name"),
            "rock": item.get("rock"),
            "max_runs": item.get("max_runs"),
            "input_units": item.get("input_units"),
        }
        for item in actual_datasets
        if isinstance(item, dict)
    ]
    if actual_contract != _dataset_contract(normalized_specs):
        raise ValueError("Cache dataset contract does not match the configured datasets")
    if manifest.get("condition_features") != list(CONDITION_FEATURES):
        raise ValueError("Cache manifest condition feature order does not match the model contract")
    if manifest.get("output_features") != list(OUTPUT_FEATURES):
        raise ValueError("Cache manifest output feature order does not match the model contract")

    n_runs = manifest.get("n_runs")
    n_timesteps = manifest.get("n_timesteps")
    if not isinstance(n_runs, int) or n_runs <= 0:
        raise ValueError("Cache manifest n_runs must be a positive integer")
    if not isinstance(n_timesteps, int) or n_timesteps <= 0:
        raise ValueError("Cache manifest n_timesteps must be a positive integer")
    expected_arrays = {
        "conditions": ([n_runs, len(CONDITION_FEATURES)], "float64"),
        "trajectories": ([n_runs, n_timesteps, len(OUTPUT_FEATURES)], "float64"),
        "time_axis": ([n_timesteps], "float64"),
    }
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("Cache manifest arrays must be an object")
    for array_name, (shape, dtype) in expected_arrays.items():
        metadata = arrays.get(array_name)
        if not isinstance(metadata, dict):
            raise ValueError(f"Cache manifest is missing array metadata: {array_name}")
        if metadata.get("shape") != shape or metadata.get("dtype") != dtype:
            raise ValueError(f"Cache manifest array contract is invalid: {array_name}")

    signature = manifest.get("source_inventory_signature")
    if not isinstance(signature, str) or len(signature) != 64:
        raise ValueError(
            "Cache manifest source_inventory_signature must be a SHA-256 hex digest"
        )
    try:
        int(signature, 16)
    except ValueError as exc:
        raise ValueError(
            "Cache manifest source_inventory_signature must be a SHA-256 hex digest"
        ) from exc

    expected_artifacts = {
        "bundle.npz",
        "inputs.csv",
        "run_inventory.csv",
        *(f"outputs/{spec.name}.csv.gz" for spec in normalized_specs),
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("Cache manifest artifact inventory is incomplete or unexpected")
    for relative_path, metadata in artifacts.items():
        if not _is_safe_relative_path(relative_path) or not isinstance(metadata, dict):
            raise ValueError(f"Cache manifest contains an invalid artifact path: {relative_path}")
        artifact_path = cache_dir / relative_path
        if not artifact_path.is_file():
            raise ValueError(f"Cache artifact is missing: {artifact_path}")
        expected_size = metadata.get("size_bytes")
        if not isinstance(expected_size, int) or artifact_path.stat().st_size != expected_size:
            raise ValueError(f"Cache artifact size does not match the manifest: {artifact_path}")
        expected_hash = metadata.get("sha256")
        if not _is_sha256(expected_hash):
            raise ValueError(f"Cache artifact SHA-256 is missing or invalid: {artifact_path}")
        if verify_artifact_hashes and _file_sha256(artifact_path) != expected_hash:
            raise ValueError(f"Cache artifact SHA-256 does not match: {artifact_path}")

    matched_total = 0
    for dataset in actual_datasets:
        counts = dataset.get("counts") if isinstance(dataset, dict) else None
        if not isinstance(counts, dict) or any(
            not isinstance(counts.get(status), int) or counts[status] < 0
            for status in _STATUSES
        ):
            raise ValueError("Cache manifest dataset counts are invalid")
        matched_total += counts["matched"]
    if matched_total != n_runs:
        raise ValueError("Cache manifest matched counts do not equal n_runs")
    return manifest


def _discover_runs(specs: tuple[DatasetSpec, ...]) -> _Discovery:
    records: list[_RunRecord] = []
    counts_by_dataset: dict[str, dict[str, int]] = {}

    for spec in specs:
        root = Path(spec.path)
        input_dir = root / "input"
        output_dir = root / "output"
        failed_dir = root / "failed"
        if not input_dir.is_dir() or not output_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset {spec.name} must contain input/ and output/ directories: {root}"
            )

        input_paths = _run_path_map(input_dir, "*_Input.txt")
        output_paths = _run_path_map(output_dir, "*_Output.txt")
        failed_paths = (
            _run_path_map(failed_dir, "*_Input.txt") if failed_dir.is_dir() else {}
        )
        run_ids = sorted(
            set(input_paths) | set(output_paths) | set(failed_paths),
            key=_run_sort_key,
        )
        if spec.max_runs is not None:
            run_ids = run_ids[: spec.max_runs]

        counts = {status: 0 for status in _STATUSES}
        for run_id in run_ids:
            input_path = input_paths.get(run_id)
            output_path = output_paths.get(run_id)
            failed_path = failed_paths.get(run_id)
            if failed_path is not None and (input_path is not None or output_path is not None):
                raise ValueError(
                    f"Dataset {spec.name} run {run_id} appears in failed/ and input/output"
                )
            if failed_path is not None:
                status = "failed"
                input_path = failed_path
            elif input_path is None:
                status = "missing_input"
            elif output_path is None:
                status = "missing_output"
            else:
                status = "matched"
            counts[status] += 1
            records.append(
                _RunRecord(
                    spec=spec,
                    run_id=run_id,
                    status=status,
                    input_path=input_path,
                    output_path=output_path,
                )
            )
        if counts["matched"] == 0:
            raise ValueError(f"Dataset {spec.name} has zero matched successful runs")
        counts_by_dataset[spec.name] = counts

    return _Discovery(
        records=tuple(records),
        counts_by_dataset=counts_by_dataset,
        source_inventory_signature=_source_inventory_signature(specs, records),
    )


def _write_staged_cache(
    *,
    specs: tuple[DatasetSpec, ...],
    discovery: _Discovery,
    staging_dir: Path,
    progress_every: int,
) -> None:
    matched_records = tuple(
        record for record in discovery.records if record.status == "matched"
    )
    first_conditions, first_trajectory, time_axis = _parse_matched_run(
        matched_records[0],
        reference_time=None,
    )
    n_runs = len(matched_records)
    n_timesteps = len(time_axis)
    conditions_path = staging_dir / ".conditions.npy"
    trajectories_path = staging_dir / ".trajectories.npy"
    inputs_path = staging_dir / "inputs.csv"
    inventory_path = staging_dir / "run_inventory.csv"
    outputs_dir = staging_dir / "outputs"
    outputs_dir.mkdir()
    input_fields = ["dataset", "rock", "run_id", *CONDITION_FEATURES]
    inventory_fields = [
        "dataset",
        "rock",
        "run_id",
        "status",
        "input_path",
        "output_path",
    ]
    run_ids: list[str] = []
    rocks: list[str] = []
    output_headers_written = {spec.name: False for spec in specs}
    conditions: np.memmap | None = None
    trajectories: np.memmap | None = None
    primary_error: BaseException | None = None

    try:
        conditions = np.lib.format.open_memmap(
            conditions_path,
            mode="w+",
            dtype=np.float64,
            shape=(n_runs, len(CONDITION_FEATURES)),
        )
        trajectories = np.lib.format.open_memmap(
            trajectories_path,
            mode="w+",
            dtype=np.float64,
            shape=(n_runs, n_timesteps, len(OUTPUT_FEATURES)),
        )
        with (
            inputs_path.open("w", encoding="utf-8", newline="") as inputs_file,
            inventory_path.open("w", encoding="utf-8", newline="") as inventory_file,
            ExitStack() as output_stack,
        ):
            input_writer = csv.DictWriter(inputs_file, fieldnames=input_fields)
            inventory_writer = csv.DictWriter(inventory_file, fieldnames=inventory_fields)
            input_writer.writeheader()
            inventory_writer.writeheader()
            output_files: dict[str, TextIO] = {
                spec.name: output_stack.enter_context(
                    gzip.open(
                        outputs_dir / f"{spec.name}.csv.gz",
                        mode="wt",
                        encoding="utf-8",
                        newline="",
                    )
                )
                for spec in specs
            }

            for record in discovery.records:
                inventory_writer.writerow(
                    {
                        "dataset": record.spec.name,
                        "rock": record.spec.rock,
                        "run_id": record.run_id,
                        "status": record.status,
                        "input_path": _relative_source_path(record, record.input_path),
                        "output_path": _relative_source_path(record, record.output_path),
                    }
                )

            for index, record in enumerate(matched_records):
                if index == 0:
                    condition_row = first_conditions
                    trajectory = first_trajectory
                else:
                    condition_row, trajectory, _ = _parse_matched_run(
                        record,
                        reference_time=time_axis,
                    )
                conditions[index] = condition_row
                trajectories[index] = trajectory
                run_ids.append(f"{record.spec.name}:{record.run_id}")
                rocks.append(record.spec.rock)
                input_writer.writerow(
                    {
                        "dataset": record.spec.name,
                        "rock": record.spec.rock,
                        "run_id": record.run_id,
                        **dict(zip(CONDITION_FEATURES, condition_row, strict=True)),
                    }
                )
                _write_output_chunk(
                    output_files[record.spec.name],
                    record=record,
                    time_axis=time_axis,
                    trajectory=trajectory,
                    write_header=not output_headers_written[record.spec.name],
                )
                output_headers_written[record.spec.name] = True
                if progress_every > 0 and (index + 1) % progress_every == 0:
                    print(f"prepared_runs={index + 1}/{n_runs}")

        conditions.flush()
        trajectories.flush()
        np.savez_compressed(
            staging_dir / "bundle.npz",
            conditions=conditions,
            trajectories=trajectories,
            time_axis=time_axis,
            run_ids=np.asarray(run_ids, dtype=np.str_),
            rocks=np.asarray(rocks, dtype=np.str_),
            condition_features=np.asarray(CONDITION_FEATURES, dtype=np.str_),
            output_features=np.asarray(OUTPUT_FEATURES, dtype=np.str_),
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_errors: list[OSError] = []
        for array in (conditions, trajectories):
            if array is None:
                continue
            try:
                array._mmap.close()
            except OSError as exc:
                cleanup_errors.append(exc)
        conditions = None
        trajectories = None
        for temporary_path in (conditions_path, trajectories_path):
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_errors.append(exc)
        if cleanup_errors and primary_error is None:
            raise cleanup_errors[0]

    artifact_paths = [
        staging_dir / "bundle.npz",
        inputs_path,
        inventory_path,
        *(outputs_dir / f"{spec.name}.csv.gz" for spec in specs),
    ]
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_inventory_signature_kind": (
            "sha256(dataset_contract + relative_path + size_bytes); not a content hash"
        ),
        "source_inventory_signature": discovery.source_inventory_signature,
        "datasets": [
            {
                "name": spec.name,
                "rock": spec.rock,
                "max_runs": spec.max_runs,
                "input_units": spec.input_units,
                "counts": discovery.counts_by_dataset[spec.name],
                "output_csv": f"outputs/{spec.name}.csv.gz",
            }
            for spec in specs
        ],
        "n_runs": n_runs,
        "n_timesteps": n_timesteps,
        "condition_features": list(CONDITION_FEATURES),
        "output_features": list(OUTPUT_FEATURES),
        "solution_unit_conversion": conversion_table(),
        "arrays": {
            "conditions": {
                "shape": [n_runs, len(CONDITION_FEATURES)],
                "dtype": "float64",
            },
            "trajectories": {
                "shape": [n_runs, n_timesteps, len(OUTPUT_FEATURES)],
                "dtype": "float64",
            },
            "time_axis": {"shape": [n_timesteps], "dtype": "float64"},
        },
        "artifacts": {
            path.relative_to(staging_dir).as_posix(): {
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in artifact_paths
        },
    }
    (staging_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_matched_run(
    record: _RunRecord,
    *,
    reference_time: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if record.input_path is None or record.output_path is None:
        raise AssertionError("Matched run must have input and output paths")
    input_row = load_input_parameters(record.input_path, record.spec)
    try:
        conditions = np.asarray(
            [input_row[name] for name in CONDITION_FEATURES],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{record.input_path} contains non-numeric condition values"
        ) from exc
    if not np.isfinite(conditions).all():
        raise ValueError(f"{record.input_path} must contain only finite condition values")

    output_frame = load_output_trajectory(record.output_path)
    data_columns = [
        column for column in output_frame.columns if column not in {"output_path", "run_id"}
    ]
    # Multi-mineral datasets (e.g. Sandstone) report extra per-mineral columns such
    # as `Barite`/`Calcite`, and the legacy rocks report totals the schists lack.
    # Only the shared OUTPUT_FEATURES plus `time_d` are selected;
    # unknown extra columns are ignored, while missing expected columns still raise.
    missing = [column for column in ("time_d", *OUTPUT_FEATURES) if column not in data_columns]
    if missing:
        raise ValueError(f"{record.output_path} missing outputs: {missing}")
    try:
        time_axis = output_frame["time_d"].to_numpy(dtype=np.float64)
        trajectory = output_frame.loc[:, list(OUTPUT_FEATURES)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{record.output_path} contains non-numeric output values") from exc
    if len(time_axis) == 0:
        raise ValueError(f"{record.output_path} contains no timesteps")
    if not np.isfinite(time_axis).all() or not np.isfinite(trajectory).all():
        raise ValueError(f"{record.output_path} must contain only finite output values")
    if not np.all(np.diff(time_axis) > 0):
        raise ValueError(f"{record.output_path} time axis must be strictly increasing")
    if reference_time is not None and (
        time_axis.shape != reference_time.shape
        or not np.allclose(time_axis, reference_time, rtol=0.0, atol=1e-12)
    ):
        raise ValueError(f"{record.output_path} has a different time axis")
    return conditions, trajectory, time_axis


def _write_output_chunk(
    file_obj: TextIO,
    *,
    record: _RunRecord,
    time_axis: np.ndarray,
    trajectory: np.ndarray,
    write_header: bool,
) -> None:
    frame = pd.DataFrame(trajectory, columns=OUTPUT_FEATURES)
    frame.insert(0, "time_d", time_axis)
    frame.insert(0, "timestep_index", np.arange(len(time_axis), dtype=np.int64))
    frame.insert(0, "run_id", record.run_id)
    frame.insert(0, "rock", record.spec.rock)
    frame.insert(0, "dataset", record.spec.name)
    frame.to_csv(file_obj, index=False, header=write_header)


def _source_inventory_signature(
    specs: tuple[DatasetSpec, ...],
    records: Sequence[_RunRecord],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(
            _dataset_contract(specs),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for record in records:
        root = Path(record.spec.path)
        for role, path in (("input", record.input_path), ("output", record.output_path)):
            if path is None:
                continue
            relative_path = path.relative_to(root).as_posix()
            hasher.update(
                f"\n{record.spec.name}\0{record.run_id}\0{role}\0{relative_path}\0{path.stat().st_size}".encode(
                    "utf-8"
                )
            )
    return hasher.hexdigest()


def _validate_specs(specs: tuple[DatasetSpec, ...]) -> None:
    if not specs:
        raise ValueError("At least one dataset must be configured")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("Configured dataset names must be unique")
    resolved_paths = [
        os.path.normcase(str(Path(spec.path).expanduser().resolve(strict=False)))
        for spec in specs
    ]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Configured dataset paths must be unique after resolution")
    for spec in specs:
        if spec.input_units not in SUPPORTED_INPUT_UNITS:
            raise ValueError(
                f"Dataset {spec.name} has unsupported input_units {spec.input_units!r}"
            )
        if not _is_safe_file_name(spec.name):
            raise ValueError(f"Dataset name must be a safe file name: {spec.name!r}")
        if not spec.rock or spec.rock != spec.rock.strip():
            raise ValueError(f"Dataset rock must be non-empty: {spec.name}")
        if spec.max_runs is not None and spec.max_runs <= 0:
            raise ValueError(f"Dataset max_runs must be positive: {spec.name}")


def _dataset_contract(specs: Sequence[DatasetSpec]) -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "rock": spec.rock,
            "max_runs": spec.max_runs,
            "input_units": spec.input_units,
        }
        for spec in specs
    ]


def _run_path_map(directory: Path, pattern: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in directory.glob(pattern):
        run_id = run_id_from_path(path)
        if run_id in paths:
            raise ValueError(f"Duplicate run id {run_id!r} in {directory}")
        paths[run_id] = path
    return paths


def _run_sort_key(run_id: str) -> tuple[int, int | str]:
    return (0, int(run_id)) if run_id.isdigit() else (1, run_id)


def _is_safe_file_name(value: str) -> bool:
    return (
        bool(value)
        and value == value.strip()
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and "\x00" not in value
    )


def _is_safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _relative_source_path(record: _RunRecord, path: Path | None) -> str:
    if path is None:
        return ""
    return path.relative_to(Path(record.spec.path)).as_posix()


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _publish_cache(staging_dir: Path, cache_dir: Path) -> None:
    """Publish a complete same-filesystem staging directory without overwrite."""
    if cache_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache: {cache_dir}")
    staging_dir.replace(cache_dir)
