from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from conditional_model_v1.data import (
    CONDITION_FEATURES,
    OUTPUT_FEATURES,
    TrajectoryBundle,
)

RUNTIME_DISK_HEADROOM = 1.10
ZIP_COPY_BUFFER_BYTES = 1024 * 1024
FINITE_CHECK_RUN_CHUNK = 256

_ARRAY_MEMBERS: tuple[str, ...] = (
    "conditions.npy",
    "trajectories.npy",
    "time_axis.npy",
    "run_ids.npy",
    "rocks.npy",
    "condition_features.npy",
    "output_features.npy",
)


def estimate_runtime_bytes(bundle_path: Path | str) -> int:
    """Return uncompressed NPZ bytes plus 10% local-disk headroom."""
    with zipfile.ZipFile(bundle_path) as archive:
        members = _validated_member_info(archive)
    return math.ceil(
        sum(member.file_size for member in members.values()) * RUNTIME_DISK_HEADROOM
    )


def materialize_mmap_bundle(
    *,
    cache_dir: Path | str,
    manifest: dict[str, Any],
    runtime_root: Path | str,
) -> tuple[TrajectoryBundle, Path]:
    """Stream NPZ members to local disk and open numeric arrays read-only.

    Compressed NPZ arrays cannot be memory-mapped directly. This function
    extracts only the seven known ``.npy`` members into a SHA-versioned local
    runtime directory. The immutable Drive/cache artifact is never modified.
    """
    cache_dir = Path(cache_dir)
    bundle_path = cache_dir / "bundle.npz"
    source_sha256 = _source_bundle_sha256(manifest)
    source_identity = _file_identity(bundle_path)
    runtime_root = Path(runtime_root)
    runtime_dir = runtime_root / f"{cache_dir.name}-{source_sha256[:16]}"

    if runtime_dir.exists():
        return (
            _load_complete_runtime(
                runtime_dir,
                manifest,
                source_sha256,
                bundle_path,
                source_identity,
            ),
            runtime_dir,
        )

    actual_sha256 = _file_sha256(bundle_path)
    if actual_sha256 != source_sha256:
        raise ValueError(
            "bundle.npz SHA-256 does not match the cache manifest: "
            f"{actual_sha256} != {source_sha256}"
        )
    if _file_identity(bundle_path) != source_identity:
        raise RuntimeError("bundle.npz changed while its SHA-256 was being verified")

    with zipfile.ZipFile(bundle_path) as archive:
        members = _validated_member_info(archive)
        required_bytes = math.ceil(
            sum(member.file_size for member in members.values())
            * RUNTIME_DISK_HEADROOM
        )
        runtime_root.mkdir(parents=True, exist_ok=True)
        available_bytes = shutil.disk_usage(runtime_root).free
        if available_bytes < required_bytes:
            raise OSError(
                "Insufficient runtime disk space: "
                f"required={required_bytes} available={available_bytes}"
            )

        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{runtime_dir.name}.building-",
                dir=runtime_root,
            )
        )
        try:
            for member_name in _ARRAY_MEMBERS:
                _copy_zip_member(
                    archive=archive,
                    member_name=member_name,
                    destination=staging_dir / member_name,
                )
            staged_bundle = _validate_and_open_bundle(staging_dir, manifest)
            _close_bundle_memmaps(staged_bundle)
            if _file_identity(bundle_path) != source_identity:
                raise RuntimeError("bundle.npz changed while runtime data was being built")
            (staging_dir / "runtime_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_bundle_sha256": source_sha256,
                        "source_file_identity": source_identity,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (staging_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
            if runtime_dir.exists():
                raise FileExistsError(f"Refusing to overwrite runtime data: {runtime_dir}")
            staging_dir.replace(runtime_dir)
        except BaseException:
            _remove_staging(staging_dir)
            raise

    return (
        _load_complete_runtime(
            runtime_dir,
            manifest,
            source_sha256,
            bundle_path,
            source_identity,
        ),
        runtime_dir,
    )


def _load_complete_runtime(
    runtime_dir: Path,
    manifest: dict[str, Any],
    source_sha256: str,
    bundle_path: Path,
    source_identity: dict[str, int],
) -> TrajectoryBundle:
    required_files = {*_ARRAY_MEMBERS, "runtime_manifest.json", "_SUCCESS"}
    existing_files = {path.name for path in runtime_dir.iterdir() if path.is_file()}
    if existing_files != required_files:
        raise ValueError(f"runtime is incomplete: {runtime_dir}")
    try:
        runtime_manifest = json.loads(
            (runtime_dir / "runtime_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"runtime manifest is unreadable: {runtime_dir}") from exc
    if (
        runtime_manifest.get("schema_version") != 1
        or runtime_manifest.get("source_bundle_sha256") != source_sha256
        or not isinstance(runtime_manifest.get("source_file_identity"), dict)
    ):
        raise ValueError(f"runtime manifest does not match the source bundle: {runtime_dir}")
    if runtime_manifest["source_file_identity"] != source_identity:
        if _file_sha256(bundle_path) != source_sha256:
            raise ValueError("bundle.npz SHA-256 does not match the cache manifest")
    return _validate_and_open_bundle(
        runtime_dir,
        manifest,
        validate_trajectory_finite=False,
    )


def _validate_and_open_bundle(
    directory: Path,
    manifest: dict[str, Any],
    *,
    validate_trajectory_finite: bool = True,
) -> TrajectoryBundle:
    arrays = {
        member_name.removesuffix(".npy"): _load_mmap(directory / member_name)
        for member_name in _ARRAY_MEMBERS
    }
    try:
        return _bundle_from_runtime_arrays(
            arrays,
            manifest,
            validate_trajectory_finite=validate_trajectory_finite,
        )
    except BaseException:
        _close_memmaps(arrays.values(), suppress_errors=True)
        raise


def _bundle_from_runtime_arrays(
    arrays: dict[str, np.memmap],
    manifest: dict[str, Any],
    *,
    validate_trajectory_finite: bool,
) -> TrajectoryBundle:
    conditions = arrays["conditions"]
    trajectories = arrays["trajectories"]
    time_axis = arrays["time_axis"]
    run_ids_array = arrays["run_ids"]
    rocks_array = arrays["rocks"]
    condition_features_array = arrays["condition_features"]
    output_features_array = arrays["output_features"]

    expected_conditions = tuple(manifest["arrays"]["conditions"]["shape"])
    expected_trajectories = tuple(manifest["arrays"]["trajectories"]["shape"])
    expected_time = tuple(manifest["arrays"]["time_axis"]["shape"])
    if conditions.shape != expected_conditions:
        raise ValueError(
            f"runtime condition shape does not match manifest: {conditions.shape}"
        )
    if trajectories.shape != expected_trajectories:
        raise ValueError(
            f"runtime trajectory shape does not match manifest: {trajectories.shape}"
        )
    if time_axis.shape != expected_time:
        raise ValueError(f"runtime time shape does not match manifest: {time_axis.shape}")
    if conditions.ndim != 2 or conditions.shape[1] != len(CONDITION_FEATURES):
        raise ValueError(
            "runtime condition feature count does not match the model contract"
        )
    if trajectories.ndim != 3 or trajectories.shape[2] != len(OUTPUT_FEATURES):
        raise ValueError(
            "runtime trajectory output feature count does not match the model contract"
        )
    if trajectories.shape[1] != len(time_axis):
        raise ValueError("runtime trajectory length does not match the time axis")
    for name, array in (
        ("conditions", conditions),
        ("trajectories", trajectories),
        ("time_axis", time_axis),
    ):
        expected_dtype = manifest["arrays"][name]["dtype"]
        if array.dtype.name != expected_dtype:
            raise ValueError(
                f"runtime {name} dtype does not match manifest: {array.dtype.name}"
            )
    if not np.isfinite(conditions).all():
        raise ValueError("runtime conditions must contain only finite values")
    if not np.isfinite(time_axis).all() or not np.all(np.diff(time_axis) > 0):
        raise ValueError("runtime time axis must be finite and strictly increasing")
    if validate_trajectory_finite and not _trajectory_is_finite(trajectories):
        raise ValueError("runtime trajectory values must contain only finite values")

    metadata_arrays = (
        run_ids_array,
        rocks_array,
        condition_features_array,
        output_features_array,
    )
    if any(array.ndim != 1 for array in metadata_arrays):
        raise ValueError("runtime metadata arrays must be one-dimensional")
    run_ids = run_ids_array.astype(str).tolist()
    rocks = rocks_array.astype(str)
    condition_features = tuple(condition_features_array.astype(str).tolist())
    output_features = tuple(output_features_array.astype(str).tolist())
    n_runs = conditions.shape[0]
    if trajectories.shape[0] != n_runs or len(run_ids) != n_runs or len(rocks) != n_runs:
        raise ValueError("runtime run arrays and metadata lengths do not match")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("runtime run ids must be unique")
    if condition_features != CONDITION_FEATURES:
        raise ValueError("runtime condition feature order does not match the model contract")
    if output_features != OUTPUT_FEATURES:
        raise ValueError("runtime output feature order does not match the model contract")

    return TrajectoryBundle(
        conditions=conditions,
        trajectories=trajectories,
        time_axis=time_axis,
        run_ids=run_ids,
        rocks=rocks,
        condition_features=condition_features,
        output_features=output_features,
    )


def _load_mmap(path: Path) -> np.memmap:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError as exc:
        if "Python objects" in str(exc) or "pickle" in str(exc).lower():
            raise ValueError(f"runtime metadata must be pickle-free: {path}") from exc
        raise
    if not isinstance(array, np.memmap):
        raise ValueError(f"runtime array was not opened as a memmap: {path}")
    return array


def _validated_member_info(
    archive: zipfile.ZipFile,
) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or set(names) != set(_ARRAY_MEMBERS):
        raise ValueError(
            "bundle.npz members do not match the required pickle-free array members"
        )
    return {info.filename: info for info in infos}


def _copy_zip_member(
    *,
    archive: zipfile.ZipFile,
    member_name: str,
    destination: Path,
) -> None:
    with archive.open(member_name, mode="r") as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=ZIP_COPY_BUFFER_BYTES)


def _source_bundle_sha256(manifest: dict[str, Any]) -> str:
    try:
        value = manifest["artifacts"]["bundle.npz"]["sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Cache manifest is missing bundle.npz SHA-256") from exc
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("Cache manifest bundle.npz SHA-256 is invalid")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError("Cache manifest bundle.npz SHA-256 is invalid") from exc
    return value


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(ZIP_COPY_BUFFER_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def _file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "inode": stat.st_ino,
        "device": stat.st_dev,
    }


def _trajectory_is_finite(trajectories: np.ndarray) -> bool:
    for start in range(0, trajectories.shape[0], FINITE_CHECK_RUN_CHUNK):
        chunk = trajectories[start : start + FINITE_CHECK_RUN_CHUNK]
        if not np.isfinite(chunk).all():
            return False
    return True


def _close_bundle_memmaps(bundle: TrajectoryBundle) -> None:
    _close_memmaps(
        (bundle.conditions, bundle.trajectories, bundle.time_axis),
        suppress_errors=False,
    )


def _close_memmaps(arrays, *, suppress_errors: bool) -> None:
    errors: list[Exception] = []
    for array in arrays:
        if not isinstance(array, np.memmap):
            continue
        try:
            array._mmap.close()
        except Exception as exc:
            errors.append(exc)
    if errors and not suppress_errors:
        raise errors[0]


def _remove_staging(staging_dir: Path) -> None:
    shutil.rmtree(staging_dir, ignore_errors=True)
    if staging_dir.exists():
        warnings.warn(
            f"Could not remove incomplete runtime directory: {staging_dir}",
            RuntimeWarning,
            stacklevel=2,
        )
