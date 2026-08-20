from __future__ import annotations

import logging
import math
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

NORMALIZED_TARGET_DISK_HEADROOM = 1.05
NPY_HEADER_ALLOWANCE_BYTES = 4096
CONDITION_STD_EPSILON = 1e-12

LOGGER = logging.getLogger(__name__)


class ConditionScaler:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, conditions: np.ndarray) -> ConditionScaler:
        candidate = StandardScaler().fit(conditions)
        _guard_zero_variance(candidate)
        self.scaler = candidate
        self.fitted = True
        return self

    def fit_indexed(
        self,
        conditions: np.ndarray,
        indices: np.ndarray,
        *,
        chunk_runs: int,
    ) -> ConditionScaler:
        if conditions.ndim != 2:
            raise ValueError(f"conditions must be 2D, got {conditions.shape}")
        selected = _validated_indices(indices, conditions.shape[0], chunk_runs)
        candidate = StandardScaler()
        for start in range(0, len(selected), chunk_runs):
            index_chunk = selected[start : start + chunk_runs]
            candidate.partial_fit(conditions[index_chunk])
        _guard_zero_variance(candidate)
        self.scaler = candidate
        self.fitted = True
        return self

    def transform(self, conditions: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ConditionScaler must be fitted before transform")
        return self.scaler.transform(conditions).astype(np.float32)


def _guard_zero_variance(
    scaler: StandardScaler,
    *,
    eps: float = CONDITION_STD_EPSILON,
) -> None:
    """Pass constant condition columns through instead of dividing by ~0.

    The fixed mineral dictionary contributes columns that are all-zero for a given
    rock mix (for example `EPIDOTE_MOLES` in the five-rock pilot). Their std is 0,
    so the divisor is forced to 1.0 and the column stays exactly zero after z-scoring.
    """
    scale = getattr(scaler, "scale_", None)
    if scale is None:
        return
    scale[~np.isfinite(scale) | (scale <= eps)] = 1.0


class OutputScaler:
    def __init__(self, log_feature_indices: tuple[int, ...]) -> None:
        self.log_feature_indices = log_feature_indices
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, trajectories: np.ndarray) -> OutputScaler:
        candidate = StandardScaler().fit(self._prepare(trajectories))
        self.scaler = candidate
        self.fitted = True
        return self

    def fit_indexed(
        self,
        trajectories: np.ndarray,
        indices: np.ndarray,
        *,
        chunk_runs: int,
    ) -> OutputScaler:
        if trajectories.ndim != 3:
            raise ValueError(f"trajectories must be 3D, got {trajectories.shape}")
        selected = _validated_indices(indices, trajectories.shape[0], chunk_runs)
        candidate = StandardScaler()
        for start in range(0, len(selected), chunk_runs):
            index_chunk = selected[start : start + chunk_runs]
            candidate.partial_fit(self._prepare(trajectories[index_chunk]))
        self.scaler = candidate
        self.fitted = True
        return self

    def transform(self, trajectories: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("OutputScaler must be fitted before transform")
        original_shape = trajectories.shape
        transformed = self.scaler.transform(self._prepare(trajectories))
        return transformed.reshape(original_shape).astype(np.float32)

    def inverse_transform(self, trajectories_norm: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("OutputScaler must be fitted before inverse_transform")
        original_shape = trajectories_norm.shape
        flat = trajectories_norm.reshape(-1, original_shape[-1])
        restored = self.scaler.inverse_transform(flat)
        for index in self.log_feature_indices:
            restored[:, index] = np.expm1(restored[:, index])
        return restored.reshape(original_shape)

    def _prepare(self, trajectories: np.ndarray) -> np.ndarray:
        flat = np.array(trajectories, dtype=np.float64, copy=True).reshape(
            -1,
            trajectories.shape[-1],
        )
        for index in self.log_feature_indices:
            column = flat[:, index]
            np.maximum(column, 0.0, out=column)
            np.log1p(column, out=column)
        return flat


def estimate_normalized_target_bytes(shape: tuple[int, ...]) -> int:
    """Estimate fixed-contract float32 NPY bytes plus local-disk headroom."""
    if not shape or any(not isinstance(size, int) or size <= 0 for size in shape):
        raise ValueError("Normalized target shape must contain positive integers")
    raw_bytes = math.prod(shape) * np.dtype(np.float32).itemsize
    return (
        math.ceil(raw_bytes * NORMALIZED_TARGET_DISK_HEADROOM)
        + NPY_HEADER_ALLOWANCE_BYTES
    )


def materialize_normalized_targets(
    *,
    trajectories: np.ndarray,
    scaler: OutputScaler,
    output_path: Path | str,
    chunk_runs: int,
) -> np.memmap:
    """Transform raw targets once into an atomic read-only float32 memmap."""
    if trajectories.ndim != 3:
        raise ValueError(f"trajectories must be 3D, got {trajectories.shape}")
    if chunk_runs <= 0:
        raise ValueError("chunk_runs must be positive")
    if not scaler.fitted:
        raise RuntimeError("OutputScaler must be fitted before target materialization")

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite normalized targets: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    required_bytes = estimate_normalized_target_bytes(tuple(trajectories.shape))
    available_bytes = shutil.disk_usage(output_path.parent).free
    if available_bytes < required_bytes:
        raise OSError(
            "Insufficient normalized-target disk space: "
            f"required={required_bytes} available={available_bytes}"
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.building-",
        suffix=".npy",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    target_map: np.memmap | None = None
    loaded: np.memmap | None = None
    published = False
    primary_error: BaseException | None = None
    try:
        target_map = np.lib.format.open_memmap(
            temporary_path,
            mode="w+",
            dtype=np.float32,
            shape=trajectories.shape,
        )
        for start in range(0, trajectories.shape[0], chunk_runs):
            stop = min(start + chunk_runs, trajectories.shape[0])
            source_chunk = trajectories[start:stop]
            transformed = scaler.transform(source_chunk)
            if transformed.shape != source_chunk.shape:
                raise ValueError("Normalized target chunk shape changed during transform")
            if not np.isfinite(transformed).all():
                raise ValueError("Normalized target chunks must contain only finite values")
            target_map[start:stop] = transformed
        target_map.flush()
        target_map._mmap.close()
        target_map = None
        loaded = np.load(temporary_path, mmap_mode="r", allow_pickle=False)
        if not isinstance(loaded, np.memmap):
            raise ValueError(
                f"Normalized targets were not opened as a memmap: {temporary_path}"
            )
        if loaded.shape != trajectories.shape or loaded.dtype != np.dtype(np.float32):
            raise ValueError("Normalized target staging file changed shape or dtype")
        if loaded.flags.writeable:
            raise ValueError("Normalized target staging memmap must be read-only")
        loaded.filename = str(output_path)
        try:
            os.link(temporary_path, output_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite normalized targets: {output_path}"
            ) from exc
        published = True
    except BaseException as exc:
        primary_error = exc
        if loaded is not None:
            try:
                loaded._mmap.close()
            except Exception as close_error:
                exc.add_note(
                    f"Could not close normalized target staging memmap: {close_error}"
                )
            loaded = None
        raise
    finally:
        if target_map is not None:
            try:
                target_map._mmap.close()
            except Exception as close_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    f"Could not close normalized target write memmap: {close_error}"
                )
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            if primary_error is None and published:
                LOGGER.warning(
                    "Normalized targets were published, but the staging hard link "
                    "could not be removed: %s: %s",
                    temporary_path,
                    cleanup_error,
                )
            elif primary_error is None:
                raise
            else:
                primary_error.add_note(
                    f"Could not remove staging file {temporary_path}: {cleanup_error}"
                )

    if loaded is None:
        raise RuntimeError("Normalized targets were published without a readable memmap")
    return loaded


def _validated_indices(
    indices: np.ndarray,
    n_runs: int,
    chunk_runs: int,
) -> np.ndarray:
    if chunk_runs <= 0:
        raise ValueError("chunk_runs must be positive")
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
    return selected.astype(np.int64, copy=False)


@dataclass
class PreprocessorBundle:
    condition_scaler: ConditionScaler
    output_scaler: OutputScaler
    time_mean: float
    time_std: float

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file_obj:
            pickle.dump(self, file_obj)

    @classmethod
    def load(cls, path: Path | str) -> PreprocessorBundle:
        with Path(path).open("rb") as file_obj:
            return pickle.load(file_obj)
