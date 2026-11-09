"""Save a RunTable to a cache folder and load it again.

The cache folder has one .npy file for each RunTable array and one
manifest.json file. The manifest is written last, so a build that stops
before the end leaves no valid cache.

The manifest keeps a fingerprint of the source files. The fingerprint uses
the name, the size and the modification time of each file, not the content,
because a hash of 10 GB of text takes too long. If a file changes, the
fingerprint changes and the cache is built again.

Format choice: .npy files, not one .npz file. numpy.load can memory-map a
.npy file, but not an array in an .npz file. Thus the 3.5 GB output array is
read from disk only when a feature is used. The files are not compressed,
because float64 simulation output compresses to only about 78 % of its size.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import pathlib
from collections.abc import Sequence

import numpy as np

from conditional_input_optimization.run_data import (
    rock_source,
    run_reader,
    run_table,
)

SCHEMA_VERSION = 1

_MANIFEST_FILE = "manifest.json"
_ARRAY_NAMES = ("rocks", "run_ids", "conditions", "time_days", "outputs")
# outputs is the only large array. The others are loaded into memory.
_MEMORY_MAPPED = frozenset({"outputs"})


class CacheMismatchError(ValueError):
    """The cache folder does not match the source files or the schema."""


def source_fingerprint(sources: Sequence[rock_source.RockSource]) -> str:
    """Return a SHA-256 fingerprint of the sources and their files.

    The fingerprint covers the schema version, the feature lists, the rock
    names and units, and the name, size and modification time of each file
    in input/ and output/. It does not cover the folder path, so a moved
    repository keeps a valid cache.

    Complexity: O(F log F) time for F files, because of the sort, plus one
    stat call for each file.

    Args:
        sources: The rocks, in table order.

    Returns:
        The fingerprint as 64 hexadecimal characters.

    Raises:
        FileNotFoundError: If an input/ or output/ folder does not exist.
    """
    digest = hashlib.sha256()
    header = {
        "schema_version": SCHEMA_VERSION,
        "condition_features": run_reader.CONDITION_FEATURES,
        "output_features": run_reader.OUTPUT_FEATURES,
        "rocks": [[s.name, s.input_units] for s in sources],
    }
    digest.update(json.dumps(header).encode())
    for source in sources:
        for folder in (source.input_folder(), source.output_folder()):
            digest.update(_folder_listing(folder).encode())
    return digest.hexdigest()


def is_cache_valid(cache_dir: pathlib.Path, fingerprint: str) -> bool:
    """Return True if cache_dir has a complete cache with this fingerprint.

    Args:
        cache_dir: The cache folder. It can be absent.
        fingerprint: The value from source_fingerprint.

    Returns:
        True if the manifest exists and matches. Else False.
    """
    manifest_path = cache_dir / _MANIFEST_FILE
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    return (
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("fingerprint") == fingerprint
    )


def save_run_table(
    table: run_table.RunTable, cache_dir: pathlib.Path, fingerprint: str
) -> None:
    """Write the table to cache_dir. Replace a cache that is there.

    Complexity: O(size of the table) time and disk space, about 3.5 GB for
    55,122 runs.

    Args:
        table: The table to save.
        cache_dir: The cache folder. It is made if it does not exist.
        fingerprint: The value from source_fingerprint for the table sources.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Remove the old manifest first: a stop during the write then leaves an
    # invalid cache, not a valid manifest next to half-written arrays.
    (cache_dir / _MANIFEST_FILE).unlink(missing_ok=True)
    for name in _ARRAY_NAMES:
        np.save(cache_dir / f"{name}.npy", getattr(table, name))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "n_runs": len(table.rocks),
        "n_timesteps": len(table.time_days),
        "output_features": list(run_reader.OUTPUT_FEATURES),
        "rocks": sorted(set(table.rocks.tolist())),
    }
    temporary = cache_dir / f"{_MANIFEST_FILE}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, cache_dir / _MANIFEST_FILE)


def load_run_table(
    cache_dir: pathlib.Path, fingerprint: str
) -> run_table.RunTable:
    """Load the table from cache_dir.

    The outputs array is memory-mapped in read-only mode. The operating
    system reads a part of it from disk only when the code uses that part.

    Args:
        cache_dir: The cache folder.
        fingerprint: The expected value from source_fingerprint.

    Returns:
        The RunTable. All its arrays are read-only.

    Raises:
        CacheMismatchError: If the cache is absent, incomplete or stale.
    """
    if not is_cache_valid(cache_dir, fingerprint):
        raise CacheMismatchError(
            f"{cache_dir} has no valid cache for fingerprint "
            f"{fingerprint[:12]}. Build it with load_or_build_run_table."
        )
    arrays = {name: _load_array(cache_dir, name) for name in _ARRAY_NAMES}
    return run_table.RunTable(**arrays)


def load_or_build_run_table(
    sources: Sequence[rock_source.RockSource],
    cache_dir: pathlib.Path,
    max_workers: int | None = None,
    chunk_size: int = run_table.DEFAULT_CHUNK_SIZE,
) -> run_table.RunTable:
    """Load the cached RunTable, or build and save it first.

    Design pattern: Facade. A notebook or a script needs one call. The call
    hides the fingerprint, the process pool and the file format.

    Args:
        sources: The rocks, in table order.
        cache_dir: The cache folder.
        max_workers: The number of worker processes for a build. None uses
            the ProcessPoolExecutor default, os.cpu_count().
        chunk_size: The number of runs in one worker task.

    Returns:
        The RunTable, loaded from the cache.
    """
    fingerprint = source_fingerprint(sources)
    if not is_cache_valid(cache_dir, fingerprint):
        with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
            table = run_table.build_run_table(sources, executor, chunk_size)
        save_run_table(table, cache_dir, fingerprint)
    return load_run_table(cache_dir, fingerprint)


def _load_array(cache_dir: pathlib.Path, name: str) -> np.ndarray:
    """Load one array without pickle and make it read-only."""
    mmap_mode = "r" if name in _MEMORY_MAPPED else None
    array = np.load(
        cache_dir / f"{name}.npy", mmap_mode=mmap_mode, allow_pickle=False
    )
    array.setflags(write=False)
    return array


def _folder_listing(folder: pathlib.Path) -> str:
    """Return one sorted line "name size mtime_ns" for each file in folder."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    with os.scandir(folder) as entries:
        lines = sorted(
            f"{entry.name} {stat.st_size} {stat.st_mtime_ns}"
            for entry in entries
            if entry.is_file()
            for stat in (entry.stat(),)
        )
    return f"{folder.name}\n" + "\n".join(lines) + "\n"
