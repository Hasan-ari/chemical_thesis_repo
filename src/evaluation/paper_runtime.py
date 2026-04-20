from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


_RUNTIME_CACHE_DIR = Path(tempfile.gettempdir()) / "chemical_thesis_runtime"
_MPLCONFIGDIR = _RUNTIME_CACHE_DIR / "matplotlib"


def configure_paper_runtime() -> None:
    _RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(_RUNTIME_CACHE_DIR))
    os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))


def validate_repo_env_python(
    repo_root: str | Path,
    executable: str | Path | None = None,
) -> Path:
    repo_root_path = Path(repo_root).resolve()
    actual = Path(executable or sys.executable).resolve()
    expected = (repo_root_path / "env" / "bin" / "python").resolve()
    if actual != expected:
        raise RuntimeError(
            f"Expected repo env interpreter at {expected}, got {actual}. "
            f"Run this workflow with {expected}."
        )
    return actual


configure_paper_runtime()
