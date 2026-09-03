"""Leave-one-rock-out (LORO) bookkeeping shared by ``cli.loro`` and tests.

One LORO experiment trains K models: fold *k* holds rock *k* out of training
and validation entirely and evaluates on it (``test_unseen``) plus, when
``split.test > 0``, on the remaining rocks' test share (``test_seen``). Each
fold is a normal ``run_training`` run with its own run folder; this module
collects those folds into one aggregate folder (``loro_summary.csv``,
``loro_rock_feature_metrics.csv``, ``loro_metrics.json``, plots).

Normalized RMSE values come from each fold's own output scaler (fitted on that
fold's training rocks), so they are comparable in spirit, not byte-for-byte.
Original-scale RMSE is in chemistry units and directly comparable.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conditional_model_v1.config import ExperimentConfig

HELD_OUT_OVERVIEW_KINDS: tuple[str, ...] = ("best", "worst", "mean")


@dataclass(frozen=True)
class LoroFoldResult:
    """Headline numbers of one fold, read back from its ``metrics.json``."""

    held_out_rock: str
    run_name: str
    run_dir: str
    n_train: int
    n_val: int
    n_test_unseen: int
    n_test_seen: int
    unseen_rmse_mean_original: float
    unseen_rmse_mean_normalized: float
    unseen_mae_mean_original: float
    seen_rmse_mean_original: float | None
    seen_rmse_mean_normalized: float | None
    best_val_loss: float

    def as_row(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def resolve_held_out_rocks(
    config: ExperimentConfig,
    override: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    """Rocks to hold out, in ``data.datasets`` order.

    ``override`` (the ``--held-out-rocks`` CLI flag) wins over
    ``split.held_out_rocks``; both must name rocks present in the config.
    """
    configured = tuple(dict.fromkeys(dataset.rock for dataset in config.data.datasets))
    if override:
        unknown = sorted(set(override) - set(configured))
        if unknown:
            raise ValueError(f"--held-out-rocks names rocks missing from data.datasets: {unknown}")
        return tuple(rock for rock in configured if rock in set(override))
    if config.data.split.held_out_rocks == "all":
        return configured
    wanted = set(config.data.split.held_out_rocks)
    return tuple(rock for rock in configured if rock in wanted)


def fold_config(config: ExperimentConfig, held_out_rock: str) -> ExperimentConfig:
    """Same experiment, unique name per fold (run dir, runtime files, sqlite key)."""
    return dataclasses.replace(config, name=f"{config.name}_loro_{safe_rock_name(held_out_rock)}")


def safe_rock_name(rock: str) -> str:
    return rock.replace("/", "_").replace("\\", "_").replace(" ", "_")


def read_fold_result(run_dir: Path | str, held_out_rock: str) -> LoroFoldResult:
    """Read one finished fold's ``metrics.json`` into a :class:`LoroFoldResult`."""
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    if metrics.get("held_out_rock") != held_out_rock:
        raise ValueError(
            f"{run_dir} reports held_out_rock={metrics.get('held_out_rock')!r}, "
            f"expected {held_out_rock!r}"
        )
    unseen = metrics.get("test_unseen")
    if not unseen:
        raise ValueError(f"{run_dir} has no test_unseen block; is it a LORO fold?")
    seen = metrics.get("test_seen") or {}
    return LoroFoldResult(
        held_out_rock=held_out_rock,
        run_name=run_dir.name,
        run_dir=str(run_dir),
        n_train=int(metrics["n_train_runs"]),
        n_val=int(metrics["n_val_runs"]),
        n_test_unseen=int(unseen["n_runs"]),
        n_test_seen=int(seen.get("n_runs", 0)),
        unseen_rmse_mean_original=float(unseen["rmse_mean_original"]),
        unseen_rmse_mean_normalized=float(unseen["rmse_mean_normalized"]),
        unseen_mae_mean_original=float(unseen["mae_mean_original"]),
        seen_rmse_mean_original=_optional_float(seen.get("rmse_mean_original")),
        seen_rmse_mean_normalized=_optional_float(seen.get("rmse_mean_normalized")),
        best_val_loss=float(metrics["best_val_loss"]),
    )


def read_fold_rock_feature_rows(run_dir: Path | str, held_out_rock: str) -> list[dict[str, Any]]:
    """Per-feature RMSE rows of the held-out rock from ``rock_feature_metrics.csv``."""
    run_dir = Path(run_dir)
    with (run_dir / "rock_feature_metrics.csv").open(newline="") as file_obj:
        rows = [row for row in csv.DictReader(file_obj) if row["rock"] == held_out_rock]
    if not rows:
        raise ValueError(f"{run_dir} has no rock_feature_metrics rows for {held_out_rock!r}")
    return [
        {
            "held_out_rock": held_out_rock,
            "run_name": run_dir.name,
            "feature": row["feature"],
            "n_runs": int(row["n_runs"]),
            "rmse_original": float(row["rmse_original"]),
            "rmse_normalized": float(row["rmse_normalized"]),
        }
        for row in rows
    ]


def load_reference_per_rock(reference_run_dir: Path | str | None) -> dict[str, dict[str, float]]:
    """Per-rock RMSE of the all-rocks reference run (empty when not given)."""
    if reference_run_dir is None:
        return {}
    metrics = json.loads((Path(reference_run_dir) / "metrics.json").read_text())
    if metrics.get("held_out_rock") is not None:
        raise ValueError("reference run must be an all-rocks run, not a LORO fold")
    return {
        rock: {
            "rmse_mean_original": float(values["rmse_mean_original"]),
            "rmse_mean_normalized": float(values["rmse_mean_normalized"]),
        }
        for rock, values in metrics["per_rock"].items()
    }


def summary_rows(
    folds: list[LoroFoldResult],
    reference: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Fold rows joined with the reference run's per-rock RMSE."""
    rows = []
    for fold in folds:
        ref = reference.get(fold.held_out_rock, {})
        ref_norm = ref.get("rmse_mean_normalized")
        rows.append(
            {
                **fold.as_row(),
                "reference_rmse_mean_original": ref.get("rmse_mean_original"),
                "reference_rmse_mean_normalized": ref_norm,
                "unseen_over_reference_normalized": (
                    fold.unseen_rmse_mean_normalized / ref_norm if ref_norm else None
                ),
            }
        )
    return rows


def write_loro_summary(
    path: Path | str,
    folds: list[LoroFoldResult],
    reference: dict[str, dict[str, float]],
) -> None:
    """Rewrite the one-row-per-fold summary (called after every fold)."""
    rows = summary_rows(folds, reference)
    _write_csv(Path(path), rows)


def write_loro_rock_feature_metrics(path: Path | str, rows: list[dict[str, Any]]) -> None:
    _write_csv(Path(path), rows)


def copy_held_out_overviews(
    fold_run_dir: Path | str,
    held_out_rock: str,
    destination: Path | str,
) -> list[Path]:
    """Copy the held-out rock's best/worst/mean overview PNGs into the aggregate folder."""
    source_dir = Path(fold_run_dir) / "plots" / "rock_overviews"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for kind in HELD_OUT_OVERVIEW_KINDS:
        name = f"{safe_rock_name(held_out_rock)}_{kind}_overview.png"
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Fold overview missing: {source}")
        target = destination / name
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
