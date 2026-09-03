"""Leave-one-rock-out driver: one ``run_training`` fold per held-out rock.

Usage (Colab or local)::

    python -m conditional_model_v1.cli.loro \
        --config configs/conditional_model_v1/loro_colab_eight_rocks.yaml \
        [--held-out-rocks Calcite Trona] \
        [--reference-run-dir /content/runs/<all-rocks run>]

Each fold prints its own ``run_dir=...`` line (from ``run_training``) so the
notebook can copy folds to Drive as they finish; the aggregate folder is
printed last as ``loro_dir=...``. ``loro_summary.csv`` is rewritten after every
fold, so an interrupted session keeps the finished folds.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from datetime import datetime
from pathlib import Path

from conditional_model_v1.cli.train import run_training
from conditional_model_v1.config import ExperimentConfig, load_config
from conditional_model_v1.data import OUTPUT_FEATURES
from conditional_model_v1.loro import (
    LoroFoldResult,
    copy_held_out_overviews,
    fold_config,
    load_reference_per_rock,
    load_reference_run_rmse,
    read_fold_result,
    read_fold_rock_feature_rows,
    resolve_held_out_rocks,
    summary_rows,
    unseen_run_rmse,
    write_loro_rock_feature_metrics,
    write_loro_summary,
)
from conditional_model_v1.plotting import (
    plot_loro_feature_heatmap,
    plot_loro_rock_bars,
    plot_loro_unseen_boxplot,
)
from conditional_model_v1.tracking import record_loro_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-rock-out generalization test")
    parser.add_argument("--config", required=True, help="YAML config with split.strategy=leave_one_rock_out")
    parser.add_argument(
        "--held-out-rocks",
        nargs="+",
        default=None,
        help="Subset of rocks to hold out (default: split.held_out_rocks from the config)",
    )
    parser.add_argument(
        "--reference-run-dir",
        default=None,
        help="Run folder of the all-rocks model, for seen-vs-unseen comparison plots",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    run_loro(
        config=config,
        config_path=config_path,
        held_out_rocks=tuple(args.held_out_rocks) if args.held_out_rocks else None,
        reference_run_dir=Path(args.reference_run_dir) if args.reference_run_dir else None,
    )


def run_loro(
    *,
    config: ExperimentConfig,
    config_path: Path,
    held_out_rocks: tuple[str, ...] | None = None,
    reference_run_dir: Path | None = None,
) -> Path:
    """Train every fold sequentially and aggregate the results into one folder."""
    if config.data.split.strategy != "leave_one_rock_out":
        raise ValueError("cli.loro needs split.strategy=leave_one_rock_out")
    rocks = resolve_held_out_rocks(config, held_out_rocks)
    reference = load_reference_per_rock(reference_run_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loro_dir = Path(config.run_root) / f"{timestamp}_{config.name}_loro"
    plots_dir = loro_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, loro_dir / "config.yaml")
    _write_json(
        loro_dir / "loro_config.json",
        {
            "held_out_rocks": list(rocks),
            "reference_run_dir": None if reference_run_dir is None else str(reference_run_dir),
            "split": {
                "train": config.data.split.train,
                "val": config.data.split.val,
                "test": config.data.split.test,
                "seed": config.data.split.seed,
            },
            "epochs": config.training.epochs,
        },
    )
    print(f"loro held_out_rocks={list(rocks)} loro_dir={loro_dir}")

    folds: list[LoroFoldResult] = []
    feature_rows: list[dict] = []
    unseen_rmse: dict[str, list[float]] = {}
    for index, rock in enumerate(rocks, start=1):
        print(f"loro_fold_start={rock} ({index}/{len(rocks)})")
        run_dir = run_training(
            config=fold_config(config, rock),
            config_path=config_path,
            held_out_rock=rock,
        )
        folds.append(read_fold_result(run_dir, rock))
        feature_rows.extend(read_fold_rock_feature_rows(run_dir, rock))
        unseen_rmse[rock] = unseen_run_rmse(run_dir, rock).tolist()
        copy_held_out_overviews(run_dir, rock, plots_dir / "rock_overviews")
        write_loro_summary(loro_dir / "loro_summary.csv", folds, reference)
        _cleanup_fold_runtime(config, run_dir)
        _free_memory()
        print(f"loro_fold_done={rock}")

    write_loro_rock_feature_metrics(loro_dir / "loro_rock_feature_metrics.csv", feature_rows)
    write_loro_rock_feature_metrics(
        loro_dir / "loro_run_rmse.csv",
        [
            {"held_out_rock": rock, "rmse_normalized": value}
            for rock in rocks
            for value in unseen_rmse[rock]
        ],
    )
    rows = summary_rows(folds, reference)
    _write_json(
        loro_dir / "loro_metrics.json",
        {
            "loro_name": loro_dir.name,
            "experiment": config.name,
            "held_out_rocks": list(rocks),
            "reference_run_dir": None if reference_run_dir is None else str(reference_run_dir),
            "folds": rows,
            "reference_per_rock": reference,
        },
    )
    record_loro_folds(
        Path(config.run_root) / "registry.sqlite",
        loro_name=loro_dir.name,
        loro_dir=loro_dir,
        folds=[fold.as_row() for fold in folds],
    )
    plot_loro_rock_bars(
        rows=rows,
        path_normalized=plots_dir / "loro_rmse_normalized_by_rock.png",
        path_original=plots_dir / "loro_rmse_original_by_rock.png",
    )
    plot_loro_feature_heatmap(
        feature_rows=feature_rows,
        rocks=rocks,
        output_features=OUTPUT_FEATURES,
        path=plots_dir / "loro_unseen_rmse_normalized_heatmap.png",
    )
    plot_loro_unseen_boxplot(
        unseen=unseen_rmse,
        reference=load_reference_run_rmse(reference_run_dir) or None,
        rocks=rocks,
        path=plots_dir / "loro_unseen_rmse_boxplot.png",
    )
    print(f"loro_dir={loro_dir}")
    return loro_dir


def _cleanup_fold_runtime(config: ExperimentConfig, run_dir: Path) -> None:
    """Drop the fold's large normalized-target mmap (Colab disk)."""
    runtime_dir = Path(config.data.processed_root) / ".runtime" / "training" / run_dir.name
    shutil.rmtree(runtime_dir, ignore_errors=True)


def _free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
