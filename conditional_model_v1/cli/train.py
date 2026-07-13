from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from conditional_model_v1.config import ExperimentConfig, load_config
from conditional_model_v1.data import (
    LOG_OUTPUT_FEATURES,
    DatasetSpec,
    IndexedTrajectoryDataset,
    build_bundle,
    load_cached_bundle,
    write_processed_bundle,
)
from conditional_model_v1.metrics import (
    evaluate_by_rock,
    regression_metrics_original_scale,
)
from conditional_model_v1.models import ConditionTimeLSTM
from conditional_model_v1.plotting import plot_loss_curve, plot_rock_overviews
from conditional_model_v1.preprocessing import (
    ConditionScaler,
    OutputScaler,
    PreprocessorBundle,
    materialize_normalized_targets,
)
from conditional_model_v1.preparation import validate_cache_manifest
from conditional_model_v1.runtime_data import materialize_mmap_bundle
from conditional_model_v1.splitting import rock_aware_split
from conditional_model_v1.tracking import ExperimentTracker
from conditional_model_v1.training import (
    get_device,
    predict_with_targets,
    set_seed,
    train_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train condition-to-trajectory LSTM v1")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    run_training(config=config, config_path=config_path)


def run_training(*, config: ExperimentConfig, config_path: Path) -> Path:
    """Run the full Colab-friendly training pipeline for one config."""
    set_seed(config.training.seed)
    tracker = ExperimentTracker(config)
    tracker.copy_config(config_path)

    bundle = _load_or_build_bundle(config)
    split = rock_aware_split(
        rocks=bundle.rocks,
        train_ratio=config.data.split.train,
        val_ratio=config.data.split.val,
        test_ratio=config.data.split.test,
        seed=config.data.split.seed,
    )

    log_indices = tuple(
        index for index, feature in enumerate(bundle.output_features) if feature in LOG_OUTPUT_FEATURES
    )
    chunk_runs = config.training.preprocessing_chunk_runs
    condition_scaler = ConditionScaler().fit_indexed(
        bundle.conditions,
        split.train,
        chunk_runs=chunk_runs,
    )
    output_scaler = OutputScaler(log_feature_indices=log_indices).fit_indexed(
        bundle.trajectories,
        split.train,
        chunk_runs=chunk_runs,
    )
    time_mean = float(bundle.time_axis.mean())
    time_std = float(bundle.time_axis.std())
    preprocessors = PreprocessorBundle(
        condition_scaler=condition_scaler,
        output_scaler=output_scaler,
        time_mean=time_mean,
        time_std=time_std,
    )
    preprocessors.save(tracker.run_dir / "preprocessors.pkl")

    conditions_norm = condition_scaler.transform(bundle.conditions)
    safe_time_std = time_std if time_std > 0 else 1.0
    time_norm = ((bundle.time_axis - time_mean) / safe_time_std).astype(np.float32)
    normalized_target_path = (
        Path(config.data.processed_root)
        / ".runtime"
        / "training"
        / tracker.run_dir.name
        / "targets_norm.npy"
    )
    normalized_targets = materialize_normalized_targets(
        trajectories=bundle.trajectories,
        scaler=output_scaler,
        output_path=normalized_target_path,
        chunk_runs=chunk_runs,
    )
    normalized_targets._mmap.close()

    device = get_device()
    pin_memory = device.type == "cuda"

    train_loader = _make_loader(
        conditions_norm=conditions_norm,
        time_norm=time_norm,
        targets_path=normalized_target_path,
        indices=split.train,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        seed=config.training.seed,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    val_loader = (
        _make_loader(
            conditions_norm=conditions_norm,
            time_norm=time_norm,
            targets_path=normalized_target_path,
            indices=split.val,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            seed=config.training.seed + 1,
            pin_memory=pin_memory,
            persistent_workers=True,
        )
        if len(split.val) > 0
        else None
    )

    model = ConditionTimeLSTM(
        input_size=bundle.conditions.shape[1] + 1,
        output_size=bundle.trajectories.shape[2],
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
    )
    print(f"device={device}")
    x_shape = (
        bundle.conditions.shape[0],
        len(bundle.time_axis),
        bundle.conditions.shape[1] + 1,
    )
    print(f"x_shape={x_shape} y_shape={bundle.trajectories.shape}")
    print(f"split train={len(split.train)} val={len(split.val)} test={len(split.test)}")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.training,
        checkpoint_dir=tracker.checkpoint_dir,
        device=device,
    )
    tracker.write_history(history)
    plot_loss_curve(history, tracker.plot_dir / "loss_curve.png")
    del train_loader
    if val_loader is not None:
        del val_loader

    eval_indices = split.test if len(split.test) > 0 else split.val if len(split.val) > 0 else split.train
    eval_name = "test" if len(split.test) > 0 else "val" if len(split.val) > 0 else "train"
    model.load_state_dict(torch.load(tracker.checkpoint_dir / "best.pt", map_location=device))
    eval_loader = _make_loader(
        conditions_norm=conditions_norm,
        time_norm=time_norm,
        targets_path=normalized_target_path,
        indices=eval_indices,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        seed=config.training.seed + 2,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
    y_pred_norm, y_true_norm = predict_with_targets(model, eval_loader, device)
    y_pred = output_scaler.inverse_transform(y_pred_norm).astype(np.float32)
    y_true = output_scaler.inverse_transform(y_true_norm).astype(np.float32)
    eval_run_ids = [bundle.run_ids[int(index)] for index in eval_indices]
    eval_rocks = np.asarray(bundle.rocks[eval_indices], dtype=np.str_)
    _write_eval_predictions(
        tracker.run_dir / "eval_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        y_true_norm=y_true_norm,
        y_pred_norm=y_pred_norm,
        time_axis=bundle.time_axis,
        run_ids=eval_run_ids,
        rocks=eval_rocks,
        output_features=bundle.output_features,
        eval_split=eval_name,
    )
    metrics = regression_metrics_original_scale(
        y_true=y_true,
        y_pred=y_pred,
        output_features=bundle.output_features,
    )
    rock_order = tuple(dict.fromkeys(dataset.rock for dataset in config.data.datasets))
    per_rock_metrics, rock_feature_rows, overviews = evaluate_by_rock(
        y_true=y_true,
        y_pred=y_pred,
        y_true_norm=y_true_norm,
        y_pred_norm=y_pred_norm,
        rocks=eval_rocks,
        run_ids=eval_run_ids,
        output_features=bundle.output_features,
        rock_order=rock_order,
    )
    metrics.update(
        {
            "eval_split": eval_name,
            "n_runs_total": int(bundle.conditions.shape[0]),
            "n_train_runs": int(len(split.train)),
            "n_val_runs": int(len(split.val)),
            "n_test_runs": int(len(split.test)),
            "x_shape": list(x_shape),
            "y_shape": list(bundle.trajectories.shape),
            "final_train_loss": float(history[-1]["train_loss"]),
            "final_val_loss": float(history[-1]["val_loss"]),
            "best_val_loss": float(min(row["val_loss"] for row in history)),
            "device": str(device),
            "per_rock": per_rock_metrics,
        }
    )
    tracker.write_metrics(metrics)
    tracker.write_feature_metrics(metrics)
    tracker.write_rock_feature_metrics(rock_feature_rows)
    tracker.record_registry(config, metrics, rock_feature_rows)
    plot_rock_overviews(
        time_axis=bundle.time_axis,
        output_features=bundle.output_features,
        overviews=overviews,
        output_dir=tracker.plot_dir / "rock_overviews",
    )
    print(f"run_dir={tracker.run_dir}")
    return tracker.run_dir


def _load_or_build_bundle(config: ExperimentConfig):
    """Use NPZ cache when available; otherwise parse raw txt and write processed files."""
    processed_dir = config.cache_dir
    cache_path = processed_dir / "bundle.npz"
    specs = _configured_dataset_specs(config)
    if config.data.use_cache and cache_path.exists() and not config.data.rebuild_cache:
        manifest = None
        if config.data.cache_name is not None:
            manifest = validate_cache_manifest(
                cache_dir=processed_dir,
                specs=specs,
            )
        if manifest is not None:
            bundle, _runtime_dir = materialize_mmap_bundle(
                cache_dir=processed_dir,
                manifest=manifest,
                runtime_root=Path(config.data.processed_root) / ".runtime",
            )
        else:
            try:
                bundle = load_cached_bundle(cache_path)
            except ValueError as exc:
                if "pickle-free Unicode metadata" not in str(exc):
                    raise
                if config.data.require_cache:
                    guidance = "Prepare a new versioned cache before training."
                else:
                    guidance = (
                        "Set data.rebuild_cache=true for an explicit raw-data rebuild, "
                        "or prepare a new versioned cache name."
                    )
                raise ValueError(
                    f"Legacy data cache is not safe to load: {cache_path}. {guidance}"
                ) from exc
        if manifest is not None:
            _validate_bundle_matches_manifest(bundle, manifest)
        return bundle
    if config.data.require_cache:
        raise FileNotFoundError(
            f"Required data cache does not exist: {cache_path}. "
            "Prepare it before starting training."
        )

    bundle, inputs, outputs, inventory = build_bundle(
        specs,
        keep_outputs_frame=config.data.write_outputs_csv,
    )
    write_processed_bundle(bundle, inputs, outputs, inventory, processed_dir)
    return bundle


def _configured_dataset_specs(config: ExperimentConfig) -> tuple[DatasetSpec, ...]:
    return tuple(
        DatasetSpec(
            name=dataset.name,
            rock=dataset.rock,
            path=dataset.path,
            max_runs=dataset.max_runs,
        )
        for dataset in config.data.datasets
    )


def _validate_bundle_matches_manifest(bundle, manifest: dict) -> None:
    if bundle.conditions.shape[0] != manifest["n_runs"]:
        raise ValueError(
            "Cache manifest n_runs does not match the loaded bundle: "
            f"{manifest['n_runs']} != {bundle.conditions.shape[0]}"
        )
    if len(bundle.time_axis) != manifest["n_timesteps"]:
        raise ValueError(
            "Cache manifest n_timesteps does not match the loaded bundle: "
            f"{manifest['n_timesteps']} != {len(bundle.time_axis)}"
        )


def _make_loader(
    *,
    conditions_norm: np.ndarray,
    time_norm: np.ndarray,
    targets_path: Path,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    """Build a DataLoader for a run-level split without exposing rock as model input."""
    dataset = IndexedTrajectoryDataset(
        conditions_norm=conditions_norm,
        time_norm=time_norm,
        targets_path=targets_path,
        indices=indices,
    )
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = 1
    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def _write_eval_predictions(
    path: Path,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    time_axis: np.ndarray,
    run_ids: list[str],
    rocks: np.ndarray,
    output_features: tuple[str, ...],
    eval_split: str,
) -> None:
    """Write numeric predictions with pickle-free Unicode metadata."""
    np.savez_compressed(
        path,
        y_true=y_true,
        y_pred=y_pred,
        y_true_norm=y_true_norm,
        y_pred_norm=y_pred_norm,
        time_axis=time_axis,
        run_ids=np.asarray(run_ids, dtype=np.str_),
        rocks=np.asarray(rocks, dtype=np.str_),
        output_features=np.asarray(output_features, dtype=np.str_),
        eval_split=np.asarray([eval_split], dtype=np.str_),
    )


if __name__ == "__main__":
    main()
