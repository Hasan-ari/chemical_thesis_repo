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
    FullTrajectoryDataset,
    build_bundle,
    build_condition_time_tensor,
    load_cached_bundle,
    write_processed_bundle,
)
from conditional_model_v1.metrics import regression_metrics_original_scale
from conditional_model_v1.models import ConditionTimeLSTM
from conditional_model_v1.plotting import plot_loss_curve, plot_trajectory_examples
from conditional_model_v1.preprocessing import ConditionScaler, OutputScaler, PreprocessorBundle
from conditional_model_v1.splitting import rock_aware_split
from conditional_model_v1.tracking import ExperimentTracker
from conditional_model_v1.training import get_device, predict, set_seed, train_model


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
    condition_scaler = ConditionScaler().fit(bundle.conditions[split.train])
    output_scaler = OutputScaler(log_feature_indices=log_indices).fit(bundle.trajectories[split.train])
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
    trajectories_norm = output_scaler.transform(bundle.trajectories)
    x_all = build_condition_time_tensor(
        conditions_norm,
        bundle.time_axis,
        time_mean=time_mean,
        time_std=time_std,
    )

    train_loader = _make_loader(
        x_all=x_all,
        y_all=trajectories_norm,
        indices=split.train,
        run_ids=bundle.run_ids,
        rocks=bundle.rocks,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    val_loader = (
        _make_loader(
            x_all=x_all,
            y_all=trajectories_norm,
            indices=split.val,
            run_ids=bundle.run_ids,
            rocks=bundle.rocks,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
        )
        if len(split.val) > 0
        else None
    )

    model = ConditionTimeLSTM(
        input_size=x_all.shape[-1],
        output_size=trajectories_norm.shape[-1],
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
    )
    device = get_device()
    print(f"device={device}")
    print(f"x_shape={x_all.shape} y_shape={trajectories_norm.shape}")
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

    eval_indices = split.test if len(split.test) > 0 else split.val if len(split.val) > 0 else split.train
    eval_name = "test" if len(split.test) > 0 else "val" if len(split.val) > 0 else "train"
    model.load_state_dict(torch.load(tracker.checkpoint_dir / "best.pt", map_location=device))
    eval_loader = _make_loader(
        x_all=x_all,
        y_all=trajectories_norm,
        indices=eval_indices,
        run_ids=bundle.run_ids,
        rocks=bundle.rocks,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    y_pred_norm = predict(model, eval_loader, device)
    y_true_norm = trajectories_norm[eval_indices]
    y_pred = output_scaler.inverse_transform(y_pred_norm)
    y_true = output_scaler.inverse_transform(y_true_norm)
    eval_run_ids = [bundle.run_ids[int(index)] for index in eval_indices]
    np.savez_compressed(
        tracker.run_dir / "eval_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        y_true_norm=y_true_norm,
        y_pred_norm=y_pred_norm,
        time_axis=bundle.time_axis,
        run_ids=np.array(eval_run_ids, dtype=object),
        output_features=np.array(bundle.output_features, dtype=object),
        eval_split=np.array([eval_name], dtype=object),
    )
    metrics = regression_metrics_original_scale(
        y_true=y_true,
        y_pred=y_pred,
        output_features=bundle.output_features,
    )
    metrics.update(
        {
            "eval_split": eval_name,
            "n_runs_total": int(bundle.conditions.shape[0]),
            "n_train_runs": int(len(split.train)),
            "n_val_runs": int(len(split.val)),
            "n_test_runs": int(len(split.test)),
            "x_shape": list(x_all.shape),
            "y_shape": list(trajectories_norm.shape),
            "final_train_loss": float(history[-1]["train_loss"]),
            "final_val_loss": float(history[-1]["val_loss"]),
            "best_val_loss": float(min(row["val_loss"] for row in history)),
            "device": str(device),
        }
    )
    tracker.write_metrics(metrics)
    tracker.write_feature_metrics(metrics)
    tracker.record_registry(config, metrics)
    plot_trajectory_examples(
        time_axis=bundle.time_axis,
        y_true=y_true,
        y_pred=y_pred,
        output_features=bundle.output_features,
        run_ids=eval_run_ids,
        output_dir=tracker.plot_dir / "trajectory_examples",
        max_runs=config.plots.max_runs,
        feature_names=None if config.plots.features == "all" else config.plots.features,
    )
    print(f"run_dir={tracker.run_dir}")
    return tracker.run_dir


def _load_or_build_bundle(config: ExperimentConfig):
    """Use NPZ cache when available; otherwise parse raw txt and write processed files."""
    processed_dir = Path(config.data.processed_root) / config.name
    cache_path = processed_dir / "bundle.npz"
    if config.data.use_cache and cache_path.exists() and not config.data.rebuild_cache:
        return load_cached_bundle(cache_path)

    specs = [
        DatasetSpec(
            name=dataset.name,
            rock=dataset.rock,
            path=dataset.path,
            max_runs=dataset.max_runs,
        )
        for dataset in config.data.datasets
    ]
    bundle, inputs, outputs, inventory = build_bundle(
        specs,
        keep_outputs_frame=config.data.write_outputs_csv,
    )
    write_processed_bundle(bundle, inputs, outputs, inventory, processed_dir)
    return bundle


def _make_loader(
    *,
    x_all: np.ndarray,
    y_all: np.ndarray,
    indices: np.ndarray,
    run_ids: list[str],
    rocks: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Build a DataLoader for a run-level split without exposing rock as model input."""
    selected = indices.astype(int)
    dataset = FullTrajectoryDataset(
        x=x_all[selected],
        y=y_all[selected],
        run_ids=[run_ids[index] for index in selected],
        rocks=[str(rocks[index]) for index in selected],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()
