from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from conditional_model_v1.config import SchedulerConfig, TrainingConfig


def set_seed(seed: int) -> None:
    """Set deterministic seeds for repeatable experiment comparisons."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Prefer Colab CUDA GPU, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    *,
    epochs: int,
) -> Any:
    """Create a PyTorch scheduler from config."""
    if config.type == "none":
        return None
    if config.type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.factor,
            patience=config.patience,
            min_lr=config.min_lr,
        )
    if config.type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if config.type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
    raise ValueError(f"Unsupported scheduler type: {config.type}")


def train_model(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: TrainingConfig,
    checkpoint_dir: Path,
    device: torch.device,
) -> list[dict[str, float | int]]:
    """Train with normalized global MSE over batch, time, and feature dimensions."""
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config.scheduler, epochs=config.epochs)
    # MSELoss default reduction="mean": averages over every element in
    # (batch, time, feature), matching our normalized global MSE decision.
    loss_fn = nn.MSELoss()
    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            grad_clip=config.grad_clip,
        )
        val_loss = (
            _run_epoch(model=model, loader=val_loader, loss_fn=loss_fn, device=device)
            if val_loader is not None
            else train_loss
        )
        lr = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})

        if scheduler is not None:
            if config.scheduler.type == "reduce_on_plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")

        print(
            f"epoch={epoch:04d} train={train_loss:.6e} "
            f"val={val_loss:.6e} lr={lr:.3e}",
            flush=True,
        )

    torch.save(model.state_dict(), checkpoint_dir / "final.pt")
    return history


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Return normalized predictions for every batch in loader order."""
    model.eval()
    batches: list[np.ndarray] = []
    for x_batch, _y_batch in loader:
        output = model(x_batch.to(device)).detach().cpu().numpy()
        batches.append(output)
    return np.concatenate(batches, axis=0)


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader | None,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
) -> float:
    """Run one train/eval pass and return mean loss weighted by batch size."""
    if loader is None:
        return float("nan")
    total_loss = 0.0
    total_samples = 0
    is_train = optimizer is not None
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            prediction = model(x_batch)
            loss = loss_fn(prediction, y_batch)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            batch_size = x_batch.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)
