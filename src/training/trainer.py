from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    """Result of a single training run."""
    final_loss: float
    best_loss: float
    loss_history: List[float]
    training_time_seconds: float
    total_epochs: int


def setup_device() -> torch.device:
    """Detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("Device: CPU")
    return device


class Trainer:
    """Training loop for one experiment."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        self.criterion = nn.MSELoss()

    def train(
        self,
        epochs: int,
        print_every: int = 10,
        save_dir: Optional[Path] = None,
    ) -> TrainResult:
        """Run training loop."""
        loss_history = []
        best_loss = float("inf")
        start_time = time.time()

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, Y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                output = self.model(X_batch)
                loss = self.criterion(output, Y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)
            self.scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                if save_dir:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), save_dir / "best_model.pt")

            if epoch % print_every == 0 or epoch == epochs - 1:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch:4d}/{epochs} | "
                    f"Loss: {avg_loss:.2e} | Best: {best_loss:.2e} | LR: {lr:.1e}"
                )

        training_time = time.time() - start_time

        # Save final model
        if save_dir:
            torch.save(self.model.state_dict(), save_dir / "final_model.pt")

        return TrainResult(
            final_loss=loss_history[-1],
            best_loss=best_loss,
            loss_history=loss_history,
            training_time_seconds=training_time,
            total_epochs=epochs,
        )
