"""
============================================================================
LSTM TRAINING PIPELINE - Chemical Thesis Project (Week 04)
============================================================================
Production-grade LSTM training pipeline for time series forecasting of
two-phase anaerobic model state variables.

Features:
- PyTorch stacked LSTM with configurable architecture (128->64 hidden units)
- Sliding window sequence creation with configurable horizon
- StandardScaler normalization with scaler persistence
- Early stopping with model checkpointing
- Recursive forecasting (chain correction)
- Divergence analysis at steps 50, 100, 150
- Full CLI interface

Author: Chemical Thesis Project
Date: 2026-W04
Framework: PyTorch (v2.x)
============================================================================
"""

import argparse
import json
import logging
import pickle
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure logging with console and optional file output."""
    logger = logging.getLogger("lstm_trainer")
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class TrainingConfig:
    """Configuration for LSTM training pipeline."""

    # Data paths
    data_path: str = "data/output/basalt_25c_lstm_input_500pts.npy"
    output_dir: str = "outputs/lstm_training"

    # LSTM Architecture
    n_features: int = 14
    hidden_1: int = 128
    hidden_2: int = 64
    dropout: float = 0.0

    # Sequence parameters
    seq_len: int = 100
    horizon: int = 10  # Prediction horizon (10, 20, or 30 steps)

    # Train/val/test split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Training parameters
    epochs: int = 500
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 0.0

    # Early stopping
    patience: int = 100
    min_delta: float = 1e-7

    # LR scheduler
    scheduler_factor: float = 0.5
    scheduler_patience: int = 30
    min_lr: float = 1e-7

    # Gradient clipping
    grad_clip_value: float = 0.5

    # Preprocessing - log transform columns (indices)
    log_cols: List[int] = field(default_factory=lambda: [3, 7, 9, 12, 13])

    # Evaluation
    forecast_steps: int = 150
    divergence_checkpoints: List[int] = field(default_factory=lambda: [50, 100, 150])

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "auto"  # "auto", "cuda", "mps", or "cpu"

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**d)

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# ============================================================================
# STATE VARIABLE NAMES
# ============================================================================
STATE_NAMES = [
    "nH2_g", "nCO2_g", "nCH4_g", "nH2S_g",
    "H2_aq", "CO2_aq", "SO4", "FeS",
    "X", "Acetate", "HCO3", "S_tot", "Lag", "Fe_pool"
]


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
class DataProcessor:
    """Handles data loading, preprocessing, and sequence creation."""

    def __init__(self, config: TrainingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.scaler: Optional[StandardScaler] = None
        self.data_raw: Optional[np.ndarray] = None
        self.data_norm: Optional[np.ndarray] = None

    def load_data(self) -> np.ndarray:
        """Load data from .npy file."""
        data_path = Path(self.config.data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.data_raw = np.load(data_path)
        self.logger.info(f"Loaded data: {self.data_raw.shape} from {data_path}")

        # Validate shape
        if self.data_raw.shape[1] != self.config.n_features:
            raise ValueError(
                f"Expected {self.config.n_features} features, "
                f"got {self.data_raw.shape[1]}"
            )

        return self.data_raw

    def preprocess(
        self,
        data: np.ndarray,
        fit_scaler: bool = True
    ) -> np.ndarray:
        """
        Apply log1p transform and Z-score normalization.

        Args:
            data: Raw data array (N, n_features)
            fit_scaler: If True, fit a new scaler; otherwise use existing

        Returns:
            Normalized data array
        """
        data_processed = data.copy()

        # Log1p transform for small-valued columns
        if self.config.log_cols:
            data_processed[:, self.config.log_cols] = np.log1p(
                np.maximum(data[:, self.config.log_cols], 0)
            )

        # Z-score normalization
        if fit_scaler:
            self.scaler = StandardScaler()
            data_norm = self.scaler.fit_transform(data_processed)
            self.logger.info("Fitted new StandardScaler")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit_scaler=True first.")
            data_norm = self.scaler.transform(data_processed)

        self.data_norm = data_norm
        return data_norm

    def inverse_preprocess(self, data_norm: np.ndarray) -> np.ndarray:
        """
        Inverse transform: Z-score -> original scale -> expm1 for log columns.

        Args:
            data_norm: Normalized data array

        Returns:
            Data in original scale
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Cannot inverse transform.")

        data_processed = self.scaler.inverse_transform(data_norm)

        # Inverse log1p transform
        if self.config.log_cols:
            data_processed[:, self.config.log_cols] = np.expm1(
                data_processed[:, self.config.log_cols]
            )

        # Ensure non-negative values
        data_processed = np.maximum(data_processed, 0)

        return data_processed

    def create_sequences(
        self,
        data: np.ndarray,
        seq_len: Optional[int] = None,
        horizon: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training with sliding window.

        ============================================================================
        SLIDING WINDOW SEQUENCE CREATION - Educational Overview
        ============================================================================

        The sliding window approach converts a continuous time series into
        supervised learning (X, Y) pairs suitable for LSTM training.

        CONCEPT:
        --------
        Given a time series: [p₀, p₁, p₂, p₃, ..., pₙ]

        We create input-output pairs where:
          - X (input):  A window of `seq_len` consecutive timesteps
          - Y (target): The value `horizon` steps AFTER the window ends

        VISUAL EXAMPLE (seq_len=100, horizon=10):
        ------------------------------------------

        Time series: [p₀, p₁, p₂, ..., p₉₉, p₁₀₀, ..., p₁₀₉, p₁₁₀, ...]
                      |<--- seq_len=100 --->|     |<-h->|
                      |      X[0]          |           Y[0]=p₁₀₉

        Sequence 0:  X = [p₀,   p₁,  ..., p₉₉ ]  →  Y = p₁₀₉
        Sequence 1:  X = [p₁,   p₂,  ..., p₁₀₀]  →  Y = p₁₁₀
        Sequence 2:  X = [p₂,   p₃,  ..., p₁₀₁]  →  Y = p₁₁₁
        ...
        Sequence i:  X = [pᵢ, pᵢ₊₁, ..., pᵢ₊₉₉]  →  Y = pᵢ₊₁₀₉

        WHY HORIZON > 1?
        ----------------
        - horizon=1 (next-step prediction) is too easy - model learns identity
        - horizon=10 forces model to learn actual dynamics, not just "copy"
        - Larger horizons test if model understands underlying ODE behavior

        NUMBER OF SEQUENCES:
        --------------------
        n_sequences = len(data) - seq_len - horizon + 1

        Example: 1750 points, seq_len=100, horizon=10
                 1750 - 100 - 10 + 1 = 1641 sequences

        OUTPUT SHAPES:
        --------------
        X: (n_sequences, seq_len, n_features)  = (1641, 100, 14)
        Y: (n_sequences, n_features)           = (1641, 14)

        Each X[i] is a 2D array (100 timesteps × 14 features)
        Each Y[i] is a 1D array (14 features) - the target state
        ============================================================================

        Args:
            data: Normalized data array (N, n_features)
            seq_len: Sequence length (default: config.seq_len)
            horizon: Prediction horizon (default: config.horizon)

        Returns:
            X: Input sequences (N_seq, seq_len, n_features)
            Y: Target values (N_seq, n_features)
        """
        seq_len = seq_len or self.config.seq_len
        horizon = horizon or self.config.horizon

        X, Y = [], []

        # Sliding window loop:
        # - Start at i=0, end when we can't fit seq_len + horizon anymore
        # - Each iteration slides the window by 1 timestep
        for i in range(len(data) - seq_len - horizon + 1):
            # X[i] = data[i : i+seq_len]
            # This is the input context: 100 consecutive timesteps
            X.append(data[i:i + seq_len])

            # Y[i] = data[i + seq_len + horizon - 1]
            # This is the target: the state `horizon` steps after X ends
            # Note: We use (horizon - 1) because indices are 0-based
            # If seq_len=100 and horizon=10, target is at index i+109
            Y.append(data[i + seq_len + horizon - 1])

        X = np.array(X)
        Y = np.array(Y)

        self.logger.info(
            f"Created sequences: X={X.shape}, Y={Y.shape} "
            f"(seq_len={seq_len}, horizon={horizon})"
        )

        return X, Y

    def split_data(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/val/test sets.

        Args:
            data: Full data array

        Returns:
            train_data, val_data, test_data
        """
        n = len(data)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        self.logger.info(
            f"Data split: train={len(train_data)}, "
            f"val={len(val_data)}, test={len(test_data)}"
        )

        return train_data, val_data, test_data

    def save_scaler(self, path: Path) -> None:
        """Save fitted scaler for inference."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Cannot save.")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
        self.logger.info(f"Saved scaler to {path}")

    def load_scaler(self, path: Path) -> StandardScaler:
        """Load scaler from file."""
        with open(path, "rb") as f:
            self.scaler = pickle.load(f)
        self.logger.info(f"Loaded scaler from {path}")
        return self.scaler


# ============================================================================
# PYTORCH DATASET
# ============================================================================
class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


# ============================================================================
# LSTM MODEL
# ============================================================================
class StackedLSTM(nn.Module):
    """
    Stacked LSTM model for time series forecasting.

    Architecture: LSTM(hidden_1) -> LSTM(hidden_2) -> Linear(n_features)
    """

    def __init__(
        self,
        input_size: int,
        hidden_1: int,
        hidden_2: int,
        output_size: int,
        dropout: float = 0.0
    ):
        super(StackedLSTM, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_1,
            num_layers=1,
            batch_first=True,
            dropout=0.0  # Dropout only between layers
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lstm2 = nn.LSTM(
            input_size=hidden_1,
            hidden_size=hidden_2,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_2, output_size)

        # Store architecture info for checkpointing
        self.architecture = {
            "input_size": input_size,
            "hidden_1": hidden_1,
            "hidden_2": hidden_2,
            "output_size": output_size,
            "dropout": dropout
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_size)

        Returns:
            Output tensor (batch, output_size)
        """
        # First LSTM layer
        out1, _ = self.lstm1(x)  # (batch, seq_len, hidden_1)
        out1 = self.dropout(out1)

        # Second LSTM layer
        out2, _ = self.lstm2(out1)  # (batch, seq_len, hidden_2)

        # Take only the last time step
        out = self.fc(out2[:, -1, :])  # (batch, output_size)

        return out


def build_model(config: TrainingConfig, device: torch.device) -> StackedLSTM:
    """Build and initialize LSTM model."""
    model = StackedLSTM(
        input_size=config.n_features,
        hidden_1=config.hidden_1,
        hidden_2=config.hidden_2,
        output_size=config.n_features,
        dropout=config.dropout
    )
    return model.to(device)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================
@dataclass
class TrainingHistory:
    """Stores training metrics history."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_mae: List[float] = field(default_factory=list)
    val_mae: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_mae": self.train_mae,
            "val_mae": self.val_mae,
            "learning_rates": self.learning_rates,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss
        }

    def save(self, path: Path) -> None:
        """Save history to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class EarlyStopping:
    """Early stopping handler with patience."""

    def __init__(
        self,
        patience: int = 100,
        min_delta: float = 1e-7,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# ============================================================================
# TRAINER CLASS
# ============================================================================
class LSTMTrainer:
    """Handles LSTM model training with all utilities."""

    def __init__(
        self,
        config: TrainingConfig,
        logger: logging.Logger,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.logger = logger

        # Setup device
        if device is not None:
            self.device = device
        elif config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        self.logger.info(f"Using device: {self.device}")

        # Set random seeds for reproducibility
        self._set_seed(config.seed)

        # Initialize components (will be set during training)
        self.model: Optional[StackedLSTM] = None
        self.optimizer: Optional[torch.optim.Adam] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None
        self.criterion = nn.MSELoss()
        self.history = TrainingHistory()

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.logger.info(f"Set random seed: {seed}")

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        checkpoint_dir: Optional[Path] = None
    ) -> TrainingHistory:
        """
        Train the LSTM model.

        Args:
            X_train: Training input sequences
            Y_train: Training targets
            X_val: Validation input sequences (optional)
            Y_val: Validation targets (optional)
            checkpoint_dir: Directory for saving checkpoints

        Returns:
            Training history
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting LSTM Training")
        self.logger.info("=" * 60)

        # Build model
        self.model = build_model(self.config, self.device)
        self.logger.info(
            f"Model architecture: LSTM({self.config.hidden_1}) -> "
            f"LSTM({self.config.hidden_2}) -> Linear({self.config.n_features})"
        )

        # Create data loaders
        train_dataset = TimeSeriesDataset(X_train, Y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        val_loader = None
        if X_val is not None and Y_val is not None:
            val_dataset = TimeSeriesDataset(X_val, Y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            min_lr=self.config.min_lr
        )

        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )

        # Training state
        best_state_dict = None

        self.logger.info(
            f"Training: {len(X_train)} samples, "
            f"{self.config.epochs} epochs, "
            f"batch_size={self.config.batch_size}"
        )

        # Training loop
        for epoch in range(self.config.epochs):
            # Train epoch
            train_loss, train_mae = self._train_epoch(train_loader)
            self.history.train_loss.append(train_loss)
            self.history.train_mae.append(train_mae)

            # Validation epoch
            if val_loader is not None:
                val_loss, val_mae = self._validate_epoch(val_loader)
                self.history.val_loss.append(val_loss)
                self.history.val_mae.append(val_mae)
            else:
                val_loss = train_loss
                val_mae = train_mae

            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history.learning_rates.append(current_lr)

            # Scheduler step
            self.scheduler.step(val_loss)

            # Track best model
            if val_loss < self.history.best_val_loss:
                self.history.best_val_loss = val_loss
                self.history.best_epoch = epoch
                best_state_dict = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }

            # Logging (every 50 epochs)
            if (epoch + 1) % 50 == 0 or epoch == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"LR: {current_lr:.2e}"
                )

            # Early stopping check
            if early_stopping(val_loss):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Restore best model
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            self.logger.info(
                f"Restored best model from epoch {self.history.best_epoch + 1} "
                f"(val_loss={self.history.best_val_loss:.6f})"
            )

        # Save checkpoint
        if checkpoint_dir is not None:
            self._save_checkpoint(checkpoint_dir)

        self.logger.info("Training complete!")
        return self.history

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0

        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, Y_batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip_value > 0:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(),
                    self.config.grad_clip_value
                )

            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(predictions - Y_batch)).item()
            n_batches += 1

        return total_loss / n_batches, total_mae / n_batches

    def _validate_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0

        with torch.no_grad():
            for X_batch, Y_batch in loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                predictions = self.model(X_batch)
                loss = self.criterion(predictions, Y_batch)

                total_loss += loss.item()
                total_mae += torch.mean(torch.abs(predictions - Y_batch)).item()
                n_batches += 1

        return total_loss / n_batches, total_mae / n_batches

    def _save_checkpoint(self, checkpoint_dir: Path) -> None:
        """Save model checkpoint and training state."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = checkpoint_dir / f"lstm_horizon_{self.config.horizon}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "architecture": self.model.architecture,
            "config": self.config.to_dict(),
            "history": self.history.to_dict(),
            "best_epoch": self.history.best_epoch,
            "best_val_loss": self.history.best_val_loss
        }, model_path)
        self.logger.info(f"Saved checkpoint: {model_path}")

        # Save history separately
        history_path = checkpoint_dir / f"training_history_horizon_{self.config.horizon}.json"
        self.history.save(history_path)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Rebuild model
        arch = checkpoint["architecture"]
        self.model = StackedLSTM(
            input_size=arch["input_size"],
            hidden_1=arch["hidden_1"],
            hidden_2=arch["hidden_2"],
            output_size=arch["output_size"],
            dropout=arch.get("dropout", 0.0)
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.logger.info(f"Loaded checkpoint: {checkpoint_path}")


# ============================================================================
# RECURSIVE FORECASTING (CHAIN CORRECTION)
# ============================================================================
#
# ============================================================================
# RECURSIVE FORECASTING - Educational Overview
# ============================================================================
#
# WHAT IS RECURSIVE FORECASTING?
# ------------------------------
# Unlike training where we have ground truth for every target, at inference
# time we need to predict multiple steps into the future. Recursive forecasting
# (also called "chain correction" or "autoregressive forecasting") feeds the
# model's own predictions back as input to generate longer forecasts.
#
# TRAINING vs INFERENCE:
# ----------------------
#
# TRAINING (Teacher Forcing):
#   Input:  [t₀, t₁, ..., t₉₉]  (ground truth)
#   Target: t₁₀₉                 (ground truth)
#   → Model learns to predict horizon steps ahead given perfect history
#
# INFERENCE (Chain Correction):
#   Step 1: Input [t₀...t₉₉]      → Predict t̂₁₀₉
#   Step 2: Input [t₁...t₉₉, t̂₁₀₉] → Predict t̂₁₁₉
#   Step 3: Input [t₂...t̂₁₀₉, t̂₁₁₉] → Predict t̂₁₂₉
#   ...
#   → Model uses its OWN predictions as future inputs (errors can compound!)
#
# VISUAL DIAGRAM:
# ---------------
#
#   Context Window (100 steps)        Prediction
#   ┌─────────────────────────┐          │
#   │ t₀  t₁  t₂  ...  t₉₉   │ ──LSTM──▶ t̂₁₀₉
#   └─────────────────────────┘
#              │
#              ▼ slide window, add prediction
#   ┌─────────────────────────┐          │
#   │ t₁  t₂  t₃  ...  t̂₁₀₉  │ ──LSTM──▶ t̂₁₁₉
#   └─────────────────────────┘
#              │
#              ▼ repeat...
#   ┌─────────────────────────┐          │
#   │ t₂  t₃  t₄  ...  t̂₁₁₉  │ ──LSTM──▶ t̂₁₂₉
#   └─────────────────────────┘
#
# WHY THIS MATTERS:
# -----------------
# 1. Tests if model learned TRUE dynamics (not just memorization)
# 2. Errors can compound - small mistakes grow over time
# 3. Divergence analysis checks if predictions "blow up" at steps 50, 100, 150
# 4. A stable model maintains reasonable errors even after 150 recursive steps
#
# CHAIN vs DENSE FORECASTING:
# ---------------------------
# - forecast():       Sparse output at horizon intervals (every 10 steps)
# - forecast_dense(): Interpolates between sparse predictions for smooth output
#
# ============================================================================

class RecursiveForecaster:
    """
    Performs recursive (autoregressive) forecasting.

    This class implements "chain correction" where the model's predictions
    are fed back as input to generate multi-step forecasts beyond the
    training horizon.
    """

    def __init__(
        self,
        model: StackedLSTM,
        device: torch.device,
        config: TrainingConfig,
        logger: logging.Logger
    ):
        self.model = model
        self.device = device
        self.config = config
        self.logger = logger

    def forecast(
        self,
        initial_context: np.ndarray,
        n_steps: int,
        horizon: Optional[int] = None
    ) -> np.ndarray:
        """
        Perform recursive forecast (chain correction).

        The model predicts 'horizon' steps ahead, then feeds the prediction
        back as input to continue the chain.

        ALGORITHM:
        ----------
        1. Start with initial context window [t₀, t₁, ..., t₉₉]
        2. Predict t̂₁₀₉ (horizon=10 steps ahead)
        3. Slide window: remove t₀, append t̂₁₀₉ → [t₁, t₂, ..., t̂₁₀₉]
        4. Predict t̂₁₁₉
        5. Repeat until we have n_steps predictions

        Args:
            initial_context: Starting context (seq_len, n_features) - normalized
            n_steps: Total number of steps to forecast
            horizon: Steps model was trained to predict (default: config.horizon)

        Returns:
            predictions: (n_steps, n_features) - normalized
        """
        horizon = horizon or self.config.horizon
        self.model.eval()

        # Copy context to avoid modifying original
        context = initial_context.copy()
        predictions = []

        # Calculate how many chain iterations we need
        # Example: n_steps=150, horizon=10 → n_chains=15
        n_chains = (n_steps + horizon - 1) // horizon

        with torch.no_grad():
            for _ in range(n_chains):
                # Step 1: Convert numpy context to PyTorch tensor
                # Shape: (1, seq_len, features) - batch size of 1
                context_tensor = torch.FloatTensor(context).unsqueeze(0).to(self.device)

                # Step 2: Forward pass through LSTM
                # Model outputs prediction for `horizon` steps ahead
                pred = self.model(context_tensor)
                pred = pred.cpu().numpy()[0]  # Back to numpy, remove batch dim

                # Step 3: Store this prediction
                predictions.append(pred)

                # Step 4: CHAIN CORRECTION - Update context for next iteration
                # Remove oldest timestep (context[0]), append our prediction
                # context[1:] = [t₁, t₂, ..., t₉₉]  (99 steps)
                # vstack adds pred → [t₁, t₂, ..., t₉₉, t̂₁₀₉]  (100 steps again)
                context = np.vstack([context[1:], pred])

        predictions = np.array(predictions)

        # Truncate to requested steps (we might have generated extras)
        if len(predictions) > n_steps:
            predictions = predictions[:n_steps]

        return predictions

    def forecast_dense(
        self,
        initial_context: np.ndarray,
        n_steps: int,
        horizon: Optional[int] = None
    ) -> np.ndarray:
        """
        Dense recursive forecast with interpolation for intermediate steps.

        SPARSE vs DENSE FORECASTING:
        ----------------------------
        The model predicts every `horizon` steps (e.g., every 10 steps).
        This gives us SPARSE predictions at times [0, 10, 20, 30, ...].

        For visualization and evaluation, we often want DENSE predictions
        at every timestep [0, 1, 2, 3, ...]. This method interpolates
        between sparse predictions to fill in the gaps.

        VISUAL EXAMPLE (horizon=10, n_steps=30):
        ----------------------------------------

        Sparse predictions from model:
          Time:  0    10    20    30
          Pred:  p₀   p₁₀   p₂₀   p₃₀
                 •────•────•────•

        Dense output after interpolation:
          Time:  0  1  2  3  4  5  6  7  8  9 10 11 12 ... 30
          Pred:  p₀ ·  ·  ·  ·  ·  ·  ·  ·  · p₁₀ ·  ·  ... p₃₀
                 •──────────────────────•──────────...────•
                       linear interp

        WHY INTERPOLATION?
        ------------------
        1. Smooth visualization curves
        2. Point-by-point comparison with ground truth
        3. RMSE calculation at every timestep

        Args:
            initial_context: Starting context (seq_len, n_features) - normalized
            n_steps: Total number of steps to forecast
            horizon: Steps model was trained to predict

        Returns:
            predictions: (n_steps, n_features) - normalized, dense
        """
        horizon = horizon or self.config.horizon
        self.model.eval()

        context = initial_context.copy()

        # Start with the last point of context as our t=0 reference
        sparse_preds = [context[-1]]

        n_chains = (n_steps + horizon - 1) // horizon

        with torch.no_grad():
            for _ in range(n_chains):
                context_tensor = torch.FloatTensor(context).unsqueeze(0).to(self.device)
                pred = self.model(context_tensor).cpu().numpy()[0]
                sparse_preds.append(pred)

                # Shift context by horizon steps (not by 1 like in forecast())
                # This is more aggressive context updating for dense mode
                if horizon < self.config.seq_len:
                    # Tile the prediction to fill `horizon` slots
                    new_rows = np.tile(pred, (horizon, 1))
                    context = np.vstack([context[horizon:], new_rows])
                else:
                    # If horizon >= seq_len, fill entire context with prediction
                    context = np.tile(pred, (self.config.seq_len, 1))

        # ================================================================
        # LINEAR INTERPOLATION: Fill gaps between sparse predictions
        # ================================================================
        sparse_preds = np.array(sparse_preds)

        # Sparse times: [0, 10, 20, 30, ...] (at horizon intervals)
        sparse_times = np.arange(len(sparse_preds)) * horizon

        # Dense times: [0, 1, 2, 3, ..., n_steps-1] (every timestep)
        dense_times = np.arange(n_steps)

        # Interpolate each feature independently
        dense_preds = np.zeros((n_steps, self.config.n_features))
        for feat in range(self.config.n_features):
            # np.interp does linear interpolation
            # For time=5 with sparse points at 0 and 10, it computes:
            #   pred[5] = pred[0] + (5/10) * (pred[10] - pred[0])
            dense_preds[:, feat] = np.interp(
                dense_times,
                sparse_times[:len(sparse_preds)],
                sparse_preds[:, feat]
            )

        return dense_preds

    def forecast_from_point(
        self,
        data_norm: np.ndarray,
        start_idx: int,
        n_steps: int,
        dense: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast starting from a specific point in the dataset.

        Args:
            data_norm: Full normalized dataset
            start_idx: Index to start forecasting from
            n_steps: Number of steps to forecast
            dense: If True, use dense interpolated forecast

        Returns:
            predictions: (n_steps, n_features)
            ground_truth: (n_steps, n_features)
        """
        # Get context window
        context_start = start_idx - self.config.seq_len
        if context_start < 0:
            raise ValueError(
                f"start_idx must be >= seq_len ({self.config.seq_len})"
            )

        initial_context = data_norm[context_start:start_idx]

        # Get ground truth
        end_idx = min(start_idx + n_steps, len(data_norm))
        ground_truth = data_norm[start_idx:end_idx]

        # Forecast
        if dense:
            predictions = self.forecast_dense(initial_context, len(ground_truth))
        else:
            predictions = self.forecast(initial_context, len(ground_truth))

        return predictions, ground_truth


# ============================================================================
# EVALUATION
# ============================================================================
class ModelEvaluator:
    """Handles model evaluation and divergence analysis."""

    def __init__(
        self,
        config: TrainingConfig,
        data_processor: DataProcessor,
        logger: logging.Logger
    ):
        self.config = config
        self.data_processor = data_processor
        self.logger = logger

    def compute_rmse(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute RMSE for each state variable.

        Args:
            predictions: Predicted values (N, n_features)
            ground_truth: True values (N, n_features)

        Returns:
            Dictionary mapping state names to RMSE values
        """
        rmse_dict = {}
        for i, name in enumerate(STATE_NAMES):
            rmse = np.sqrt(np.mean((predictions[:, i] - ground_truth[:, i]) ** 2))
            rmse_dict[name] = float(rmse)

        # Overall RMSE
        rmse_dict["overall"] = float(np.sqrt(np.mean((predictions - ground_truth) ** 2)))

        return rmse_dict

    def analyze_divergence(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        checkpoints: Optional[List[int]] = None
    ) -> Dict:
        """
        Analyze prediction divergence at specified checkpoints.

        Args:
            predictions: Predicted values (N, n_features)
            ground_truth: True values (N, n_features)
            checkpoints: Steps to analyze (default: [50, 100, 150])

        Returns:
            Analysis report dictionary
        """
        checkpoints = checkpoints or self.config.divergence_checkpoints
        key_vars = [0, 3, 6]  # nH2_g, nH2S_g, SO4

        report = {
            "checkpoints": {},
            "stability": {}
        }

        # Error at each checkpoint
        for cp in checkpoints:
            if cp > len(predictions):
                continue

            errors = {}
            for var_idx in key_vars:
                err = abs(predictions[cp-1, var_idx] - ground_truth[cp-1, var_idx])
                errors[STATE_NAMES[var_idx]] = float(err)

            report["checkpoints"][cp] = errors

        # Stability check: error ratio between first and last checkpoint
        valid_cps = [cp for cp in checkpoints if cp <= len(predictions)]
        if len(valid_cps) >= 2:
            first_cp, last_cp = valid_cps[0], valid_cps[-1]

            for var_idx in key_vars:
                var_name = STATE_NAMES[var_idx]
                err_first = abs(predictions[first_cp-1, var_idx] - ground_truth[first_cp-1, var_idx])
                err_last = abs(predictions[last_cp-1, var_idx] - ground_truth[last_cp-1, var_idx])

                ratio = err_last / (err_first + 1e-10)
                status = "DIVERGING" if ratio > 10 else "STABLE"

                report["stability"][var_name] = {
                    "error_ratio": float(ratio),
                    "status": status
                }

        return report

    def generate_report(
        self,
        predictions_orig: np.ndarray,
        ground_truth_orig: np.ndarray,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            predictions_orig: Predictions in original scale
            ground_truth_orig: Ground truth in original scale
            output_path: Path to save report (optional)

        Returns:
            Report string
        """
        # Compute metrics
        rmse = self.compute_rmse(predictions_orig, ground_truth_orig)
        divergence = self.analyze_divergence(predictions_orig, ground_truth_orig)

        # Build report
        lines = [
            "=" * 70,
            "LSTM EVALUATION REPORT",
            f"Horizon: {self.config.horizon} steps",
            f"Forecast length: {len(predictions_orig)} steps",
            "=" * 70,
            "",
            "--- RMSE PER STATE VARIABLE ---"
        ]

        for name in STATE_NAMES:
            lines.append(f"  {name:12s}: {rmse[name]:.6f}")
        lines.append(f"  {'Overall':12s}: {rmse['overall']:.6f}")

        lines.append("")
        lines.append("--- DIVERGENCE ANALYSIS ---")

        for cp, errors in divergence["checkpoints"].items():
            lines.append(f"\nStep {cp}:")
            for var_name, err in errors.items():
                lines.append(f"  {var_name}: Error = {err:.6f}")

        lines.append("")
        lines.append("--- STABILITY CHECK ---")

        for var_name, info in divergence["stability"].items():
            lines.append(
                f"  {var_name}: Ratio = {info['error_ratio']:.2f} -> {info['status']}"
            )

        lines.append("")
        lines.append("=" * 70)

        report = "\n".join(lines)

        # Save if path provided
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            self.logger.info(f"Saved report: {output_path}")

        return report


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
def run_training_pipeline(config: TrainingConfig) -> Dict:
    """
    Execute the full training pipeline.

    Args:
        config: Training configuration

    Returns:
        Dictionary with trained model, history, and evaluation results
    """
    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"training_{timestamp}.log"
    logger = setup_logging(log_file)

    logger.info("=" * 70)
    logger.info("LSTM TRAINING PIPELINE - Chemical Thesis Project")
    logger.info("=" * 70)
    logger.info(f"Config: {config.to_dict()}")

    # Save config
    config.save(output_dir / "config.json")

    # Initialize data processor
    data_processor = DataProcessor(config, logger)

    # Load and preprocess data
    data_raw = data_processor.load_data()

    # Split data BEFORE preprocessing (to avoid data leakage)
    train_raw, val_raw, test_raw = data_processor.split_data(data_raw)

    # Preprocess - fit scaler on training data only
    train_norm = data_processor.preprocess(train_raw, fit_scaler=True)
    val_norm = data_processor.preprocess(val_raw, fit_scaler=False)
    test_norm = data_processor.preprocess(test_raw, fit_scaler=False)

    # Also preprocess full data for evaluation
    full_norm = data_processor.preprocess(data_raw, fit_scaler=False)

    # Save scaler
    data_processor.save_scaler(output_dir / "scaler.pkl")

    # Create sequences
    X_train, Y_train = data_processor.create_sequences(train_norm)
    X_val, Y_val = data_processor.create_sequences(val_norm)

    # Initialize trainer
    trainer = LSTMTrainer(config, logger)

    # Train model
    history = trainer.train(
        X_train, Y_train,
        X_val, Y_val,
        checkpoint_dir=output_dir / "checkpoints"
    )

    # Initialize forecaster
    forecaster = RecursiveForecaster(trainer.model, trainer.device, config, logger)

    # Evaluate on test set
    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluation on Test Set")
    logger.info("=" * 60)

    # Find valid start point in test region
    n_train = len(train_raw)
    n_val = len(val_raw)
    test_start_idx = n_train + n_val + config.seq_len

    # Random test point
    np.random.seed(config.seed)
    max_start = len(data_raw) - config.forecast_steps
    if test_start_idx < max_start:
        start_idx = np.random.randint(test_start_idx, max_start)
    else:
        start_idx = test_start_idx

    logger.info(f"Test forecast starting at index: {start_idx}")

    # Forecast
    predictions_norm, ground_truth_norm = forecaster.forecast_from_point(
        full_norm, start_idx, config.forecast_steps, dense=True
    )

    # Inverse transform
    predictions_orig = data_processor.inverse_preprocess(predictions_norm)
    ground_truth_orig = data_processor.inverse_preprocess(ground_truth_norm)

    # Evaluate
    evaluator = ModelEvaluator(config, data_processor, logger)
    report = evaluator.generate_report(
        predictions_orig,
        ground_truth_orig,
        output_dir / "evaluation_report.txt"
    )
    logger.info("\n" + report)

    # Save predictions
    np.savez(
        output_dir / "predictions.npz",
        predictions_norm=predictions_norm,
        ground_truth_norm=ground_truth_norm,
        predictions_orig=predictions_orig,
        ground_truth_orig=ground_truth_orig,
        start_idx=start_idx
    )
    logger.info(f"Saved predictions to {output_dir / 'predictions.npz'}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info("=" * 70)

    return {
        "model": trainer.model,
        "history": history,
        "data_processor": data_processor,
        "evaluator": evaluator,
        "config": config
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================
def parse_args() -> TrainingConfig:
    """Parse command line arguments and return config."""
    parser = argparse.ArgumentParser(
        description="LSTM Training Pipeline for Chemical Thesis Project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        "--data_path", type=str,
        default="data/output/basalt_25c_lstm_input_500pts.npy",
        help="Path to input .npy data file"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="outputs/lstm_training",
        help="Output directory for checkpoints and logs"
    )

    # Model architecture
    parser.add_argument(
        "--hidden_1", type=int, default=128,
        help="Hidden size for first LSTM layer"
    )
    parser.add_argument(
        "--hidden_2", type=int, default=64,
        help="Hidden size for second LSTM layer"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0,
        help="Dropout rate between LSTM layers"
    )

    # Sequence parameters
    parser.add_argument(
        "--seq_len", type=int, default=100,
        help="Input sequence length"
    )
    parser.add_argument(
        "--horizon", type=int, default=10,
        choices=[10, 20, 30],
        help="Prediction horizon (steps ahead)"
    )

    # Data splits
    parser.add_argument(
        "--train_ratio", type=float, default=0.70,
        help="Training data ratio"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15,
        help="Validation data ratio"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=500,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--patience", type=int, default=100,
        help="Early stopping patience"
    )

    # Evaluation
    parser.add_argument(
        "--forecast_steps", type=int, default=150,
        help="Number of steps to forecast for evaluation"
    )

    # Other
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training"
    )

    args = parser.parse_args()

    # Create config from args
    config = TrainingConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        hidden_1=args.hidden_1,
        hidden_2=args.hidden_2,
        dropout=args.dropout,
        seq_len=args.seq_len,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        forecast_steps=args.forecast_steps,
        seed=args.seed,
        device=args.device
    )

    return config


def main():
    """Main entry point for CLI."""
    config = parse_args()
    run_training_pipeline(config)


if __name__ == "__main__":
    main()
