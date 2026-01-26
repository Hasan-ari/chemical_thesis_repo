"""
============================================================================
SEQUENCE WINDOWED LSTM - Context Window Prediction
============================================================================

This module implements the SEQUENCE WINDOWED approach where the LSTM
receives a window of past timesteps as context to predict the next step.

============================================================================
COMPARISON: NEXT-STEP vs SEQUENCE WINDOWED
============================================================================

NEXT-STEP (lstm_next_step.py):
------------------------------
    Input:  y[t]                    (single point, 14 features)
    Output: y[t+1]                  (next point)
    Memory: LSTM hidden state only

    Training pairs from 500 points:
        y[0] → y[1]
        y[1] → y[2]
        ...
        y[498] → y[499]
        Total: 499 pairs

SEQUENCE WINDOWED (this file):
------------------------------
    Input:  [y[t-W+1], ..., y[t]]   (W points as context window)
    Output: y[t+1]                  (next point)
    Memory: Explicit context + LSTM hidden state

    Training pairs from 500 points (with window W=50):
        [y[0]...y[49]]   → y[50]
        [y[1]...y[50]]   → y[51]
        ...
        [y[449]...y[498]] → y[499]
        Total: 450 pairs (fewer than next-step!)

============================================================================
WHY SEQUENCE WINDOW HELPS
============================================================================

The window provides EXPLICIT CONTEXT about recent trajectory:

    Next-step input:     y[t] = [9.07, 2.46, 0.0, ...]  (just current state)
                         ↓
                         LSTM must "remember" where it is via hidden state

    Window input:        [y[t-49], y[t-48], ..., y[t]]  (50 recent states)
                         ↓
                         LSTM can SEE the recent trend directly

ADVANTAGE:
----------
- Model sees explicit trajectory history
- Easier to learn "what comes next" from visible pattern
- Less reliance on hidden state memory

DISADVANTAGE:
-------------
- Fewer training samples (500 - window_size)
- Cannot generate from y[0] alone (needs initial window)
- Window size is a hyperparameter to tune

============================================================================
WINDOW SIZE CHOICE: 50 STEPS
============================================================================

With 500 data points over 19 days:
    - 1 step = 0.038 days ≈ 55 minutes
    - 50 steps = 1.9 days of context
    - Covers enough time to see reaction trends

Training samples: 500 - 50 = 450 pairs

============================================================================
GENERATION FROM INITIAL WINDOW
============================================================================

Unlike next-step which starts from y[0] alone, here we need an initial
window of 50 ground truth points, then generate the rest:

    Given:     [y[0], y[1], ..., y[49]]     (first 50 points from ODE)
    Generate:  ŷ[50], ŷ[51], ..., ŷ[499]   (remaining 450 points)

    Step 1: [y[0]...y[49]]   → ŷ[50]
    Step 2: [y[1]...y[49], ŷ[50]] → ŷ[51]
    Step 3: [y[2]...ŷ[50], ŷ[51]] → ŷ[52]
    ...

    After 50 steps, the window contains ONLY predictions (no ground truth).

============================================================================
Author: Chemical Thesis Project
Date: 2026-W04
Framework: PyTorch
============================================================================
"""

import json
import logging
import pickle
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# State variable names (14 features)
STATE_NAMES = [
    "nH2_g", "nCO2_g", "nCH4_g", "nH2S_g",
    "H2_aq", "CO2_aq", "SO4", "FeS",
    "X", "Acetate", "HCO3", "S_tot", "Lag", "Fe_pool"
]


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger("lstm_seq_window")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(handler.formatter)
        logger.addHandler(fh)

    return logger


# ============================================================================
# CONFIGURATION (Same as v2 next-step for fair comparison)
# ============================================================================
@dataclass
class SeqWindowConfig:
    """Configuration for sequence windowed LSTM training."""

    # Data
    data_path: str = "data/output/basalt_25c_lstm_input_500pts.npy"
    output_dir: str = "outputs/lstm_seq_window"

    # Model architecture (SAME as v2 next-step)
    n_features: int = 14
    hidden_size: int = 128     # Same as v2
    num_layers: int = 2        # Same as v2

    # Sequence window parameters
    window_size: int = 50      # Context window (50 steps ≈ 1.9 days)

    # Training (SAME as v2 next-step for fair comparison)
    epochs: int = 10000        # Same as v2
    batch_size: int = 450      # Full batch (all samples)
    learning_rate: float = 5e-4  # Same as v2
    target_loss: float = 1e-8  # Same as v2

    # Preprocessing
    use_log_transform: bool = True
    log_cols: tuple = (3, 7, 9, 12, 13)

    # Device
    device: str = "auto"
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# DATA PROCESSING
# ============================================================================
class SeqWindowDataProcessor:
    """Prepares data for sequence windowed prediction."""

    def __init__(self, config: SeqWindowConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.scaler = None

    def load_data(self) -> np.ndarray:
        """Load raw data from .npy file."""
        data = np.load(self.config.data_path)
        self.logger.info(f"Loaded data: {data.shape} from {self.config.data_path}")
        return data

    def preprocess(self, data: np.ndarray, fit_scaler: bool = True) -> np.ndarray:
        """Preprocess: log transform + standardization."""
        data = data.copy()

        if self.config.use_log_transform:
            for col in self.config.log_cols:
                data[:, col] = np.log1p(np.maximum(data[:, col], 0))

        if fit_scaler:
            self.scaler = StandardScaler()
            data_norm = self.scaler.fit_transform(data)
            self.logger.info("Fitted new StandardScaler")
        else:
            data_norm = self.scaler.transform(data)

        return data_norm

    def inverse_preprocess(self, data_norm: np.ndarray) -> np.ndarray:
        """Convert normalized data back to original scale."""
        data = self.scaler.inverse_transform(data_norm)

        if self.config.use_log_transform:
            for col in self.config.log_cols:
                data[:, col] = np.expm1(data[:, col])

        return data

    def create_windowed_pairs(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output pairs with sliding window.

        ============================================================
        SLIDING WINDOW PAIR CREATION
        ============================================================

        For data [y[0], y[1], ..., y[N-1]] with window size W:

            X[0] = [y[0], y[1], ..., y[W-1]]     → Y[0] = y[W]
            X[1] = [y[1], y[2], ..., y[W]]       → Y[1] = y[W+1]
            X[2] = [y[2], y[3], ..., y[W+1]]     → Y[2] = y[W+2]
            ...

        Result: N - W training pairs

        Example with N=500, W=50:
            X[0] = [y[0]...y[49]]   → Y[0] = y[50]
            X[1] = [y[1]...y[50]]   → Y[1] = y[51]
            ...
            X[449] = [y[449]...y[498]] → Y[449] = y[499]

            Total: 450 pairs

        Shape:
            X: (N-W, W, 14) = (450, 50, 14)
            Y: (N-W, 14)    = (450, 14)
        ============================================================
        """
        W = self.config.window_size
        N = len(data)

        X, Y = [], []
        for i in range(N - W):
            # Input: window of W consecutive points
            X.append(data[i:i + W])
            # Target: the point right after the window
            Y.append(data[i + W])

        X = np.array(X)
        Y = np.array(Y)

        self.logger.info(
            f"Created windowed pairs: X={X.shape}, Y={Y.shape} "
            f"(window_size={W})"
        )

        return X, Y

    def save_scaler(self, path: Path):
        """Save scaler for later use."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        self.logger.info(f"Saved scaler to {path}")


# ============================================================================
# MODEL (Same architecture as v2 next-step)
# ============================================================================
class SeqWindowLSTM(nn.Module):
    """
    LSTM for sequence windowed prediction.

    ============================================================
    ARCHITECTURE (Same as v2 next-step for fair comparison)
    ============================================================

    Input:  [y[t-W+1], ..., y[t]]   (W, 14) = (50, 14)
              ↓
    LSTM Layer 1: hidden_size=128
              ↓
    LSTM Layer 2: hidden_size=128
              ↓
    Take last hidden output (captures full window context)
              ↓
    Linear: 128 → 14
              ↓
    Output: ŷ[t+1]  (14,)

    The LSTM processes the entire window sequentially, building up
    context in its hidden state. The final hidden state summarizes
    the window, which is then projected to predict the next step.
    ============================================================
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 14
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, window_size, features)

        Returns:
            output: Predicted next state (batch, features)
        """
        # LSTM processes entire window
        # lstm_out: (batch, window_size, hidden_size)
        # hidden: tuple of (h_n, c_n)
        lstm_out, _ = self.lstm(x)

        # Take ONLY the last timestep output (after seeing full window)
        # last_out: (batch, hidden_size)
        last_out = lstm_out[:, -1, :]

        # Project to output
        output = self.fc(last_out)

        return output


# ============================================================================
# TRAINER
# ============================================================================
class SeqWindowTrainer:
    """Trains the sequence windowed LSTM."""

    def __init__(self, config: SeqWindowConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Device setup
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        self.logger.info(f"Using device: {self.device}")

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.model = SeqWindowLSTM(
            input_size=config.n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=config.n_features
        ).to(self.device)

        self.logger.info(
            f"Model: LSTM({config.hidden_size}) x {config.num_layers} → Linear({config.n_features})"
        )

    def train(self, X: np.ndarray, Y: np.ndarray, checkpoint_dir: Path) -> Dict:
        """Train the model to overfit."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        X_tensor = torch.FloatTensor(X).to(self.device)
        Y_tensor = torch.FloatTensor(Y).to(self.device)

        self.logger.info(f"Training data: X={X_tensor.shape}, Y={Y_tensor.shape}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        history = {"loss": [], "epoch": []}

        self.logger.info("=" * 60)
        self.logger.info("Starting Training (Sequence Windowed, Overfitting Mode)")
        self.logger.info(f"Window size: {self.config.window_size}")
        self.logger.info(f"Target loss: {self.config.target_loss}")
        self.logger.info("=" * 60)

        for epoch in range(self.config.epochs):
            self.model.train()

            output = self.model(X_tensor)
            loss = criterion(output, Y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            history["loss"].append(loss_val)
            history["epoch"].append(epoch)

            if epoch % 100 == 0 or epoch == self.config.epochs - 1:
                self.logger.info(f"Epoch {epoch:5d}/{self.config.epochs} - Loss: {loss_val:.2e}")

            if loss_val < self.config.target_loss:
                self.logger.info(f"Target loss {self.config.target_loss} reached at epoch {epoch}!")
                break

        checkpoint_path = checkpoint_dir / "lstm_seq_window.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
            "final_loss": loss_val,
            "epochs_trained": epoch + 1
        }, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f)

        return history


# ============================================================================
# TRAJECTORY GENERATOR
# ============================================================================
class SeqWindowGenerator:
    """
    Generates trajectory from initial window.

    ============================================================
    GENERATION PROCESS
    ============================================================

    Unlike next-step which needs only y[0], sequence windowed needs
    an initial window of W ground truth points.

    Given: Initial window [y[0], y[1], ..., y[W-1]] from ground truth

    Generation:
        Step 1: window = [y[0]...y[W-1]]
                ŷ[W] = model(window)

        Step 2: window = [y[1]...y[W-1], ŷ[W]]  (slide and append)
                ŷ[W+1] = model(window)

        Step 3: window = [y[2]...ŷ[W], ŷ[W+1]]
                ŷ[W+2] = model(window)

        ...continue until all points generated...

    After W steps of generation, the window contains ONLY predictions.
    From that point on, it's fully autoregressive (like next-step).

    COMPARISON TO NEXT-STEP:
    ------------------------
    - Next-step: 499 autoregressive steps from y[0]
    - Seq window (W=50): 450 autoregressive steps from [y[0]...y[49]]

    The window approach has fewer autoregressive steps AND starts with
    more context, so errors should compound less severely.
    ============================================================
    """

    def __init__(
        self,
        model: SeqWindowLSTM,
        device: torch.device,
        config: SeqWindowConfig,
        logger: logging.Logger
    ):
        self.model = model
        self.device = device
        self.config = config
        self.logger = logger

    def generate(
        self,
        initial_window: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """
        Generate trajectory from initial window.

        Args:
            initial_window: First W points (W, 14) - normalized
            n_steps: Total steps to generate (including initial window)

        Returns:
            trajectory: Full trajectory (n_steps, 14) - normalized
        """
        self.model.eval()
        W = self.config.window_size

        # Initialize trajectory with ground truth window
        trajectory = np.zeros((n_steps, self.config.n_features))
        trajectory[:W] = initial_window

        # Current window (will be updated)
        window = initial_window.copy()

        with torch.no_grad():
            for t in range(W, n_steps):
                # Convert window to tensor: (1, W, 14)
                window_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)

                # Predict next step
                pred = self.model(window_tensor)
                next_state = pred.squeeze().cpu().numpy()

                # Store prediction
                trajectory[t] = next_state

                # Slide window: remove oldest, append prediction
                window = np.vstack([window[1:], next_state])

        self.logger.info(f"Generated trajectory: {trajectory.shape}")
        self.logger.info(f"  - Ground truth window: steps 0-{W-1}")
        self.logger.info(f"  - Generated: steps {W}-{n_steps-1} ({n_steps-W} steps)")

        return trajectory


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_seq_window_pipeline(config: SeqWindowConfig) -> Dict:
    """Execute the sequence windowed training and generation pipeline."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(output_dir / f"training_{timestamp}.log")

    logger.info("=" * 60)
    logger.info("SEQUENCE WINDOWED LSTM PIPELINE")
    logger.info(f"Window size: {config.window_size} steps")
    logger.info("=" * 60)

    config.save(output_dir / "config.json")

    # Data processing
    processor = SeqWindowDataProcessor(config, logger)
    data_raw = processor.load_data()
    data_norm = processor.preprocess(data_raw, fit_scaler=True)
    processor.save_scaler(output_dir / "scaler.pkl")

    # Create training pairs
    X, Y = processor.create_windowed_pairs(data_norm)

    # Train
    trainer = SeqWindowTrainer(config, logger)
    history = trainer.train(X, Y, output_dir / "checkpoints")

    # Generate trajectory
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAJECTORY GENERATION TEST")
    logger.info("=" * 60)

    generator = SeqWindowGenerator(trainer.model, trainer.device, config, logger)

    # Use first W points as initial window
    initial_window = data_norm[:config.window_size]
    n_steps = len(data_raw)

    trajectory_norm = generator.generate(initial_window, n_steps)

    # Convert back to original scale
    trajectory_orig = processor.inverse_preprocess(trajectory_norm)
    ground_truth_orig = data_raw

    # Save results
    np.savez(
        output_dir / "trajectory.npz",
        trajectory_norm=trajectory_norm,
        trajectory_orig=trajectory_orig,
        ground_truth_norm=data_norm,
        ground_truth_orig=ground_truth_orig,
        window_size=config.window_size
    )
    logger.info(f"Saved trajectory to {output_dir / 'trajectory.npz'}")

    # Compute RMSE (only for generated part, not the initial window)
    W = config.window_size
    gen_traj = trajectory_orig[W:]
    gen_truth = ground_truth_orig[W:]

    rmse_per_var = np.sqrt(np.mean((gen_traj - gen_truth) ** 2, axis=0))
    rmse_total = np.sqrt(np.mean((gen_traj - gen_truth) ** 2))

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"TRAJECTORY RMSE (Generated part only, steps {W}-{n_steps-1})")
    logger.info("=" * 60)
    for i, name in enumerate(STATE_NAMES):
        logger.info(f"  {name:12s}: {rmse_per_var[i]:.6f}")
    logger.info(f"  {'TOTAL':12s}: {rmse_total:.6f}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info("=" * 60)

    return {
        "model": trainer.model,
        "processor": processor,
        "history": history,
        "trajectory_orig": trajectory_orig,
        "ground_truth_orig": ground_truth_orig
    }


# ============================================================================
# CLI
# ============================================================================
def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Sequence Windowed LSTM Training")
    parser.add_argument("--data_path", type=str, default="data/output/basalt_25c_lstm_input_500pts.npy")
    parser.add_argument("--output_dir", type=str, default="outputs/lstm_seq_window")
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--target_loss", type=float, default=1e-8)

    args = parser.parse_args()

    config = SeqWindowConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        window_size=args.window_size,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        target_loss=args.target_loss
    )

    run_seq_window_pipeline(config)


if __name__ == "__main__":
    main()
