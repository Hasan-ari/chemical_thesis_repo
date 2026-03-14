"""
============================================================================
NEXT-STEP LSTM - Single Step Prediction (Horizon = 1)
============================================================================

This module implements a fundamentally different approach from multi-step
horizon prediction. Instead of predicting N steps ahead, we predict just
the NEXT timestep (t+1) from the current timestep (t).

============================================================================
500 DATA POINTS: WHAT ARE WE TRAINING ON?
============================================================================

SOURCE OF DATA:
---------------
The original ODE solver (MATLAB's ode15s / Python's solve_ivp with Radau)
produces ADAPTIVE timesteps - more points where dynamics change fast,
fewer where things are smooth. This is great for accuracy but BAD for LSTM
because LSTMs expect UNIFORM time intervals.

DOWNSAMPLING:
-------------
We resample the ODE solution onto a UNIFORM grid:

    Original ODE:  ~4800 points, variable dt (0.001 to 0.1 days)
    Downsampled:   500 points, fixed dt = 0.038 days (55 minutes)

    Time span: 0 to 19 days (Basalt @ 25C experiment duration)

    dt = 19 days / (500-1 points) = 0.038 days ≈ 55 minutes per step

WHY 500 POINTS?
---------------
    - 500 points is enough to capture the reaction dynamics
    - Each point represents ~55 minutes of chemical reaction
    - Total: 499 training pairs (y[t] → y[t+1])
    - Small enough to overfit completely
    - Large enough to capture all phases: lag, exponential, stationary

WHAT EACH DATA POINT CONTAINS:
------------------------------
Each y[t] is a 14-dimensional vector representing the system STATE:

    y[t] = [nH2_g,    # H2 gas moles in headspace (mmol)
            nCO2_g,   # CO2 gas moles (mmol)
            nCH4_g,   # CH4 gas produced (mmol)
            nH2S_g,   # H2S gas (mmol)
            H2_aq,    # Dissolved H2 (mM)
            CO2_aq,   # Dissolved CO2 (mM)
            SO4,      # Sulfate concentration (mM)
            FeS,      # Iron sulfide precipitate (mM)
            X,        # Microbial biomass (mM)
            Acetate,  # Acetate concentration (mM)
            HCO3,     # Bicarbonate (mM)
            S_tot,    # Total dissolved sulfide (mM)
            Lag,      # Lag phase factor (0-1)
            Fe_pool]  # Available iron for precipitation (mM)

TRAINING OBJECTIVE:
-------------------
Learn the mapping: y[t] → y[t+1]

This is equivalent to learning the discrete-time dynamics:

    y[t+1] = f(y[t])

Where f is the LSTM approximation of the ODE integrated over dt=55 min.

If perfectly learned, we can SIMULATE the ODE by recursive application:

    y[0] → y[1] → y[2] → ... → y[499]

Starting from ONLY the initial condition y[0].

============================================================================
KEY DIFFERENCE: HORIZON VS NEXT-STEP
============================================================================

HORIZON APPROACH (lstm_trainer.py):
-----------------------------------
    Input:  [y[t-99], y[t-98], ..., y[t]]   (100 timesteps as context)
    Output: y[t+10]                          (predict 10 steps ahead)

    - Requires long context window
    - Predicts further into future
    - More robust to single-step errors

NEXT-STEP APPROACH (this file):
-------------------------------
    Input:  y[t]                             (single timestep)
    Output: y[t+1]                           (predict next step only)

    - Minimal input (just current state)
    - LSTM hidden state carries temporal memory
    - Errors compound during autoregressive generation

============================================================================
AUTOREGRESSIVE TRAJECTORY GENERATION
============================================================================

The key insight: If the model perfectly learns y[t] → y[t+1], we can
generate the ENTIRE trajectory from just the initial condition y[0]:

    y[0] (given)
      ↓ LSTM
    ŷ[1] = model(y[0])
      ↓ feed back
    ŷ[2] = model(ŷ[1])
      ↓ feed back
    ŷ[3] = model(ŷ[2])
      ↓ ...
    ŷ[N] = model(ŷ[N-1])

This tests whether the LSTM truly learned the underlying ODE dynamics,
not just pattern matching from long context windows.

============================================================================
WHY OVERFITTING? (INTENTIONAL!)
============================================================================

Normally, overfitting is BAD - it means memorizing training data instead
of learning generalizable patterns. But here, overfitting is our GOAL:

WHY WE WANT TO OVERFIT:
-----------------------
1. We have ONE specific trajectory (Basalt @ 25C with best-fit parameters)
2. We want to PERFECTLY reproduce this trajectory
3. We're not trying to generalize to other experiments
4. This is a PROOF OF CONCEPT: Can LSTM learn ODE dynamics?

OVERFITTING STRATEGY:
---------------------
    - No train/val/test split → Use ALL 499 pairs for training
    - No early stopping → Train until loss is tiny
    - No dropout/regularization → Let model memorize freely
    - Target loss: < 1e-6 (near-perfect reconstruction)

THE CHALLENGE:
--------------
Even with perfect training loss (1e-6), autoregressive generation fails!

    Training:   y[t] → ŷ[t+1]  with loss = 1e-6 per step

    Generation: y[0] → ŷ[1] → ŷ[2] → ... → ŷ[499]
                       ↑        ↑
                   error₁   error₂ = f(error₁) + new_error

    After 499 steps: errors COMPOUND exponentially!

RESULT OF OUR EXPERIMENT:
-------------------------
    v1: LSTM(64)x1,  loss=1.88e-5 → trajectory collapsed
    v2: LSTM(128)x2, loss=1.04e-6 → trajectory STILL collapsed

LESSON LEARNED:
---------------
Single-point autoregressive generation over 499 steps is EXTREMELY hard.
Even tiny per-step errors (~1e-6) accumulate to massive drift.
The LSTM hidden state alone cannot maintain trajectory coherence
without explicit context from the ground truth sequence.

============================================================================
TRAINING DATA STRUCTURE
============================================================================

For 500 data points [y[0], y[1], ..., y[499]]:

    X (inputs):  [y[0], y[1], y[2], ..., y[498]]   → 499 samples
    Y (targets): [y[1], y[2], y[3], ..., y[499]]   → 499 samples

Each sample is just ONE timestep (14 features), not a sequence!
The LSTM processes samples sequentially, maintaining hidden state.

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
    logger = logging.getLogger("lstm_next_step")
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
# CONFIGURATION
# ============================================================================
@dataclass
class NextStepConfig:
    """Configuration for next-step LSTM training."""

    # Data
    data_path: str = "data/output/basalt_25c_lstm_input_500pts.npy"
    output_dir: str = "outputs/lstm_next_step"

    # Model architecture (simpler for overfitting)
    n_features: int = 14
    hidden_size: int = 64      # Single LSTM layer
    num_layers: int = 1        # Keep simple for overfitting

    # Training (aggressive overfitting)
    epochs: int = 5000         # Many epochs for deep overfit
    batch_size: int = 499      # Full batch (all samples at once)
    learning_rate: float = 1e-3
    target_loss: float = 1e-6  # Stop when loss is this low

    # Preprocessing
    use_log_transform: bool = True
    log_cols: tuple = (3, 7, 9, 12, 13)  # nH2S_g, FeS, Acetate, Lag, Fe_pool

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
class NextStepDataProcessor:
    """
    Prepares data for next-step prediction.

    Unlike the horizon approach, we don't create sequence windows.
    Each sample is just a single timestep.
    """

    def __init__(self, config: NextStepConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.scaler = None

    def load_data(self) -> np.ndarray:
        """Load raw data from .npy file."""
        data = np.load(self.config.data_path)
        self.logger.info(f"Loaded data: {data.shape} from {self.config.data_path}")
        return data

    def preprocess(self, data: np.ndarray, fit_scaler: bool = True) -> np.ndarray:
        """
        Preprocess data: log transform + standardization.

        Args:
            data: Raw data (N, 14)
            fit_scaler: If True, fit new scaler. If False, use existing.

        Returns:
            Normalized data (N, 14)
        """
        data = data.copy()

        # Log transform for small-valued columns
        if self.config.use_log_transform:
            for col in self.config.log_cols:
                data[:, col] = np.log1p(np.maximum(data[:, col], 0))

        # Standardization
        if fit_scaler:
            self.scaler = StandardScaler()
            data_norm = self.scaler.fit_transform(data)
            self.logger.info("Fitted new StandardScaler")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted")
            data_norm = self.scaler.transform(data)

        return data_norm

    def inverse_preprocess(self, data_norm: np.ndarray) -> np.ndarray:
        """Convert normalized data back to original scale."""
        data = self.scaler.inverse_transform(data_norm)

        # Reverse log transform
        if self.config.use_log_transform:
            for col in self.config.log_cols:
                data[:, col] = np.expm1(data[:, col])

        return data

    def create_next_step_pairs(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output pairs for next-step prediction.

        ============================================================
        NEXT-STEP PAIR CREATION
        ============================================================

        For data [y[0], y[1], y[2], ..., y[N-1]]:

            X[0] = y[0]  →  Y[0] = y[1]
            X[1] = y[1]  →  Y[1] = y[2]
            X[2] = y[2]  →  Y[2] = y[3]
            ...
            X[N-2] = y[N-2]  →  Y[N-2] = y[N-1]

        Result: N-1 training pairs

        Shape:
            X: (N-1, 14) - each row is ONE timestep
            Y: (N-1, 14) - each row is the NEXT timestep
        ============================================================

        Args:
            data: Normalized data (N, 14)

        Returns:
            X: Input states (N-1, 14)
            Y: Target states (N-1, 14)
        """
        # X = all timesteps except the last
        X = data[:-1]  # [y[0], y[1], ..., y[N-2]]

        # Y = all timesteps except the first
        Y = data[1:]   # [y[1], y[2], ..., y[N-1]]

        self.logger.info(f"Created next-step pairs: X={X.shape}, Y={Y.shape}")

        return X, Y

    def save_scaler(self, path: Path):
        """Save scaler for later use."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        self.logger.info(f"Saved scaler to {path}")


# ============================================================================
# MODEL
# ============================================================================
class NextStepLSTM(nn.Module):
    """
    Simple LSTM for next-step prediction.

    ============================================================
    ARCHITECTURE
    ============================================================

    Input:  y[t]     (14 features)
              ↓
    LSTM:   hidden state carries temporal information
              ↓
    Linear: maps hidden → output
              ↓
    Output: ŷ[t+1]   (14 features)

    The LSTM hidden state is the "memory" that allows the model
    to learn temporal dynamics without explicit sequence input.
    ============================================================
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = 14
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Output projection
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, features) or (batch, features)
            hidden: Optional (h, c) tuple for LSTM state

        Returns:
            output: Predicted next state (batch, seq_len, features)
            hidden: Updated (h, c) tuple
        """
        # Ensure 3D input: (batch, seq_len, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)

        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)

        # Project to output
        output = self.fc(lstm_out)

        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden state to zeros."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)


# ============================================================================
# TRAINER
# ============================================================================
class NextStepTrainer:
    """Trains the next-step LSTM with aggressive overfitting."""

    def __init__(self, config: NextStepConfig, logger: logging.Logger):
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

        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Model
        self.model = NextStepLSTM(
            input_size=config.n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=config.n_features
        ).to(self.device)

        self.logger.info(f"Model: LSTM({config.hidden_size}) x {config.num_layers} → Linear({config.n_features})")

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        checkpoint_dir: Path
    ) -> Dict:
        """
        Train the model to overfit on the data.

        ============================================================
        TRAINING APPROACH FOR OVERFITTING
        ============================================================

        We want the model to MEMORIZE the training data perfectly.
        This means:
        - No validation set (use all data for training)
        - No early stopping (train until loss is tiny)
        - No regularization (no dropout, no weight decay)
        - Full batch training (all samples at once)

        The goal: loss < 1e-6, meaning nearly perfect reconstruction.
        ============================================================
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        Y_tensor = torch.FloatTensor(Y).to(self.device)

        # Add sequence dimension: (N, 1, 14)
        X_tensor = X_tensor.unsqueeze(1)
        Y_tensor = Y_tensor.unsqueeze(1)

        self.logger.info(f"Training data: X={X_tensor.shape}, Y={Y_tensor.shape}")

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        # Training history
        history = {"loss": [], "epoch": []}

        self.logger.info("=" * 60)
        self.logger.info("Starting Training (Overfitting Mode)")
        self.logger.info(f"Target loss: {self.config.target_loss}")
        self.logger.info("=" * 60)

        for epoch in range(self.config.epochs):
            self.model.train()

            # Forward pass (process all timesteps at once)
            # Hidden state initialized to zeros
            output, _ = self.model(X_tensor)

            # Compute loss
            loss = criterion(output, Y_tensor)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            history["loss"].append(loss_val)
            history["epoch"].append(epoch)

            # Logging
            if epoch % 100 == 0 or epoch == self.config.epochs - 1:
                self.logger.info(f"Epoch {epoch:5d}/{self.config.epochs} - Loss: {loss_val:.2e}")

            # Check if target loss reached
            if loss_val < self.config.target_loss:
                self.logger.info(f"Target loss {self.config.target_loss} reached at epoch {epoch}!")
                break

        # Save checkpoint
        checkpoint_path = checkpoint_dir / "lstm_next_step.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
            "final_loss": loss_val,
            "epochs_trained": epoch + 1
        }, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save history
        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f)

        return history


# ============================================================================
# TRAJECTORY GENERATOR
# ============================================================================
class TrajectoryGenerator:
    """
    Generates complete trajectories from initial conditions.

    ============================================================
    AUTOREGRESSIVE GENERATION
    ============================================================

    Given only y[0], generate the entire trajectory:

        Step 0: y[0] is given (initial condition)

        Step 1: ŷ[1], h[1] = LSTM(y[0], h[0]=zeros)
                Feed y[0], get prediction ŷ[1] and hidden state h[1]

        Step 2: ŷ[2], h[2] = LSTM(ŷ[1], h[1])
                Feed previous PREDICTION (not ground truth!)
                LSTM remembers context via hidden state h[1]

        Step 3: ŷ[3], h[3] = LSTM(ŷ[2], h[2])
                Continue feeding predictions back...

        ...

        Step N: ŷ[N], h[N] = LSTM(ŷ[N-1], h[N-1])

    Key insight: The hidden state h carries "memory" of the trajectory.
    This is how the model can generate coherent sequences from just y[0].

    WARNING: Errors compound! A small error at step 1 affects all
    subsequent predictions. This is why we need near-perfect overfitting.
    ============================================================
    """

    def __init__(
        self,
        model: NextStepLSTM,
        device: torch.device,
        logger: logging.Logger
    ):
        self.model = model
        self.device = device
        self.logger = logger

    def generate(
        self,
        y0: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """
        Generate trajectory from initial condition.

        Args:
            y0: Initial state (14,) - normalized
            n_steps: Number of steps to generate (including y0)

        Returns:
            trajectory: Generated trajectory (n_steps, 14) - normalized
        """
        self.model.eval()

        # Store trajectory (including initial condition)
        trajectory = np.zeros((n_steps, self.model.input_size))
        trajectory[0] = y0

        # Initialize hidden state
        hidden = self.model.init_hidden(batch_size=1, device=self.device)

        # Current state
        current = torch.FloatTensor(y0).unsqueeze(0).unsqueeze(0).to(self.device)
        # Shape: (1, 1, 14) = (batch, seq, features)

        with torch.no_grad():
            for t in range(1, n_steps):
                # Predict next step
                output, hidden = self.model(current, hidden)

                # output shape: (1, 1, 14)
                next_state = output.squeeze().cpu().numpy()

                # Store prediction
                trajectory[t] = next_state

                # Feed prediction back as next input (AUTOREGRESSIVE)
                current = output  # Use prediction as next input

        self.logger.info(f"Generated trajectory: {trajectory.shape}")

        return trajectory


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_next_step_pipeline(config: NextStepConfig) -> Dict:
    """
    Execute the full next-step training and generation pipeline.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(output_dir / f"training_{timestamp}.log")

    logger.info("=" * 60)
    logger.info("NEXT-STEP LSTM PIPELINE")
    logger.info("Goal: Overfit to learn y[t] → y[t+1] dynamics")
    logger.info("=" * 60)

    # Save config
    config.save(output_dir / "config.json")

    # Data processing
    processor = NextStepDataProcessor(config, logger)
    data_raw = processor.load_data()
    data_norm = processor.preprocess(data_raw, fit_scaler=True)
    processor.save_scaler(output_dir / "scaler.pkl")

    # Create training pairs
    X, Y = processor.create_next_step_pairs(data_norm)

    # Train
    trainer = NextStepTrainer(config, logger)
    history = trainer.train(X, Y, output_dir / "checkpoints")

    # Generate trajectory from initial condition
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAJECTORY GENERATION TEST")
    logger.info("=" * 60)

    generator = TrajectoryGenerator(trainer.model, trainer.device, logger)

    # Use the FIRST data point as initial condition
    y0_norm = data_norm[0]
    n_steps = len(data_raw)  # Generate same length as original

    trajectory_norm = generator.generate(y0_norm, n_steps)

    # Convert back to original scale
    trajectory_orig = processor.inverse_preprocess(trajectory_norm)
    ground_truth_orig = data_raw

    # Save results
    np.savez(
        output_dir / "trajectory.npz",
        trajectory_norm=trajectory_norm,
        trajectory_orig=trajectory_orig,
        ground_truth_norm=data_norm,
        ground_truth_orig=ground_truth_orig
    )
    logger.info(f"Saved trajectory to {output_dir / 'trajectory.npz'}")

    # Compute RMSE
    rmse_per_var = np.sqrt(np.mean((trajectory_orig - ground_truth_orig) ** 2, axis=0))
    rmse_total = np.sqrt(np.mean((trajectory_orig - ground_truth_orig) ** 2))

    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAJECTORY RMSE (vs Ground Truth)")
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

    parser = argparse.ArgumentParser(description="Next-Step LSTM Training")
    parser.add_argument("--data_path", type=str, default="data/output/basalt_25c_lstm_input_500pts.npy")
    parser.add_argument("--output_dir", type=str, default="outputs/lstm_next_step")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target_loss", type=float, default=1e-6)

    args = parser.parse_args()

    config = NextStepConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        target_loss=args.target_loss
    )

    run_next_step_pipeline(config)


if __name__ == "__main__":
    main()
