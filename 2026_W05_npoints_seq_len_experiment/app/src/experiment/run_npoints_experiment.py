"""
================================================================================
N_POINTS vs SEQ_LEN EXPERIMENT
================================================================================

ARAŞTIRMA SORUSU:
-----------------
Veri boyutu (n_points) arttıkça, minimum çalışan seq_len nasıl değişiyor?

HİPOTEZ:
--------
Daha fazla veri noktası → Daha fazla training sample → Daha düşük seq_len yeterli

DENEY MATRİSİ:
--------------
    n_points=500:  seq_len = [50, 30, 20, 10, 5]
    n_points=1000: seq_len = [50, 30, 20, 10, 5, 3]
    n_points=2500: seq_len = [50, 30, 20, 10, 5, 3, 2]

BAŞARI KRİTERLERİ:
------------------
    1. Training loss < 1e-7
    2. Trajectory RMSE < 0.5
    3. No collapse (NaN, Inf)

================================================================================
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ==============================================================================
# CONSTANTS
# ==============================================================================
STATE_NAMES: List[str] = [
    "nH2_g", "nCO2_g", "nCH4_g", "nH2S_g",
    "H2_aq", "CO2_aq", "SO4", "FeS",
    "X", "Acetate", "HCO3", "S_tot",
    "Lag", "Fe_pool"
]

# Deney matrisi: Her n_points için test edilecek seq_len değerleri
EXPERIMENT_MATRIX: Dict[int, List[int]] = {
    500:  [50, 30, 20, 10, 5],
    1000: [50, 30, 20, 10, 5, 3],
    2500: [50, 30, 20, 10, 5, 3, 2],
}


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Logging kurulumu."""
    logger = logging.getLogger("npoints_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(console_handler.formatter)
        logger.addHandler(file_handler)

    return logger


# ==============================================================================
# CONFIGURATION
# ==============================================================================
@dataclass
class ExperimentConfig:
    """Deney konfigürasyonu."""

    # Paths
    data_dir: str = "data"
    output_dir: str = "outputs"

    # Model architecture (sabit)
    n_features: int = 14
    hidden_size: int = 128
    num_layers: int = 2

    # Training
    epochs: int = 10000
    learning_rate: float = 5e-4
    target_loss: float = 1e-8

    # Data preprocessing
    use_log_transform: bool = True
    log_cols: tuple = (3, 7, 9, 12, 13)

    # Success criteria
    rmse_threshold: float = 0.5

    # Device & seed
    device: str = "auto"
    seed: int = 42

    def to_dict(self) -> dict:
        return {
            "data_dir": self.data_dir,
            "output_dir": self.output_dir,
            "n_features": self.n_features,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "target_loss": self.target_loss,
            "use_log_transform": self.use_log_transform,
            "log_cols": list(self.log_cols),
            "rmse_threshold": self.rmse_threshold,
            "seed": self.seed
        }


# ==============================================================================
# DATA PROCESSING
# ==============================================================================
class DataProcessor:
    """Veri yükleme ve işleme."""

    def __init__(self, config: ExperimentConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.scaler: Optional[StandardScaler] = None

    def load_data(self, n_points: int) -> np.ndarray:
        """Belirli nokta sayısına sahip veriyi yükle."""
        data_path = Path(self.config.data_dir) / f"basalt_25c_lstm_input_{n_points}pts.npy"
        data = np.load(data_path)
        self.logger.info(f"Loaded {data_path.name}: shape={data.shape}")
        return data

    def preprocess(self, data: np.ndarray, fit_scaler: bool = True) -> np.ndarray:
        """Log transform + standardization."""
        data = data.copy()

        if self.config.use_log_transform:
            for col in self.config.log_cols:
                data[:, col] = np.log1p(np.maximum(data[:, col], 0))

        if fit_scaler:
            self.scaler = StandardScaler()
            data_norm = self.scaler.fit_transform(data)
        else:
            data_norm = self.scaler.transform(data)

        return data_norm

    def inverse_preprocess(self, data_norm: np.ndarray) -> np.ndarray:
        """Inverse transform."""
        data = self.scaler.inverse_transform(data_norm)

        if self.config.use_log_transform:
            for col in self.config.log_cols:
                data[:, col] = np.expm1(data[:, col])

        return data

    def create_windows(self, data: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sliding window ile X, Y oluştur."""
        X, Y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            Y.append(data[i + seq_len])
        return np.array(X), np.array(Y)


# ==============================================================================
# LSTM MODEL
# ==============================================================================
class SeqWindowLSTM(nn.Module):
    """LSTM model for next-step prediction."""

    def __init__(self, input_size: int = 14, hidden_size: int = 128,
                 num_layers: int = 2, output_size: int = 14):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ==============================================================================
# SINGLE EXPERIMENT
# ==============================================================================
def run_single_experiment(
    n_points: int,
    seq_len: int,
    data_norm: np.ndarray,
    data_raw: np.ndarray,
    processor: DataProcessor,
    config: ExperimentConfig,
    device: torch.device,
    logger: logging.Logger,
    output_dir: Path
) -> Dict:
    """Tek bir (n_points, seq_len) kombinasyonu için deney."""

    logger.info(f"  seq_len={seq_len}")

    # Output directory
    exp_dir = output_dir / f"seq_len_{seq_len}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create windows
    X, Y = processor.create_windows(data_norm, seq_len)
    n_samples = len(X)

    # To tensors
    X_tensor = torch.FloatTensor(X).to(device)
    Y_tensor = torch.FloatTensor(Y).to(device)

    # Model
    torch.manual_seed(config.seed)
    model = SeqWindowLSTM(
        input_size=config.n_features,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.n_features
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    # Training
    history = {"loss": [], "epoch": []}
    final_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        output = model(X_tensor)
        loss = criterion(output, Y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        history["loss"].append(loss_val)
        history["epoch"].append(epoch)

        if epoch % 1000 == 0:
            logger.info(f"    Epoch {epoch:5d} - Loss: {loss_val:.2e}")

        if loss_val < config.target_loss:
            logger.info(f"    Target reached at epoch {epoch}!")
            final_loss = loss_val
            break

        final_loss = loss_val

    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_points": n_points,
        "seq_len": seq_len,
        "final_loss": final_loss
    }, exp_dir / "model.pt")

    # Autoregressive trajectory
    model.eval()
    n_steps = len(data_raw)
    trajectory = np.zeros((n_steps, config.n_features))
    trajectory[:seq_len] = data_norm[:seq_len]
    window = data_norm[:seq_len].copy()

    with torch.no_grad():
        for t in range(seq_len, n_steps):
            window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
            pred = model(window_tensor)
            next_state = pred.squeeze().cpu().numpy()
            trajectory[t] = next_state
            window = np.vstack([window[1:], next_state])

    # Inverse transform & RMSE
    trajectory_orig = processor.inverse_preprocess(trajectory)
    gen_traj = trajectory_orig[seq_len:]
    gen_truth = data_raw[seq_len:]

    rmse_per_var = np.sqrt(np.mean((gen_traj - gen_truth) ** 2, axis=0))
    rmse_total = float(np.sqrt(np.mean((gen_traj - gen_truth) ** 2)))

    # Collapse check
    has_nan = bool(np.isnan(trajectory_orig).any())
    has_inf = bool(np.isinf(trajectory_orig).any())
    traj_max = float(trajectory_orig.max())
    traj_min = float(trajectory_orig.min())
    collapsed = has_nan or has_inf or traj_max > 1e6 or traj_min < -1e6

    # Success
    success = (
        (final_loss < config.target_loss * 10) and
        (rmse_total < config.rmse_threshold) and
        not collapsed
    )

    # Result
    result = {
        "n_points": n_points,
        "seq_len": seq_len,
        "n_samples": n_samples,
        "final_loss": float(final_loss),
        "epochs_trained": len(history["loss"]),
        "rmse_total": rmse_total,
        "rmse_per_var": rmse_per_var.tolist(),
        "collapsed": collapsed,
        "success": success
    }

    # Save
    with open(exp_dir / "result.json", 'w') as f:
        json.dump(result, f, indent=2)

    np.savez(exp_dir / "trajectory.npz",
             trajectory_orig=trajectory_orig,
             ground_truth_orig=data_raw)

    status = "SUCCESS" if success else "FAILED"
    logger.info(f"    → {status} | loss={final_loss:.2e} | RMSE={rmse_total:.4f} | samples={n_samples}")

    return result


# ==============================================================================
# VISUALIZATION
# ==============================================================================
def plot_scaling_analysis(all_results: Dict[int, List[Dict]], output_dir: Path, logger: logging.Logger):
    """n_points vs min_seq_len scaling analizi."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Data extraction
    n_points_list = sorted(all_results.keys())
    colors = {500: 'blue', 1000: 'orange', 2500: 'green'}

    # Plot 1: RMSE vs seq_len for each n_points
    ax1 = axes[0, 0]
    for n_points in n_points_list:
        results = all_results[n_points]
        seq_lens = [r["seq_len"] for r in results]
        rmses = [r["rmse_total"] for r in results]
        ax1.plot(seq_lens, rmses, 'o-', label=f'n={n_points}', color=colors[n_points], linewidth=2, markersize=8)

    ax1.axhline(y=0.5, color='red', linestyle='--', label='Threshold')
    ax1.set_xlabel('Sequence Length (seq_len)', fontsize=12)
    ax1.set_ylabel('Trajectory RMSE', fontsize=12)
    ax1.set_title('RMSE vs Sequence Length', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([2, 3, 5, 10, 20, 30, 50])

    # Plot 2: Training loss vs seq_len
    ax2 = axes[0, 1]
    for n_points in n_points_list:
        results = all_results[n_points]
        seq_lens = [r["seq_len"] for r in results]
        losses = [r["final_loss"] for r in results]
        ax2.plot(seq_lens, losses, 'o-', label=f'n={n_points}', color=colors[n_points], linewidth=2, markersize=8)

    ax2.axhline(y=1e-7, color='red', linestyle='--', label='Target×10')
    ax2.set_xlabel('Sequence Length (seq_len)', fontsize=12)
    ax2.set_ylabel('Final Training Loss', fontsize=12)
    ax2.set_title('Training Loss vs Sequence Length', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([2, 3, 5, 10, 20, 30, 50])

    # Plot 3: n_samples vs seq_len
    ax3 = axes[1, 0]
    for n_points in n_points_list:
        results = all_results[n_points]
        seq_lens = [r["seq_len"] for r in results]
        samples = [r["n_samples"] for r in results]
        ax3.plot(seq_lens, samples, 'o-', label=f'n={n_points}', color=colors[n_points], linewidth=2, markersize=8)

    ax3.set_xlabel('Sequence Length (seq_len)', fontsize=12)
    ax3.set_ylabel('Number of Training Samples', fontsize=12)
    ax3.set_title('Training Samples vs Sequence Length', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks([2, 3, 5, 10, 20, 30, 50])

    # Plot 4: Minimum successful seq_len vs n_points (SCALING LAW)
    ax4 = axes[1, 1]
    min_seq_lens = []
    for n_points in n_points_list:
        results = all_results[n_points]
        successful = [r for r in results if r["success"]]
        if successful:
            min_seq = min(r["seq_len"] for r in successful)
        else:
            min_seq = None
        min_seq_lens.append((n_points, min_seq))

    valid_points = [(n, s) for n, s in min_seq_lens if s is not None]
    if valid_points:
        ns, ss = zip(*valid_points)
        ax4.bar(range(len(ns)), ss, color=[colors[n] for n in ns], alpha=0.7, edgecolor='black')
        ax4.set_xticks(range(len(ns)))
        ax4.set_xticklabels([str(n) for n in ns])

    ax4.set_xlabel('Number of Data Points (n_points)', fontsize=12)
    ax4.set_ylabel('Minimum Successful seq_len', fontsize=12)
    ax4.set_title('SCALING LAW: n_points vs min_seq_len', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "scaling_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_dir / 'scaling_analysis.png'}")


def plot_summary_table(all_results: Dict[int, List[Dict]], output_dir: Path, logger: logging.Logger):
    """Özet tablo grafiği."""

    # Collect all data
    rows = []
    for n_points in sorted(all_results.keys()):
        for r in all_results[n_points]:
            rows.append({
                "n_points": n_points,
                "seq_len": r["seq_len"],
                "rmse": r["rmse_total"],
                "loss": r["final_loss"],
                "success": r["success"]
            })

    # Create heatmap-style table
    n_points_list = sorted(all_results.keys())
    all_seq_lens = sorted(set(r["seq_len"] for row in all_results.values() for r in row), reverse=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create matrix
    matrix = np.full((len(n_points_list), len(all_seq_lens)), np.nan)
    for i, n_points in enumerate(n_points_list):
        for r in all_results[n_points]:
            j = all_seq_lens.index(r["seq_len"])
            matrix[i, j] = r["rmse_total"]

    # Plot
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(all_seq_lens)))
    ax.set_xticklabels(all_seq_lens)
    ax.set_yticks(range(len(n_points_list)))
    ax.set_yticklabels(n_points_list)

    ax.set_xlabel('Sequence Length (seq_len)', fontsize=12)
    ax.set_ylabel('Number of Data Points (n_points)', fontsize=12)
    ax.set_title('RMSE Heatmap: n_points × seq_len\n(Green=Good, Red=Bad, White=Not Tested)', fontsize=13, fontweight='bold')

    # Add values
    for i in range(len(n_points_list)):
        for j in range(len(all_seq_lens)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.3 else 'black'
                text = f'{val:.3f}'
                # Find success status
                n_points = n_points_list[i]
                seq_len = all_seq_lens[j]
                success = any(r["success"] for r in all_results[n_points] if r["seq_len"] == seq_len)
                if success:
                    text += '\n✓'
                ax.text(j, i, text, ha='center', va='center', fontsize=10, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='RMSE', shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / "summary_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_dir / 'summary_heatmap.png'}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    """Ana deney fonksiyonu."""

    # Config
    config = ExperimentConfig()

    # Output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(output_dir / f"experiment_{timestamp}.log")

    logger.info("=" * 70)
    logger.info("N_POINTS vs SEQ_LEN EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Experiment matrix: {EXPERIMENT_MATRIX}")
    logger.info("")

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    # Device
    if config.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    logger.info(f"Using device: {device}")

    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Run all experiments
    all_results: Dict[int, List[Dict]] = {}

    for n_points, seq_lens in EXPERIMENT_MATRIX.items():
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"N_POINTS = {n_points}")
        logger.info("=" * 70)

        # Create output dir for this n_points
        n_output_dir = output_dir / f"n{n_points}"
        n_output_dir.mkdir(parents=True, exist_ok=True)

        # Load and preprocess data
        processor = DataProcessor(config, logger)
        data_raw = processor.load_data(n_points)
        data_norm = processor.preprocess(data_raw, fit_scaler=True)

        results: List[Dict] = []

        for seq_len in seq_lens:
            result = run_single_experiment(
                n_points=n_points,
                seq_len=seq_len,
                data_norm=data_norm,
                data_raw=data_raw,
                processor=processor,
                config=config,
                device=device,
                logger=logger,
                output_dir=n_output_dir
            )
            results.append(result)

        all_results[n_points] = results

        # Save results for this n_points
        with open(n_output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

    # Save all results
    with open(output_dir / "all_results.json", 'w') as f:
        # Convert keys to strings for JSON
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    # Generate plots
    logger.info("")
    logger.info("=" * 70)
    logger.info("GENERATING ANALYSIS PLOTS")
    logger.info("=" * 70)

    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_scaling_analysis(all_results, figures_dir, logger)
    plot_summary_table(all_results, figures_dir, logger)

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("EXPERIMENT COMPLETE - SUMMARY")
    logger.info("=" * 70)

    for n_points in sorted(all_results.keys()):
        logger.info(f"\nn_points = {n_points}:")
        results = all_results[n_points]
        for r in results:
            status = "SUCCESS" if r["success"] else "FAILED"
            logger.info(f"  seq_len={r['seq_len']:3d}: {status:7s} | RMSE={r['rmse_total']:.4f} | loss={r['final_loss']:.2e}")

        # Min successful
        successful = [r for r in results if r["success"]]
        if successful:
            min_seq = min(r["seq_len"] for r in successful)
            logger.info(f"  → Minimum successful seq_len: {min_seq}")
        else:
            logger.info(f"  → No successful experiments!")

    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
