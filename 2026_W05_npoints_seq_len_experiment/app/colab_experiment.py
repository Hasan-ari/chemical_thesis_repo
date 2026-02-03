"""
================================================================================
N_POINTS vs SEQ_LEN EXPERIMENT - GOOGLE COLAB VERSION
================================================================================

Google Colab'da çalıştırmak için:
1. Bu dosyayı Colab'a yükle
2. Runtime → Change runtime type → GPU seç
3. Hücreleri sırayla çalıştır

Ya da direkt:
    !python colab_experiment.py
================================================================================
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# GPU SETUP
# ==============================================================================
def setup_device():
    """GPU varsa kullan, yoksa CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU bulundu: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Apple Silicon GPU (MPS) kullanılıyor")
    else:
        device = torch.device("cpu")
        print("⚠ GPU bulunamadı, CPU kullanılıyor")
    return device


# ==============================================================================
# CONSTANTS
# ==============================================================================
STATE_NAMES: List[str] = [
    "nH2_g", "nCO2_g", "nCH4_g", "nH2S_g",
    "H2_aq", "CO2_aq", "SO4", "FeS",
    "X", "Acetate", "HCO3", "S_tot",
    "Lag", "Fe_pool"
]

# Deney matrisi
EXPERIMENT_MATRIX: Dict[int, List[int]] = {
    500:  [50, 30, 20, 10, 5],
    1000: [50, 30, 20, 10, 5, 3],
    2500: [50, 30, 20, 10, 5, 3, 2],
}

# Hiperparametreler
CONFIG = {
    "n_features": 14,
    "hidden_size": 128,
    "num_layers": 2,
    "epochs": 10000,
    "learning_rate": 5e-4,
    "use_log_transform": True,
    "log_cols": (3, 7, 9, 12, 13),
    "seed": 42,
}


# ==============================================================================
# DATA PROCESSING
# ==============================================================================
class DataProcessor:
    """Veri yükleme ve işleme."""

    def __init__(self, use_log_transform: bool = True, log_cols: tuple = (3, 7, 9, 12, 13)):
        self.use_log_transform = use_log_transform
        self.log_cols = log_cols
        self.scaler: Optional[StandardScaler] = None

    def load_data(self, data_path: str) -> np.ndarray:
        """Veriyi yükle."""
        data = np.load(data_path)
        print(f"  Loaded: {data_path} | shape={data.shape}")
        return data

    def preprocess(self, data: np.ndarray, fit_scaler: bool = True) -> np.ndarray:
        """Log transform + standardization."""
        data = data.copy()

        if self.use_log_transform:
            for col in self.log_cols:
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

        if self.use_log_transform:
            for col in self.log_cols:
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
# TRAINING FUNCTION
# ==============================================================================
def train_model(
    model: nn.Module,
    X_tensor: torch.Tensor,
    Y_tensor: torch.Tensor,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    print_every: int = 2000
) -> Tuple[float, List[float]]:
    """Model eğitimi - sınırsız, FAILED yok."""

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    loss_history = []

    for epoch in range(epochs):
        model.train()
        output = model(X_tensor)
        loss = criterion(output, Y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if epoch % print_every == 0:
            print(f"      Epoch {epoch:5d} | Loss: {loss_val:.2e}")

    final_loss = loss_history[-1]
    print(f"      Final Loss: {final_loss:.2e}")

    return final_loss, loss_history


# ==============================================================================
# AUTOREGRESSIVE TRAJECTORY
# ==============================================================================
def generate_trajectory(
    model: nn.Module,
    data_norm: np.ndarray,
    seq_len: int,
    n_steps: int,
    device: torch.device
) -> np.ndarray:
    """Autoregressive trajectory üret."""

    model.eval()
    trajectory = np.zeros((n_steps, data_norm.shape[1]))
    trajectory[:seq_len] = data_norm[:seq_len]
    window = data_norm[:seq_len].copy()

    with torch.no_grad():
        for t in range(seq_len, n_steps):
            window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
            pred = model(window_tensor)
            next_state = pred.squeeze().cpu().numpy()
            trajectory[t] = next_state
            window = np.vstack([window[1:], next_state])

    return trajectory


# ==============================================================================
# SINGLE EXPERIMENT
# ==============================================================================
def run_single_experiment(
    n_points: int,
    seq_len: int,
    data_norm: np.ndarray,
    data_raw: np.ndarray,
    processor: DataProcessor,
    device: torch.device,
    output_dir: Path
) -> Dict:
    """Tek bir (n_points, seq_len) kombinasyonu için deney."""

    print(f"\n    → seq_len={seq_len}")

    # Output directory
    exp_dir = output_dir / f"seq_len_{seq_len}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create windows
    X, Y = processor.create_windows(data_norm, seq_len)
    n_samples = len(X)
    print(f"      Samples: {n_samples}")

    # To tensors
    X_tensor = torch.FloatTensor(X).to(device)
    Y_tensor = torch.FloatTensor(Y).to(device)

    # Model
    torch.manual_seed(CONFIG["seed"])
    model = SeqWindowLSTM(
        input_size=CONFIG["n_features"],
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        output_size=CONFIG["n_features"]
    ).to(device)

    # Train
    final_loss, loss_history = train_model(
        model, X_tensor, Y_tensor,
        epochs=CONFIG["epochs"],
        learning_rate=CONFIG["learning_rate"],
        device=device
    )

    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_points": n_points,
        "seq_len": seq_len,
        "final_loss": final_loss
    }, exp_dir / "model.pt")

    # Generate trajectory
    trajectory = generate_trajectory(
        model, data_norm, seq_len, len(data_raw), device
    )

    # Inverse transform
    trajectory_orig = processor.inverse_preprocess(trajectory)

    # RMSE (sadece generated kısım)
    gen_traj = trajectory_orig[seq_len:]
    gen_truth = data_raw[seq_len:]
    rmse_per_var = np.sqrt(np.mean((gen_traj - gen_truth) ** 2, axis=0))
    rmse_total = float(np.sqrt(np.mean((gen_traj - gen_truth) ** 2)))

    print(f"      RMSE: {rmse_total:.4f}")

    # Result - FAILED/SUCCESS YOK, sadece değerler
    result = {
        "n_points": n_points,
        "seq_len": seq_len,
        "n_samples": n_samples,
        "final_loss": float(final_loss),
        "rmse_total": rmse_total,
        "rmse_per_var": rmse_per_var.tolist(),
    }

    # Save
    with open(exp_dir / "result.json", 'w') as f:
        json.dump(result, f, indent=2)

    np.savez(exp_dir / "trajectory.npz",
             trajectory_orig=trajectory_orig,
             ground_truth_orig=data_raw)

    return result


# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================
def run_full_experiment(data_dir: str = "data", output_dir: str = "outputs_colab"):
    """Tüm deneyleri çalıştır."""

    print("=" * 70)
    print("N_POINTS vs SEQ_LEN EXPERIMENT - COLAB VERSION")
    print("=" * 70)

    # Device
    device = setup_device()

    # Seed
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # Output dir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_path / "config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)

    # All results
    all_results: Dict[int, List[Dict]] = {}

    # Run experiments
    for n_points, seq_lens in EXPERIMENT_MATRIX.items():
        print("\n" + "=" * 70)
        print(f"N_POINTS = {n_points}")
        print("=" * 70)

        # Data path
        data_path = f"{data_dir}/basalt_25c_lstm_input_{n_points}pts.npy"

        # Check if file exists
        if not os.path.exists(data_path):
            print(f"  ⚠ Dosya bulunamadı: {data_path}")
            print(f"  → Colab'da dosyayı yükleyin veya drive'dan bağlayın")
            continue

        # Output dir for this n_points
        n_output_dir = output_path / f"n{n_points}"
        n_output_dir.mkdir(parents=True, exist_ok=True)

        # Load and preprocess
        processor = DataProcessor(
            use_log_transform=CONFIG["use_log_transform"],
            log_cols=CONFIG["log_cols"]
        )
        data_raw = processor.load_data(data_path)
        data_norm = processor.preprocess(data_raw, fit_scaler=True)

        results: List[Dict] = []

        for seq_len in seq_lens:
            result = run_single_experiment(
                n_points=n_points,
                seq_len=seq_len,
                data_norm=data_norm,
                data_raw=data_raw,
                processor=processor,
                device=device,
                output_dir=n_output_dir
            )
            results.append(result)

        all_results[n_points] = results

        # Save results for this n_points
        with open(n_output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

    # Save all results
    with open(output_path / "all_results.json", 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE - SUMMARY")
    print("=" * 70)

    for n_points in sorted(all_results.keys()):
        print(f"\nn_points = {n_points}:")
        print(f"  {'seq_len':<10} {'RMSE':<12} {'Loss':<12} {'Samples':<10}")
        print(f"  {'-'*44}")
        for r in all_results[n_points]:
            rmse_status = "✓" if r["rmse_total"] < 0.5 else "✗"
            print(f"  {r['seq_len']:<10} {r['rmse_total']:<12.4f} {r['final_loss']:<12.2e} {r['n_samples']:<10} {rmse_status}")

        # Find min seq_len with RMSE < 0.5
        good_results = [r for r in all_results[n_points] if r["rmse_total"] < 0.5]
        if good_results:
            min_seq = min(r["seq_len"] for r in good_results)
            print(f"  → Min seq_len (RMSE < 0.5): {min_seq}")

    print(f"\n✓ Outputs saved to: {output_path}")

    return all_results


# ==============================================================================
# COLAB NOTEBOOK CELLS (copy-paste için)
# ==============================================================================
COLAB_CELLS = """
# ============================================================================
# CELL 1: Setup & GPU Check
# ============================================================================
!pip install -q torch numpy scikit-learn matplotlib

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# CELL 2: Upload Data Files
# ============================================================================
from google.colab import files
import os

os.makedirs("data", exist_ok=True)

print("Upload your data files:")
print("  - basalt_25c_lstm_input_500pts.npy")
print("  - basalt_25c_lstm_input_1000pts.npy")
print("  - basalt_25c_lstm_input_2500pts.npy")

uploaded = files.upload()
for fname in uploaded.keys():
    os.rename(fname, f"data/{fname}")
    print(f"  ✓ Moved to data/{fname}")

# ============================================================================
# CELL 3: Upload & Run Experiment Script
# ============================================================================
# colab_experiment.py dosyasını yükle
uploaded = files.upload()  # colab_experiment.py seç

# ============================================================================
# CELL 4: Run Experiment
# ============================================================================
from colab_experiment import run_full_experiment

results = run_full_experiment(data_dir="data", output_dir="outputs_colab")

# ============================================================================
# CELL 5: Download Results
# ============================================================================
!zip -r results.zip outputs_colab/
files.download("results.zip")
"""


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COLAB EXPERIMENT SCRIPT")
    print("=" * 70)
    print("\nBu script'i 2 şekilde kullanabilirsin:\n")
    print("1. Direkt çalıştır (local veya Colab):")
    print("   python colab_experiment.py\n")
    print("2. Colab'da import et:")
    print("   from colab_experiment import run_full_experiment")
    print("   results = run_full_experiment()\n")
    print("=" * 70 + "\n")

    # Run experiment
    run_full_experiment()
