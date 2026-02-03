"""
seq_len=5 için detaylı trajectory analiz grafiği.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# State isimleri
STATE_NAMES = [
    "nH2_g", "nCO2_g", "nCH4_g", "nH2S_g",
    "H2_aq", "CO2_aq", "SO4", "FeS",
    "X", "Acetate", "HCO3", "S_tot",
    "Lag", "Fe_pool"
]


def main():
    # Veriyi yükle
    data = np.load("outputs/seq_len_experiment/seq_len_5/trajectory.npz")
    trajectory = data["trajectory_orig"]
    ground_truth = data["ground_truth_orig"]

    # Key variables
    key_vars = [0, 2, 6, 8]  # nH2_g, nCH4_g, SO4, X
    seq_len = 5

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    t = np.arange(len(trajectory))

    for i, var_idx in enumerate(key_vars):
        ax = axes[i]

        # Ground truth
        ax.plot(t, ground_truth[:, var_idx], 'b-', linewidth=2,
                label='Ground Truth (ODE)', alpha=0.8)

        # Generated
        ax.plot(t, trajectory[:, var_idx], 'r--', linewidth=2,
                label='Generated (LSTM)', alpha=0.8)

        # seq_len sınırı
        ax.axvline(x=seq_len, color='green', linestyle=':', linewidth=2,
                   label=f'seq_len={seq_len} boundary')

        # Fark alanını göster
        ax.fill_between(t[seq_len:], ground_truth[seq_len:, var_idx],
                        trajectory[seq_len:, var_idx],
                        alpha=0.2, color='red', label='Error region')

        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel(STATE_NAMES[var_idx], fontsize=12)
        ax.set_title(f'{STATE_NAMES[var_idx]} - seq_len=5', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # RMSE değerlerini hesapla
    rmse_per_var = np.sqrt(np.mean((trajectory[seq_len:] - ground_truth[seq_len:]) ** 2, axis=0))

    fig.suptitle(
        f'seq_len=5 Trajectory Analysis\nTotal RMSE = 0.5596 (> 0.5 threshold → FAILED)',
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()

    # Figures klasörünü oluştur
    Path("figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("figures/seq_len_5_detailed.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved to figures/seq_len_5_detailed.png")

    # Her değişken için RMSE
    print("\nRMSE per variable (seq_len=5):")
    for i, name in enumerate(STATE_NAMES):
        print(f"  {name:10s}: {rmse_per_var[i]:.6f}")


if __name__ == "__main__":
    main()
