"""
Tüm seq_len değerleri ve tüm 14 variable için karşılaştırma grafikleri.

Her variable için bir figür oluşturur:
- 5 subplot (seq_len = 50, 30, 20, 10, 5)
- Ground Truth vs Generated
- RMSE değerleri
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


# State isimleri ve açıklamaları
STATE_NAMES: List[str] = [
    "nH2_g",    # 0:  Gaz fazı H2 (mmol)
    "nCO2_g",   # 1:  Gaz fazı CO2 (mmol)
    "nCH4_g",   # 2:  Gaz fazı CH4 (mmol)
    "nH2S_g",   # 3:  Gaz fazı H2S (mmol)
    "H2_aq",    # 4:  Çözünmüş H2 (mmol/L)
    "CO2_aq",   # 5:  Çözünmüş CO2 (mmol/L)
    "SO4",      # 6:  Sülfat (mmol/L)
    "FeS",      # 7:  Demir sülfür (mmol/L)
    "X",        # 8:  Biyokütle (mmol/L)
    "Acetate",  # 9:  Asetat (mmol/L)
    "HCO3",     # 10: Bikarbonat (mmol/L)
    "S_tot",    # 11: Toplam sülfür (mmol/L)
    "Lag",      # 12: Lag fazı (0-1)
    "Fe_pool"   # 13: Fe havuzu (mmol/L)
]

STATE_DESCRIPTIONS: List[str] = [
    "Gas phase H₂ (mmol)",
    "Gas phase CO₂ (mmol)",
    "Gas phase CH₄ (mmol)",
    "Gas phase H₂S (mmol)",
    "Dissolved H₂ (mmol/L)",
    "Dissolved CO₂ (mmol/L)",
    "Sulfate SO₄²⁻ (mmol/L)",
    "Iron sulfide FeS (mmol/L)",
    "Biomass X (mmol/L)",
    "Acetate (mmol/L)",
    "Bicarbonate HCO₃⁻ (mmol/L)",
    "Total dissolved sulfur (mmol/L)",
    "Lag phase activation (0-1)",
    "Fe²⁺ pool (mmol/L)"
]

SEQ_LENGTHS: List[int] = [50, 30, 20, 10, 5]


def load_all_trajectories(base_dir: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Tüm seq_len değerleri için trajectory verilerini yükle.

    Returns:
        Dict[seq_len, (trajectory, ground_truth)]
    """
    data_dict: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for seq_len in SEQ_LENGTHS:
        npz_path = base_dir / f"seq_len_{seq_len}" / "trajectory.npz"
        data = np.load(npz_path)
        trajectory = data["trajectory_orig"]
        ground_truth = data["ground_truth_orig"]
        data_dict[seq_len] = (trajectory, ground_truth)
        print(f"Loaded seq_len={seq_len}: trajectory shape = {trajectory.shape}")

    return data_dict


def compute_rmse(trajectory: np.ndarray, ground_truth: np.ndarray,
                 seq_len: int, var_idx: int) -> float:
    """Belirli bir variable için RMSE hesapla (sadece generated kısım)."""
    gen_traj = trajectory[seq_len:, var_idx]
    gen_truth = ground_truth[seq_len:, var_idx]
    rmse = np.sqrt(np.mean((gen_traj - gen_truth) ** 2))
    return rmse


def plot_single_variable(var_idx: int, data_dict: Dict, output_dir: Path) -> None:
    """
    Tek bir variable için tüm seq_len'leri karşılaştıran figür oluştur.

    Layout: 1 satır × 5 sütun (her seq_len için bir subplot)
    """
    var_name = STATE_NAMES[var_idx]
    var_desc = STATE_DESCRIPTIONS[var_idx]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Ground truth (hepsi aynı)
    _, ground_truth = data_dict[50]
    t = np.arange(len(ground_truth))

    rmse_values: List[float] = []

    for i, seq_len in enumerate(SEQ_LENGTHS):
        ax = axes[i]
        trajectory, gt = data_dict[seq_len]

        # RMSE hesapla
        rmse = compute_rmse(trajectory, gt, seq_len, var_idx)
        rmse_values.append(rmse)

        # Ground truth
        ax.plot(t, gt[:, var_idx], 'b-', linewidth=1.5,
                label='Ground Truth', alpha=0.8)

        # Generated
        ax.plot(t, trajectory[:, var_idx], 'r--', linewidth=1.5,
                label='Generated', alpha=0.8)

        # seq_len boundary
        ax.axvline(x=seq_len, color='green', linestyle=':',
                   linewidth=1.5, alpha=0.7)

        # Error region (hafif)
        ax.fill_between(t[seq_len:], gt[seq_len:, var_idx],
                        trajectory[seq_len:, var_idx],
                        alpha=0.15, color='red')

        # Title ve labels
        rmse_color = 'red' if rmse > 0.5 else 'green'
        ax.set_title(f'seq_len={seq_len}\nRMSE={rmse:.4f}',
                     fontsize=11, fontweight='bold', color=rmse_color)
        ax.set_xlabel('Time Step', fontsize=10)

        if i == 0:
            ax.set_ylabel(var_name, fontsize=11)
            ax.legend(fontsize=8, loc='best')

        ax.grid(True, alpha=0.3)

    # Ana başlık
    fig.suptitle(f'{var_name}: {var_desc}', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Kaydet
    save_path = output_dir / f"var_{var_idx:02d}_{var_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path.name} | RMSE: {rmse_values}")


def plot_rmse_heatmap(data_dict: Dict, output_dir: Path) -> None:
    """
    Tüm variable'lar ve seq_len'ler için RMSE heatmap.
    """
    # RMSE matrix oluştur
    rmse_matrix = np.zeros((len(STATE_NAMES), len(SEQ_LENGTHS)))

    for j, seq_len in enumerate(SEQ_LENGTHS):
        trajectory, ground_truth = data_dict[seq_len]
        for i in range(len(STATE_NAMES)):
            rmse_matrix[i, j] = compute_rmse(trajectory, ground_truth, seq_len, i)

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 12))

    im = ax.imshow(rmse_matrix, cmap='RdYlGn_r', aspect='auto')

    # Axis labels
    ax.set_xticks(range(len(SEQ_LENGTHS)))
    ax.set_xticklabels(SEQ_LENGTHS, fontsize=11)
    ax.set_yticks(range(len(STATE_NAMES)))
    ax.set_yticklabels(STATE_NAMES, fontsize=10)

    ax.set_xlabel('Sequence Length (seq_len)', fontsize=12)
    ax.set_ylabel('State Variable', fontsize=12)
    ax.set_title('RMSE Heatmap: All Variables × All Sequence Lengths\n(Green=Low Error, Red=High Error)',
                 fontsize=13, fontweight='bold')

    # Değerleri hücrelere yaz
    for i in range(len(STATE_NAMES)):
        for j in range(len(SEQ_LENGTHS)):
            val = rmse_matrix[i, j]
            color = 'white' if val > 0.3 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('RMSE', fontsize=11)

    # 0.5 threshold line
    ax.axhline(y=-0.5, color='orange', linewidth=2)

    plt.tight_layout()

    save_path = output_dir / "rmse_heatmap.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")

    # RMSE tablosu yazdır
    print("\n" + "="*70)
    print("RMSE TABLE")
    print("="*70)
    header = f"{'Variable':<12}" + "".join([f"seq={s:<6}" for s in SEQ_LENGTHS])
    print(header)
    print("-"*70)
    for i, name in enumerate(STATE_NAMES):
        row = f"{name:<12}" + "".join([f"{rmse_matrix[i,j]:<8.4f}" for j in range(len(SEQ_LENGTHS))])
        print(row)


def plot_summary_comparison(data_dict: Dict, output_dir: Path) -> None:
    """
    Özet karşılaştırma: 4 key variable için 5 seq_len.
    """
    key_vars = [0, 2, 6, 8]  # nH2_g, nCH4_g, SO4, X

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))

    _, ground_truth = data_dict[50]
    t = np.arange(len(ground_truth))

    for i, var_idx in enumerate(key_vars):
        for j, seq_len in enumerate(SEQ_LENGTHS):
            ax = axes[i, j]
            trajectory, gt = data_dict[seq_len]

            rmse = compute_rmse(trajectory, gt, seq_len, var_idx)

            ax.plot(t, gt[:, var_idx], 'b-', linewidth=1.5, alpha=0.8)
            ax.plot(t, trajectory[:, var_idx], 'r--', linewidth=1.5, alpha=0.8)
            ax.axvline(x=seq_len, color='green', linestyle=':', alpha=0.7)

            # İlk satıra seq_len başlıkları
            if i == 0:
                ax.set_title(f'seq_len={seq_len}', fontsize=12, fontweight='bold')

            # İlk sütuna variable isimleri
            if j == 0:
                ax.set_ylabel(STATE_NAMES[var_idx], fontsize=11, fontweight='bold')

            # Son satıra x label
            if i == 3:
                ax.set_xlabel('Time Step', fontsize=10)

            # RMSE'yi subplot içine yaz
            rmse_color = 'red' if rmse > 0.5 else 'darkgreen'
            ax.text(0.95, 0.95, f'RMSE={rmse:.3f}', transform=ax.transAxes,
                    fontsize=9, ha='right', va='top', color=rmse_color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.grid(True, alpha=0.3)

    # Legend
    axes[0, 0].plot([], [], 'b-', linewidth=2, label='Ground Truth')
    axes[0, 0].plot([], [], 'r--', linewidth=2, label='Generated')
    axes[0, 0].legend(fontsize=9, loc='upper right')

    fig.suptitle('Key Variables Comparison: Ground Truth vs Generated\n(Blue=ODE, Red=LSTM)',
                 fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()

    save_path = output_dir / "summary_key_variables.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")


def main():
    # Paths
    base_dir = Path("outputs/seq_len_experiment")
    output_dir = Path("figures/all_variables")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("LOADING TRAJECTORY DATA")
    print("="*70)

    # Verileri yükle
    data_dict = load_all_trajectories(base_dir)

    print("\n" + "="*70)
    print("GENERATING FIGURES FOR EACH VARIABLE")
    print("="*70)

    # Her variable için figür oluştur
    for var_idx in range(len(STATE_NAMES)):
        plot_single_variable(var_idx, data_dict, output_dir)

    print("\n" + "="*70)
    print("GENERATING SUMMARY FIGURES")
    print("="*70)

    # RMSE heatmap
    plot_rmse_heatmap(data_dict, output_dir)

    # Summary comparison
    plot_summary_comparison(data_dict, output_dir)

    print("\n" + "="*70)
    print(f"ALL FIGURES SAVED TO: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
