"""
================================================================================
N_POINTS vs SEQ_LEN - SONUÇ GÖRSELLEŞTİRME
================================================================================

Kullanım (Colab'da):
    from plot_results import plot_all_figures
    plot_all_figures("outputs_colab")

Veya:
    !python plot_results.py outputs_colab
================================================================================
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


STATE_NAMES = [
    "nH2_g", "nCO2_g", "nCH4_g", "nH2S_g",
    "H2_aq", "CO2_aq", "SO4", "FeS",
    "X", "Acetate", "HCO3", "S_tot",
    "Lag", "Fe_pool"
]


def load_all_results(output_dir: str) -> Dict[int, List[Dict]]:
    """all_results.json'dan sonuçları yükle."""
    results_path = Path(output_dir) / "all_results.json"
    with open(results_path) as f:
        data = json.load(f)
    # String keys → int keys
    return {int(k): v for k, v in data.items()}


def plot_scaling_analysis(all_results: Dict[int, List[Dict]], output_dir: Path):
    """Ana scaling analizi figürü."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    n_points_list = sorted(all_results.keys())
    colors = {500: 'blue', 1000: 'orange', 2500: 'green'}

    # Plot 1: RMSE vs seq_len
    ax1 = axes[0, 0]
    for n_points in n_points_list:
        results = sorted(all_results[n_points], key=lambda x: x["seq_len"], reverse=True)
        seq_lens = [r["seq_len"] for r in results]
        rmses = [r["rmse_total"] for r in results]
        ax1.plot(seq_lens, rmses, 'o-', label=f'n={n_points}',
                 color=colors.get(n_points, 'gray'), linewidth=2, markersize=10)

    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax1.set_xlabel('Sequence Length (seq_len)', fontsize=12)
    ax1.set_ylabel('Trajectory RMSE', fontsize=12)
    ax1.set_title('RMSE vs Sequence Length', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_xticks([2, 3, 5, 10, 20, 30, 50])
    ax1.set_xticklabels([2, 3, 5, 10, 20, 30, 50])

    # Plot 2: Training loss vs seq_len
    ax2 = axes[0, 1]
    for n_points in n_points_list:
        results = sorted(all_results[n_points], key=lambda x: x["seq_len"], reverse=True)
        seq_lens = [r["seq_len"] for r in results]
        losses = [r["final_loss"] for r in results]
        ax2.plot(seq_lens, losses, 'o-', label=f'n={n_points}',
                 color=colors.get(n_points, 'gray'), linewidth=2, markersize=10)

    ax2.set_xlabel('Sequence Length (seq_len)', fontsize=12)
    ax2.set_ylabel('Final Training Loss', fontsize=12)
    ax2.set_title('Training Loss vs Sequence Length', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_xticks([2, 3, 5, 10, 20, 30, 50])
    ax2.set_xticklabels([2, 3, 5, 10, 20, 30, 50])

    # Plot 3: n_samples vs seq_len
    ax3 = axes[1, 0]
    for n_points in n_points_list:
        results = sorted(all_results[n_points], key=lambda x: x["seq_len"], reverse=True)
        seq_lens = [r["seq_len"] for r in results]
        samples = [r["n_samples"] for r in results]
        ax3.plot(seq_lens, samples, 'o-', label=f'n={n_points}',
                 color=colors.get(n_points, 'gray'), linewidth=2, markersize=10)

    ax3.set_xlabel('Sequence Length (seq_len)', fontsize=12)
    ax3.set_ylabel('Number of Training Samples', fontsize=12)
    ax3.set_title('Training Samples vs Sequence Length', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_xticks([2, 3, 5, 10, 20, 30, 50])
    ax3.set_xticklabels([2, 3, 5, 10, 20, 30, 50])

    # Plot 4: SCALING LAW - Min successful seq_len vs n_points
    ax4 = axes[1, 1]
    min_seq_lens = []
    for n_points in n_points_list:
        results = all_results[n_points]
        good = [r for r in results if r["rmse_total"] < 0.5]
        if good:
            min_seq = min(r["seq_len"] for r in good)
        else:
            min_seq = None
        min_seq_lens.append((n_points, min_seq))

    valid = [(n, s) for n, s in min_seq_lens if s is not None]
    if valid:
        ns, ss = zip(*valid)
        bars = ax4.bar(range(len(ns)), ss, color=[colors.get(n, 'gray') for n in ns],
                       alpha=0.8, edgecolor='black', linewidth=2)
        ax4.set_xticks(range(len(ns)))
        ax4.set_xticklabels([str(n) for n in ns], fontsize=12)

        # Değerleri bar üstüne yaz
        for bar, s in zip(bars, ss):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(s), ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax4.set_xlabel('Number of Data Points (n_points)', fontsize=12)
    ax4.set_ylabel('Minimum Successful seq_len', fontsize=12)
    ax4.set_title('SCALING LAW: n_points vs min_seq_len\n(RMSE < 0.5)',
                  fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_path = output_dir / "scaling_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_rmse_heatmap(all_results: Dict[int, List[Dict]], output_dir: Path):
    """RMSE heatmap: n_points × seq_len."""

    n_points_list = sorted(all_results.keys())
    all_seq_lens = sorted(set(
        r["seq_len"] for results in all_results.values() for r in results
    ), reverse=True)

    # Matrix oluştur
    matrix = np.full((len(n_points_list), len(all_seq_lens)), np.nan)
    for i, n_points in enumerate(n_points_list):
        for r in all_results[n_points]:
            j = all_seq_lens.index(r["seq_len"])
            matrix[i, j] = r["rmse_total"]

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1.5)

    ax.set_xticks(range(len(all_seq_lens)))
    ax.set_xticklabels(all_seq_lens, fontsize=12)
    ax.set_yticks(range(len(n_points_list)))
    ax.set_yticklabels(n_points_list, fontsize=12)

    ax.set_xlabel('Sequence Length (seq_len)', fontsize=13)
    ax.set_ylabel('Number of Data Points (n_points)', fontsize=13)
    ax.set_title('RMSE Heatmap: n_points × seq_len\n(Green ≤ 0.5 = Good, Red > 0.5 = Bad)',
                 fontsize=14, fontweight='bold')

    # Değerleri hücrelere yaz
    for i in range(len(n_points_list)):
        for j in range(len(all_seq_lens)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.4 else 'black'
                status = "✓" if val < 0.5 else "✗"
                ax.text(j, i, f'{val:.2f}\n{status}', ha='center', va='center',
                       fontsize=11, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='RMSE', shrink=0.8)

    plt.tight_layout()

    save_path = output_dir / "rmse_heatmap.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_trajectories_comparison(all_results: Dict[int, List[Dict]], output_dir: Path):
    """Her n_points için key variables trajectory karşılaştırması."""

    key_vars = [0, 2, 6, 8]  # nH2_g, nCH4_g, SO4, X
    var_names = [STATE_NAMES[i] for i in key_vars]

    for n_points in sorted(all_results.keys()):
        results = sorted(all_results[n_points], key=lambda x: x["seq_len"], reverse=True)
        n_exps = len(results)

        fig, axes = plt.subplots(len(key_vars), n_exps, figsize=(4*n_exps, 3.5*len(key_vars)))

        for j, r in enumerate(results):
            seq_len = r["seq_len"]

            # Trajectory yükle
            traj_path = output_dir / f"n{n_points}" / f"seq_len_{seq_len}" / "trajectory.npz"
            if not traj_path.exists():
                continue

            data = np.load(traj_path)
            trajectory = data["trajectory_orig"]
            ground_truth = data["ground_truth_orig"]
            t = np.arange(len(trajectory))

            for i, var_idx in enumerate(key_vars):
                ax = axes[i, j] if n_exps > 1 else axes[i]

                ax.plot(t, ground_truth[:, var_idx], 'b-', linewidth=1.5,
                       label='Ground Truth', alpha=0.8)
                ax.plot(t, trajectory[:, var_idx], 'r--', linewidth=1.5,
                       label='Generated', alpha=0.8)
                ax.axvline(x=seq_len, color='green', linestyle=':', alpha=0.6)

                if i == 0:
                    rmse_color = 'green' if r["rmse_total"] < 0.5 else 'red'
                    ax.set_title(f'seq_len={seq_len}\nRMSE={r["rmse_total"]:.3f}',
                               fontsize=11, fontweight='bold', color=rmse_color)
                if j == 0:
                    ax.set_ylabel(var_names[i], fontsize=11)
                if i == len(key_vars) - 1:
                    ax.set_xlabel('Time Step', fontsize=10)
                if i == 0 and j == 0:
                    ax.legend(fontsize=8)

                ax.grid(True, alpha=0.3)

        fig.suptitle(f'n_points = {n_points}: Key Variables Comparison',
                    fontsize=14, fontweight='bold', y=1.01)

        plt.tight_layout()

        save_path = output_dir / f"trajectories_n{n_points}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")


def plot_rmse_per_variable(all_results: Dict[int, List[Dict]], output_dir: Path):
    """Her değişken için RMSE karşılaştırması."""

    n_points_list = sorted(all_results.keys())

    fig, axes = plt.subplots(2, 7, figsize=(24, 8))
    axes = axes.flatten()

    colors = {500: 'blue', 1000: 'orange', 2500: 'green'}

    for var_idx, var_name in enumerate(STATE_NAMES):
        ax = axes[var_idx]

        for n_points in n_points_list:
            results = sorted(all_results[n_points], key=lambda x: x["seq_len"], reverse=True)
            seq_lens = [r["seq_len"] for r in results]
            rmses = [r["rmse_per_var"][var_idx] for r in results]

            ax.plot(seq_lens, rmses, 'o-', label=f'n={n_points}',
                   color=colors.get(n_points, 'gray'), linewidth=2, markersize=6)

        ax.set_title(var_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('seq_len', fontsize=10)
        ax.set_ylabel('RMSE', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        if var_idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle('RMSE per Variable: All n_points × seq_len Combinations',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    save_path = output_dir / "rmse_per_variable.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def print_summary_table(all_results: Dict[int, List[Dict]]):
    """Özet tablo yazdır."""

    print("\n" + "=" * 70)
    print("SCALING LAW SUMMARY")
    print("=" * 70)

    for n_points in sorted(all_results.keys()):
        print(f"\nn_points = {n_points}:")
        print(f"  {'seq_len':<10} {'RMSE':<12} {'Status':<10}")
        print(f"  {'-'*32}")

        results = sorted(all_results[n_points], key=lambda x: x["seq_len"], reverse=True)
        for r in results:
            status = "✓ Good" if r["rmse_total"] < 0.5 else "✗ Bad"
            print(f"  {r['seq_len']:<10} {r['rmse_total']:<12.4f} {status:<10}")

        good = [r for r in results if r["rmse_total"] < 0.5]
        if good:
            min_seq = min(r["seq_len"] for r in good)
            print(f"  → Minimum successful seq_len: {min_seq}")
        else:
            print(f"  → No successful seq_len!")

    print("\n" + "=" * 70)


def plot_all_figures(output_dir: str = "outputs_colab"):
    """Tüm figürleri oluştur."""

    output_path = Path(output_dir)

    print("=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    # Load results
    all_results = load_all_results(output_dir)
    print(f"✓ Loaded results for n_points: {list(all_results.keys())}")

    # Generate figures
    plot_scaling_analysis(all_results, output_path)
    plot_rmse_heatmap(all_results, output_path)
    plot_trajectories_comparison(all_results, output_path)
    plot_rmse_per_variable(all_results, output_path)

    # Print summary
    print_summary_table(all_results)

    print(f"\n✓ All figures saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "outputs_colab"

    plot_all_figures(output_dir)
