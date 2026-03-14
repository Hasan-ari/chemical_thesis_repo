"""
Synthetic Data Generation for LSTM Training
============================================
Generates uniformly-spaced time series data using the two-phase anaerobic model
with best-fit parameters from Basalt @ 25C experiments.

Usage:
    python -m lstm_synth_data.generate --n_points 500
    python -m lstm_synth_data.generate --n_points 500 --t_end 20.0 --output data/output
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from .ode_model import model_mixed, compute_rates, speciate_sulfide, STATE_NAMES, OUTPUT_NAMES
from .params import load_basalt_25c_params, get_experimental_data


def generate_synthetic_data(
    n_points: int = 500,
    t_start: float = 0.0,
    t_end: float = 19.0,
    mat_file: Path = None,
    verbose: bool = True
) -> dict:
    """
    Generate synthetic data with uniform time spacing.

    Args:
        n_points: Number of time points to generate
        t_start: Start time (days)
        t_end: End time (days)
        mat_file: Path to .mat file with best-fit parameters
        verbose: Print progress information

    Returns:
        Dictionary with time array, state arrays, and rates
    """
    if verbose:
        print(f"Loading parameters for Basalt @ 25C...")

    # Load parameters
    p_fit, env, y0 = load_basalt_25c_params(mat_file)

    if verbose:
        print(f"  - Loaded {len(p_fit)} parameters")
        print(f"  - Environment: T={env.T}K, Vg={env.Vg}L, Vl={env.Vl}L")
        print(f"  - Initial H2 gas: {y0[0]:.3f} mmol")

    # Create uniform time grid
    t_eval = np.linspace(t_start, t_end, n_points)
    dt = t_eval[1] - t_eval[0]

    if verbose:
        print(f"\nGenerating {n_points} data points...")
        print(f"  - Time range: [{t_start}, {t_end}] days")
        print(f"  - Time step: {dt:.6f} days ({dt*24*60:.2f} minutes)")

    # Define ODE function for solver
    def ode_func(t, y):
        return model_mixed(t, y, p_fit, env)

    # Solve ODE with uniform output
    sol = solve_ivp(
        ode_func,
        t_span=(t_start, t_end),
        y0=y0,
        method='Radau',  # Good for stiff systems (like MATLAB's ode15s)
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.5
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    if verbose:
        print(f"  - Solver status: {sol.message}")
        print(f"  - Output shape: {sol.y.shape}")

    # Extract results
    t_sim = sol.t
    y_sim = sol.y.T  # Shape: (n_points, 14)

    # Compute derived quantities
    pH_vals = np.array([env.pH_fun(t) for t in t_sim])
    H2S_aq, HS_aq = speciate_sulfide(y_sim[:, 11], pH_vals, env.pKa_H2S)

    # Compute reaction rates at each time point
    rates = np.zeros((len(t_sim), 4))
    for i in range(len(t_sim)):
        rates[i] = compute_rates(t_sim[i], y_sim[i], p_fit, env)

    if verbose:
        print("\nData generation complete!")
        print(f"  - Final H2 gas: {y_sim[-1, 0]:.3f} mmol")
        print(f"  - Final CH4 gas: {y_sim[-1, 2]:.3f} mmol")
        print(f"  - Final SO4: {y_sim[-1, 6]:.3f} mM")

    return {
        'time': t_sim,
        'states': y_sim,
        'H2S_aq': H2S_aq,
        'HS_aq': HS_aq,
        'rates': rates,
        'pH': pH_vals,
        'params': p_fit,
        'env': env,
        'y0': y0,
        'dt': dt,
        'n_points': n_points
    }


def save_data(data: dict, output_dir: Path, prefix: str = "basalt_25c"):
    """
    Save generated data in multiple formats.

    Args:
        data: Dictionary from generate_synthetic_data
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create full output array (time + states + derived + rates)
    full_data = np.column_stack([
        data['time'],
        data['states'],
        data['H2S_aq'],
        data['HS_aq'],
        data['rates']
    ])

    # Column names
    columns = ['Time'] + STATE_NAMES + ['H2S_aq', 'HS_aq', 'r_meth', 'r_sulf', 'r_precip', 'r_aceto']

    # Save as CSV
    df = pd.DataFrame(full_data, columns=columns)
    csv_path = output_dir / f"{prefix}_synth_{data['n_points']}pts.csv"
    df.to_csv(csv_path, index=False, float_format='%.8g')
    print(f"Saved CSV: {csv_path}")

    # Save as numpy arrays (for direct LSTM loading)
    npz_path = output_dir / f"{prefix}_synth_{data['n_points']}pts.npz"
    np.savez(
        npz_path,
        time=data['time'],
        states=data['states'],
        H2S_aq=data['H2S_aq'],
        HS_aq=data['HS_aq'],
        rates=data['rates'],
        pH=data['pH'],
        params=data['params'],
        dt=data['dt'],
        state_names=STATE_NAMES
    )
    print(f"Saved NPZ: {npz_path}")

    # Save LSTM-ready format (states only, normalized later by LSTM code)
    lstm_path = output_dir / f"{prefix}_lstm_input_{data['n_points']}pts.npy"
    np.save(lstm_path, data['states'])
    print(f"Saved LSTM input: {lstm_path}")

    return csv_path, npz_path, lstm_path


def plot_results(data: dict, exp_data: dict = None, output_dir: Path = None):
    """
    Create diagnostic plots comparing model output to experimental data.

    Args:
        data: Dictionary from generate_synthetic_data
        exp_data: Experimental data dictionary (optional)
        output_dir: Directory to save plots (optional)
    """
    t_sim = data['time']
    y_sim = data['states']
    rates = data['rates']

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Basalt @ 25C - Synthetic Data ({data["n_points"]} points)', fontsize=14)

    # Row 1: Gas phase
    gas_labels = ['nH2_g (mmol)', 'nCO2_g (mmol)', 'nCH4_g (mmol)', 'nH2S_g (mmol)']
    exp_keys = ['nH2_g_exp', 'nCO2_g_exp', 'nCH4_g_exp', 'nH2S_g_exp']

    for i, (label, exp_key) in enumerate(zip(gas_labels, exp_keys)):
        ax = axes[0, i]
        ax.plot(t_sim, y_sim[:, i], 'b-', label='Model', linewidth=1)
        if exp_data is not None:
            ax.plot(exp_data['t_exp'], exp_data[exp_key], 'ko', label='Exp', markersize=6)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Row 2: Aqueous phase
    aq_vars = [(4, 'H2_aq (mM)'), (5, 'CO2_aq (mM)'), (6, 'SO4 (mM)'), (11, 'S_tot (mM)')]
    for i, (idx, label) in enumerate(aq_vars):
        ax = axes[1, i]
        ax.plot(t_sim, y_sim[:, idx], 'b-', linewidth=1)
        if i == 2 and exp_data is not None:  # SO4
            ax.plot(exp_data['t_exp'], exp_data['SO4_exp'], 'ko', markersize=6)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    # Row 3: Rates and biomass
    ax = axes[2, 0]
    ax.plot(t_sim, y_sim[:, 8], 'g-', linewidth=1)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Biomass X (mM)')
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(t_sim, rates[:, 0], 'r-', label='r_meth', linewidth=1)
    ax.plot(t_sim, rates[:, 1], 'b-', label='r_sulf', linewidth=1)
    ax.plot(t_sim, rates[:, 3], 'm-', label='r_aceto', linewidth=1)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Rates (mM/day)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    ax.plot(t_sim, data['pH'], 'k-', linewidth=1)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('pH')
    ax.grid(True, alpha=0.3)

    ax = axes[2, 3]
    ax.plot(t_sim, y_sim[:, 13], 'brown', linewidth=1)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Fe_pool (mM)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir is not None:
        plot_path = Path(output_dir) / f"basalt_25c_synth_{data['n_points']}pts.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {plot_path}")

    plt.show()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic data for LSTM training (Basalt @ 25C)'
    )
    parser.add_argument(
        '--n_points', type=int, default=500,
        help='Number of time points to generate (default: 500)'
    )
    parser.add_argument(
        '--t_start', type=float, default=0.0,
        help='Start time in days (default: 0.0)'
    )
    parser.add_argument(
        '--t_end', type=float, default=19.0,
        help='End time in days (default: 19.0)'
    )
    parser.add_argument(
        '--output', type=str, default='data/output',
        help='Output directory (default: data/output)'
    )
    parser.add_argument(
        '--mat_file', type=str, default=None,
        help='Path to .mat file with best-fit parameters'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Show diagnostic plots'
    )
    parser.add_argument(
        '--no_save', action='store_true',
        help='Do not save output files'
    )

    args = parser.parse_args()

    # Generate data
    data = generate_synthetic_data(
        n_points=args.n_points,
        t_start=args.t_start,
        t_end=args.t_end,
        mat_file=Path(args.mat_file) if args.mat_file else None,
        verbose=True
    )

    # Save data
    if not args.no_save:
        output_dir = Path(args.output)
        save_data(data, output_dir)

    # Plot if requested
    if args.plot:
        exp_data = get_experimental_data()
        plot_results(data, exp_data, output_dir if not args.no_save else None)

    print("\nDone!")


if __name__ == '__main__':
    main()
