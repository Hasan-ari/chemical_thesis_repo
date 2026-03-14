"""
============================================================================
SYNTHETIC DATA GENERATOR FOR LSTM TRAINING
============================================================================

Generates synthetic training data for the two-phase anaerobic model using
best-fit parameters from Basalt 25C experiments.

Outputs:
- data/synthetic_basalt_25c.csv (time + 14 states + pH = 16 columns)
- data/synthetic_basalt_25c.npy (same data in numpy format)
- results/synthetic_data_visualization.png

Author: Chemical Thesis Project
Date: 2026-W04
============================================================================
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from ode_model import model_mixed, STATE_NAMES


def load_parameters(mat_file_path):
    """
    Load best-fit parameters and environment from MATLAB .mat file.

    Parameters
    ----------
    mat_file_path : str
        Path to best_fit_params_Basalt_25C.mat

    Returns
    -------
    p_fit : ndarray
        Parameter vector (24 elements)
    env_struct : object
        MATLAB struct with environmental parameters
    """
    mat = sio.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
    p_fit = mat['p_fit']
    env_struct = mat['env']
    return p_fit, env_struct


def load_experimental_data(txt_file_path):
    """
    Load experimental data from text file.

    Data columns:
    0: time (days)
    1: H2 (umol)
    2: CO2 (umol)
    3: CH4 (umol)
    4: H2S (umol)
    5: pH
    6: SO4 (mM)

    Parameters
    ----------
    txt_file_path : str
        Path to Muller_2024_H2_Basalt_at_25C.txt

    Returns
    -------
    df : DataFrame
        Experimental data
    """
    df = pd.read_csv(txt_file_path, sep=r'\s+', comment='%', header=None, encoding='latin1')
    df.columns = ['time', 'H2_umol', 'CO2_umol', 'CH4_umol', 'H2S_umol', 'pH', 'SO4_mM']
    return df


def create_environment(env_struct, pH_fun):
    """
    Create environment dictionary from MATLAB struct.

    Parameters
    ----------
    env_struct : object
        MATLAB struct with environmental parameters
    pH_fun : callable
        Interpolation function for pH(t)

    Returns
    -------
    env : dict
        Environment parameters dictionary
    """
    env = {
        'Vg': float(env_struct.Vg),
        'Vl': float(env_struct.Vl),
        'T': float(env_struct.T),
        'Rgas': float(env_struct.Rgas),
        'Hcp_H2_eff': float(env_struct.Hcp_H2_eff),
        'Hcp_CO2_eff': float(env_struct.Hcp_CO2_eff),
        'Hcp_H2S_eff': float(env_struct.Hcp_H2S_eff),
        'pKa_H2S': float(env_struct.pKa_H2S),
        'SO4_sat_gyp': float(env_struct.SO4_sat_gyp),
        'pH_fun': pH_fun
    }
    return env


def compute_initial_conditions(df, env):
    """
    Compute initial conditions from first experimental data point.

    Parameters
    ----------
    df : DataFrame
        Experimental data
    env : dict
        Environment parameters

    Returns
    -------
    y0 : ndarray
        Initial state vector (14 elements)
    """
    # Gas phase amounts (convert umol to mmol)
    nH2_g_0 = df.iloc[0]['H2_umol'] / 1000.0
    nCO2_g_0 = df.iloc[0]['CO2_umol'] / 1000.0
    nCH4_g_0 = df.iloc[0]['CH4_umol'] / 1000.0
    nH2S_g_0 = df.iloc[0]['H2S_umol'] / 1000.0

    # Partial pressures for equilibrium calculations
    pH2_0 = (nH2_g_0 / 1000) * env['Rgas'] * env['T'] / env['Vg']
    pCO2_0 = (nCO2_g_0 / 1000) * env['Rgas'] * env['T'] / env['Vg']

    # Aqueous concentrations at equilibrium with gas phase
    H2_aq_0 = env['Hcp_H2_eff'] * pH2_0
    CO2_aq_0 = env['Hcp_CO2_eff'] * pCO2_0

    # Sulfate from experimental data
    SO4_0 = df.iloc[0]['SO4_mM']

    # Initial conditions for remaining states
    FeS_0 = 0.01       # Small initial FeS
    X_0 = 0.01         # Small initial biomass
    Acetate_0 = 0.0    # No initial acetate
    HCO3_0 = 0.0       # Bicarbonate (held constant at 0)
    S_tot_0 = 1.0      # Initial dissolved sulfide
    Lag_0 = 0.0        # Lag indicator starts at 0
    Fe_pool_0 = 0.10   # Initial available iron

    y0 = np.array([
        nH2_g_0, nCO2_g_0, nCH4_g_0, nH2S_g_0,
        H2_aq_0, CO2_aq_0, SO4_0, FeS_0,
        X_0, Acetate_0, HCO3_0, S_tot_0, Lag_0, Fe_pool_0
    ])

    return y0


def generate_synthetic_data(p_fit, env, y0, t_start=0.0, t_end=19.0, n_points=500):
    """
    Generate synthetic data by solving the ODE system.

    Parameters
    ----------
    p_fit : ndarray
        Model parameters
    env : dict
        Environment parameters
    y0 : ndarray
        Initial conditions
    t_start : float
        Start time (days)
    t_end : float
        End time (days)
    n_points : int
        Number of time points

    Returns
    -------
    t_eval : ndarray
        Time points
    data : ndarray
        Solution array (n_points x 14)
    """
    t_eval = np.linspace(t_start, t_end, n_points)

    print(f"Solving ODE system from t={t_start} to t={t_end} with {n_points} points...")
    print(f"Using BDF method for stiff ODEs...")

    sol = solve_ivp(
        lambda t, y: model_mixed(t, y, p_fit, env),
        [t_start, t_end],
        y0,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-6,
        atol=1e-9
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    print(f"ODE solver converged successfully!")
    print(f"Number of function evaluations: {sol.nfev}")

    return t_eval, sol.y.T


def save_data(t_eval, data, pH_vals, output_dir):
    """
    Save synthetic data to CSV and NPY files.

    CSV format: time, nH2_g, nCO2_g, ..., Fe_pool, pH (16 columns)
    NPY format: same structure

    Parameters
    ----------
    t_eval : ndarray
        Time points
    data : ndarray
        State variable data (n_points x 14)
    pH_vals : ndarray
        pH values at each time point
    output_dir : str
        Output directory path
    """
    # Combine time, states, and pH
    full_data = np.column_stack([t_eval, data, pH_vals])

    # Column names
    columns = ['time'] + STATE_NAMES + ['pH']

    # Save as CSV
    csv_path = os.path.join(output_dir, 'synthetic_basalt_25c.csv')
    df = pd.DataFrame(full_data, columns=columns)
    df.to_csv(csv_path, index=False, float_format='%.8e')
    print(f"Saved CSV: {csv_path}")
    print(f"  Shape: {full_data.shape} (time + 14 states + pH)")

    # Save as NPY
    npy_path = os.path.join(output_dir, 'synthetic_basalt_25c.npy')
    np.save(npy_path, full_data)
    print(f"Saved NPY: {npy_path}")

    return csv_path, npy_path


def create_visualization(t_eval, data, pH_vals, exp_df, output_dir):
    """
    Create visualization comparing model trajectories with experimental data.

    Parameters
    ----------
    t_eval : ndarray
        Time points
    data : ndarray
        State variable data (n_points x 14)
    pH_vals : ndarray
        pH values
    exp_df : DataFrame
        Experimental data
    output_dir : str
        Output directory path
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle('Synthetic Data vs Experimental Points - Basalt 25C', fontsize=14, fontweight='bold')

    # Convert model gas phase to umol for comparison
    model_H2_umol = data[:, 0] * 1000   # mmol -> umol
    model_CO2_umol = data[:, 1] * 1000
    model_CH4_umol = data[:, 2] * 1000
    model_H2S_umol = data[:, 3] * 1000

    t_exp = exp_df['time'].values

    # Plot 1: H2 gas
    ax = axes[0, 0]
    ax.plot(t_eval, model_H2_umol, 'b-', linewidth=1.5, label='Model')
    ax.scatter(t_exp, exp_df['H2_umol'], c='red', s=50, zorder=5, label='Experimental')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('H2 (umol)')
    ax.set_title('Hydrogen Gas')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: CO2 gas
    ax = axes[0, 1]
    ax.plot(t_eval, model_CO2_umol, 'b-', linewidth=1.5, label='Model')
    ax.scatter(t_exp, exp_df['CO2_umol'], c='red', s=50, zorder=5, label='Experimental')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('CO2 (umol)')
    ax.set_title('Carbon Dioxide Gas')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: CH4 gas
    ax = axes[0, 2]
    ax.plot(t_eval, model_CH4_umol, 'b-', linewidth=1.5, label='Model')
    ax.scatter(t_exp, exp_df['CH4_umol'], c='red', s=50, zorder=5, label='Experimental')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('CH4 (umol)')
    ax.set_title('Methane Gas')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: H2S gas
    ax = axes[1, 0]
    ax.plot(t_eval, model_H2S_umol, 'b-', linewidth=1.5, label='Model')
    ax.scatter(t_exp, exp_df['H2S_umol'], c='red', s=50, zorder=5, label='Experimental')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('H2S (umol)')
    ax.set_title('Hydrogen Sulfide Gas')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: SO4
    ax = axes[1, 1]
    ax.plot(t_eval, data[:, 6], 'b-', linewidth=1.5, label='Model')
    ax.scatter(t_exp, exp_df['SO4_mM'], c='red', s=50, zorder=5, label='Experimental')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('SO4 (mM)')
    ax.set_title('Sulfate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: pH
    ax = axes[1, 2]
    ax.plot(t_eval, pH_vals, 'b-', linewidth=1.5, label='Interpolated')
    ax.scatter(t_exp, exp_df['pH'], c='red', s=50, zorder=5, label='Experimental')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('pH')
    ax.set_title('pH')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 7: Biomass
    ax = axes[2, 0]
    ax.plot(t_eval, data[:, 8], 'g-', linewidth=1.5, label='Biomass (X)')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('X (a.u.)')
    ax.set_title('Biomass')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 8: Acetate
    ax = axes[2, 1]
    ax.plot(t_eval, data[:, 9], 'purple', linewidth=1.5, label='Acetate')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Acetate (mM)')
    ax.set_title('Acetate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 9: FeS and S_tot
    ax = axes[2, 2]
    ax.plot(t_eval, data[:, 7], 'k-', linewidth=1.5, label='FeS')
    ax.plot(t_eval, data[:, 11], 'orange', linewidth=1.5, label='S_tot')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Concentration (mM)')
    ax.set_title('Sulfide Species')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, 'synthetic_data_visualization.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {fig_path}")
    plt.close()

    return fig_path


def main():
    """Main function to generate synthetic data."""
    print("=" * 70)
    print("SYNTHETIC DATA GENERATOR FOR LSTM TRAINING")
    print("=" * 70)
    print()

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))  # 2026_W04_Lstm_training_v1
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')

    # Source data paths
    mat_file = '/Users/macbook/chemical_thesis/chemical_thesis_repo/2026-W01_model_anlama/code/matlab/best_fit_params_Basalt_25C.mat'
    txt_file = '/Users/macbook/chemical_thesis/chemical_thesis_repo/2026-W01_model_anlama/code/matlab/Muller_2024_H2_Basalt_at_25C.txt'

    # Ensure output directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading parameters from: {mat_file}")
    print(f"Loading experimental data from: {txt_file}")
    print()

    # Step 1: Load parameters and experimental data
    print("[1/5] Loading parameters and experimental data...")
    p_fit, env_struct = load_parameters(mat_file)
    exp_df = load_experimental_data(txt_file)
    print(f"  Loaded {len(p_fit)} parameters")
    print(f"  Loaded {len(exp_df)} experimental time points")

    # Step 2: Create pH interpolation function
    print("\n[2/5] Creating pH interpolation function...")
    t_exp = exp_df['time'].values
    pH_exp = exp_df['pH'].values
    pH_fun = interp1d(t_exp, pH_exp, kind='linear', fill_value='extrapolate')
    print(f"  pH range: {pH_exp.min():.2f} - {pH_exp.max():.2f}")

    # Step 3: Set up environment and initial conditions
    print("\n[3/5] Setting up environment and initial conditions...")
    env = create_environment(env_struct, pH_fun)
    y0 = compute_initial_conditions(exp_df, env)
    print(f"  Environment: Vg={env['Vg']:.4f} L, Vl={env['Vl']:.4f} L, T={env['T']:.1f} K")
    print(f"  Initial H2: {y0[0]:.4f} mmol ({y0[0]*1000:.1f} umol)")
    print(f"  Initial SO4: {y0[6]:.2f} mM")

    # Step 4: Generate synthetic data
    print("\n[4/5] Generating synthetic data...")
    t_eval, data = generate_synthetic_data(
        p_fit, env, y0,
        t_start=0.0,
        t_end=19.0,
        n_points=500
    )
    pH_vals = pH_fun(t_eval)
    print(f"  Generated {len(t_eval)} time points over {t_eval[-1]:.1f} days")
    print(f"  Final H2: {data[-1, 0]*1000:.1f} umol (exp: {exp_df.iloc[-1]['H2_umol']:.1f})")
    print(f"  Final CH4: {data[-1, 2]*1000:.1f} umol (exp: {exp_df.iloc[-1]['CH4_umol']:.1f})")

    # Step 5: Save data and create visualization
    print("\n[5/5] Saving data and creating visualization...")
    csv_path, npy_path = save_data(t_eval, data, pH_vals, data_dir)
    fig_path = create_visualization(t_eval, data, pH_vals, exp_df, results_dir)

    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - {csv_path}")
    print(f"  - {npy_path}")
    print(f"  - {fig_path}")
    print()
    print(f"Data format:")
    print(f"  - Columns: time, {', '.join(STATE_NAMES)}, pH")
    print(f"  - Shape: (500, 16)")
    print()


if __name__ == '__main__':
    main()
