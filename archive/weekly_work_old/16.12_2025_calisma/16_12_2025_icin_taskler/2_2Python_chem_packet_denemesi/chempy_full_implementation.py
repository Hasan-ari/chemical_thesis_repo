# -*- coding: utf-8 -*-
"""
Full ChemPy Implementation
Uses ChemPy library to define reactions and handle stoichiometry automatically
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from collections import defaultdict
import warnings

try:
    from chempy import ReactionSystem, Substance
    from chempy.kinetics.ode import get_odesys
    CHEMPY_AVAILABLE = True
except ImportError:
    CHEMPY_AVAILABLE = False
    print("WARNING: ChemPy not installed!")
    print("Install with: pip install chempy")

warnings.filterwarnings("ignore")


def load_experimental_data(file_path):
    """Load and preprocess experimental data"""
    print(f"\nLoading data from: {file_path}")

    try:
        raw_data = np.loadtxt(file_path, comments='%', encoding='latin1')
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    t_exp = raw_data[:, 0]

    # Unit Conversion (µmol -> mM) + molar for ChemPy
    data_exp = np.column_stack((
        raw_data[:, 1] * 1e-6,  # H2 (convert to M for ChemPy)
        raw_data[:, 2] * 1e-6,  # CO2
        raw_data[:, 3] * 1e-6,  # CH4
        raw_data[:, 4] * 1e-6,  # H2S
        raw_data[:, 6] * 1e-3   # SO4 (already mM, convert to M)
    ))

    print(f"Data loaded: {len(t_exp)} time points")
    print(f"Time range: {t_exp[0]:.1f} to {t_exp[-1]:.1f} days")

    return t_exp, data_exp


def create_chempy_reaction_system():
    """
    Define the reaction system using ChemPy

    Reactions:
    1. Methanogenesis:  4 H2 + CO2 -> CH4 + 2 H2O
    2. Sulfate Reduction: 4 H2 + SO4 -> H2S + 4 H2O
    (Simplified - no acetogenesis for now)
    """

    if not CHEMPY_AVAILABLE:
        raise ImportError("ChemPy is required for this implementation")

    # Define reaction system with rate constants as parameters
    rsys = ReactionSystem.from_string("""
        4 H2 + CO2 -> CH4 + 2 H2O; k_meth
        4 H2 + SO4 -> H2S + 4 H2O; k_sulf
    """)

    print("\n" + "="*60)
    print("CHEMPY REACTION SYSTEM")
    print("="*60)
    print(f"\nNumber of reactions: {len(rsys.rxns)}")
    print(f"Number of substances: {len(rsys.substances)}")

    print("\nReactions:")
    for i, rxn in enumerate(rsys.rxns):
        print(f"  {i+1}. {rxn}")

    print("\nSubstances:")
    for sub in rsys.substances:
        print(f"  {sub}")

    return rsys


def fit_with_chempy(file_path):
    """Fit the ChemPy model to experimental data"""

    # Load data
    t_exp, data_exp = load_experimental_data(file_path)
    if t_exp is None:
        return

    # Create ChemPy reaction system
    rsys = create_chempy_reaction_system()

    # Get ODE system from ChemPy
    odesys, extra = get_odesys(rsys, include_params=False)

    print("\n" + "="*60)
    print("ODE SYSTEM GENERATED")
    print("="*60)
    print(f"State variables: {odesys.names}")
    print(f"Parameters: {odesys.param_names}")

    # Initial conditions (as dict for ChemPy)
    c0_dict = defaultdict(float, {
        'H2': data_exp[0, 0],
        'CO2': data_exp[0, 1],
        'CH4': data_exp[0, 2],
        'H2S': data_exp[0, 3],
        'SO4': data_exp[0, 4],
        'H2O': 55.4  # Constant water concentration (55.4 M)
    })

    print(f"\nInitial conditions (M):")
    for k, v in c0_dict.items():
        if k != 'H2O':
            print(f"  {k}: {v:.6e}")

    # Parameter optimization
    # We'll optimize k_meth and k_sulf
    def residuals_chempy(p_log):
        """Cost function using ChemPy ODE system"""
        try:
            # Use log-scale parameters for better optimization
            k_meth, k_sulf = np.exp(p_log)

            params = {
                'k_meth': k_meth,
                'k_sulf': k_sulf
            }

            # Convert time to seconds (ChemPy uses SI units)
            t_exp_sec = t_exp * 86400  # days to seconds

            # Integrate using ChemPy
            result = odesys.integrate(
                t_exp_sec,
                c0_dict,
                params,
                integrator='cvode',
                atol=1e-9,
                rtol=1e-6
            )

            if not result.info['success']:
                return np.ones(data_exp.size) * 10.0

            # Extract relevant species
            y_sim = np.zeros((len(t_exp), 5))
            for i, name in enumerate(['H2', 'CO2', 'CH4', 'H2S', 'SO4']):
                if name in odesys.names:
                    idx = odesys.names.index(name)
                    y_sim[:, i] = result.yout[:, idx]

            # Calculate weighted residuals
            weights = np.array([2.0, 2.0, 1.0, 0.5, 2.0])

            # Log-scale residuals
            log_sim = np.log1p(y_sim * 1e3)  # Convert back to mM scale
            log_exp = np.log1p(data_exp * 1e3)

            res = (log_sim - log_exp) * weights
            res_flat = res.flatten()

            if np.isnan(res_flat).any() or np.isinf(res_flat).any():
                return np.ones(data_exp.size) * 10.0

            return res_flat

        except Exception as e:
            print(f"Integration error: {e}")
            return np.ones(data_exp.size) * 10.0

    # Initial guess (log scale)
    p0_log = np.log([1e-7, 1e-8])  # k_meth, k_sulf in 1/(M*s)

    # Bounds (log scale)
    lb_log = np.log([1e-12, 1e-12])
    ub_log = np.log([1e-3, 1e-3])

    print("\n" + "="*60)
    print("PARAMETER OPTIMIZATION")
    print("="*60)
    print("\nStarting optimization with ChemPy ODE system...")

    res_opt = least_squares(
        residuals_chempy,
        p0_log,
        bounds=(lb_log, ub_log),
        method='trf',
        verbose=2,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=2000
    )

    p_fit = np.exp(res_opt.x)

    print("\n" + "="*60)
    print("FITTED PARAMETERS (ChemPy)")
    print("="*60)
    print(f"  k_meth: {p_fit[0]:.6e} 1/(M*s)")
    print(f"  k_sulf: {p_fit[1]:.6e} 1/(M*s)")
    print(f"\nOptimization status: {res_opt.message}")
    print(f"Cost function value: {res_opt.cost:.6e}")

    # Final simulation with fitted parameters
    params_fit = {'k_meth': p_fit[0], 'k_sulf': p_fit[1]}

    t_sim_sec = np.linspace(0, t_exp[-1] * 86400, 1000)
    result_final = odesys.integrate(
        t_sim_sec,
        c0_dict,
        params_fit,
        integrator='cvode',
        atol=1e-9,
        rtol=1e-6
    )

    # Extract simulation results
    t_sim = t_sim_sec / 86400  # Convert back to days
    y_sim = np.zeros((len(t_sim), 5))
    for i, name in enumerate(['H2', 'CO2', 'CH4', 'H2S', 'SO4']):
        if name in odesys.names:
            idx = odesys.names.index(name)
            y_sim[:, i] = result_final.yout[:, idx] * 1e3  # Convert to mM

    # Plot results
    species_names = ['H2', 'CO2', 'CH4', 'H2S', 'SO4']
    data_exp_mM = data_exp * 1e3  # Convert to mM for plotting

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i in range(5):
        ax = axes[i]

        # Experimental data
        ax.plot(t_exp, data_exp_mM[:, i], 'ko', markersize=8, label='Experimental', zorder=3)

        # ChemPy model
        ax.plot(t_sim, y_sim[:, i], 'r-', linewidth=2.5, label='ChemPy Model', zorder=2)

        ax.set_title(f'{species_names[i]}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Concentration (mM)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)

    fig.delaxes(axes[5])

    plt.suptitle("ChemPy Automatic Stoichiometry Model", fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = file_path.replace('.txt', '_chempy_full.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.show()

    return p_fit, y_sim, t_sim


if __name__ == "__main__":
    if not CHEMPY_AVAILABLE:
        print("\n" + "="*60)
        print("ERROR: ChemPy is not installed!")
        print("="*60)
        print("\nTo install ChemPy, run:")
        print("  pip install chempy")
        print("\nOr use conda:")
        print("  conda install -c conda-forge chempy")
        print("="*60)
    else:
        file_path = r'D:\chemical_thesis_repo\16.12_2025_calisma\Muller_2024_H2_Sandstone_at_25C.txt'

        print("="*60)
        print("FULL CHEMPY IMPLEMENTATION")
        print("Automatic stoichiometry and ODE generation")
        print("="*60)

        p_fit, y_sim, t_sim = fit_with_chempy(file_path)

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
