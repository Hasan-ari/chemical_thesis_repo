# -*- coding: utf-8 -*-
"""
ChemPy-based Diagnostic Implementation
Compares automatic stoichiometry handling vs. manual implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# PART 1: Load Experimental Data
# ==============================================================================

def load_experimental_data(file_path):
    """Load and preprocess experimental data"""
    print(f"\nLoading data from: {file_path}")

    try:
        raw_data = np.loadtxt(file_path, comments='%', encoding='latin1')
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    # Columns: Time(0), H2(1), CO2(2), CH4(3), H2S(4), pH(5), SO4(6)
    t_exp = raw_data[:, 0]

    # Unit Conversion (µmol -> mM)
    # H2, CO2, CH4, H2S (columns 1-4), SO4 already in mM (column 6)
    data_exp = np.column_stack((
        raw_data[:, 1] * 1e-3,  # H2
        raw_data[:, 2] * 1e-3,  # CO2
        raw_data[:, 3] * 1e-3,  # CH4
        raw_data[:, 4] * 1e-3,  # H2S
        raw_data[:, 6]          # SO4
    ))

    print(f"Data loaded: {len(t_exp)} time points")
    print(f"Time range: {t_exp[0]:.1f} to {t_exp[-1]:.1f} days")
    print(f"\nInitial concentrations (mM):")
    print(f"  H2:  {data_exp[0, 0]:.3f}")
    print(f"  CO2: {data_exp[0, 1]:.3f}")
    print(f"  CH4: {data_exp[0, 2]:.3f}")
    print(f"  H2S: {data_exp[0, 3]:.6f}")
    print(f"  SO4: {data_exp[0, 4]:.3f}")

    return t_exp, data_exp


# ==============================================================================
# PART 2: Simplified Model with Clear Stoichiometry
# ==============================================================================

def simplified_ode_model(t, y, p):
    """
    Simplified 3-guild model with explicit stoichiometry

    Reactions:
    1. Methanogenesis:  4 H2 + CO2 -> CH4 + 2 H2O
    2. Sulfate Reduction: 4 H2 + SO4 -> H2S + 4 H2O
    3. Acetogenesis: 4 H2 + 2 CO2 -> Acetate + 2 H2O
    4. Precipitation: H2S -> FeS (when H2S > threshold)

    State vector y:
    [0] H2, [1] CO2, [2] CH4, [3] H2S, [4] SO4
    """

    # Ensure non-negative concentrations
    y = np.maximum(0, y)

    H2  = y[0]
    CO2 = y[1]
    CH4 = y[2]
    H2S = y[3]
    SO4 = y[4]

    # Parameters
    k_meth = p[0]   # Methanogenesis rate constant
    k_sulf = p[1]   # Sulfate reduction rate constant
    k_aceto = p[2]  # Acetogenesis rate constant (if used)
    KI_H2S = p[3]   # H2S inhibition constant
    k_precip = p[4] # H2S precipitation rate
    H2S_sat = p[5]  # H2S saturation threshold

    # Safety for division
    eps = 1e-9
    H2_safe = max(H2, eps)
    CO2_safe = max(CO2, eps)
    SO4_safe = max(SO4, eps)

    # Inhibition by H2S (competitive inhibition)
    f_inh = KI_H2S / (KI_H2S + H2S + eps)

    # SIMPLE RATE EXPRESSIONS (no complex thermodynamics)
    # Using Monod-like kinetics

    # Methanogenesis: 4 H2 + CO2 -> CH4
    # Rate proportional to both H2 and CO2
    r_meth = k_meth * H2 * CO2_safe * f_inh

    # Sulfate Reduction: 4 H2 + SO4 -> H2S
    # Rate proportional to H2 and SO4
    r_sulf = k_sulf * H2 * SO4_safe * f_inh

    # Optional: Acetogenesis (set k_aceto = 0 to disable)
    r_aceto = k_aceto * H2 * (CO2_safe**2) * f_inh

    # Precipitation: H2S -> FeS (when exceeds saturation)
    r_precip = k_precip * max(0, H2S - H2S_sat)

    # STOICHIOMETRY (from balanced equations)
    dH2  = -4*r_meth - 4*r_sulf - 4*r_aceto
    dCO2 = -1*r_meth - 2*r_aceto
    dCH4 = +1*r_meth
    dH2S = +1*r_sulf - r_precip
    dSO4 = -1*r_sulf

    return [dH2, dCO2, dCH4, dH2S, dSO4]


def residuals_simplified(p, t_exp, data_exp, y0):
    """Cost function for optimization"""
    try:
        sol = solve_ivp(
            lambda t, y: simplified_ode_model(t, y, p),
            [0, t_exp[-1]],
            y0,
            method='Radau',
            t_eval=t_exp,
            rtol=1e-6,
            atol=1e-9
        )

        if not sol.success or len(sol.t) < len(t_exp):
            return np.ones(data_exp.size) * 10.0

        y_sim = sol.y.T

        # Weighted residuals
        # Higher weight for H2, CO2, SO4 (should fit well)
        # Lower weight for CH4, H2S (more noisy)
        weights = np.array([2.0, 2.0, 1.0, 0.5, 2.0])

        # Log-scale residuals (good for exponential behavior)
        log_sim = np.log1p(y_sim)
        log_exp = np.log1p(data_exp)

        res = (log_sim - log_exp) * weights
        res_flat = res.flatten()

        if np.isnan(res_flat).any() or np.isinf(res_flat).any():
            return np.ones(data_exp.size) * 10.0

        return res_flat

    except Exception as e:
        print(f"Integration error: {e}")
        return np.ones(data_exp.size) * 10.0


# ==============================================================================
# PART 3: Diagnostic Functions
# ==============================================================================

def diagnose_stoichiometry(y_sim, t_sim):
    """Check if stoichiometry is violated during simulation"""
    print("\n" + "="*60)
    print("STOICHIOMETRY DIAGNOSTIC")
    print("="*60)

    H2_init = y_sim[0, 0]
    CO2_init = y_sim[0, 1]
    CH4_init = y_sim[0, 2]
    H2S_init = y_sim[0, 3]
    SO4_init = y_sim[0, 4]

    H2_final = y_sim[-1, 0]
    CO2_final = y_sim[-1, 1]
    CH4_final = y_sim[-1, 2]
    H2S_final = y_sim[-1, 3]
    SO4_final = y_sim[-1, 4]

    # Mass balance checks
    # For Methanogenesis: ΔCH4 should equal -ΔCO2 (1:1 ratio)
    # For Sulfate Reduction: ΔH2S should equal -ΔSO4 (1:1 ratio)
    # For both: Total H2 consumed = 4*(ΔCH4 + ΔH2S)

    delta_H2 = H2_final - H2_init
    delta_CO2 = CO2_final - CO2_init
    delta_CH4 = CH4_final - CH4_init
    delta_H2S = H2S_final - H2S_init
    delta_SO4 = SO4_final - SO4_init

    print(f"\nConcentration Changes (mM):")
    print(f"  ΔH2:  {delta_H2:+.3f}")
    print(f"  ΔCO2: {delta_CO2:+.3f}")
    print(f"  ΔCH4: {delta_CH4:+.3f}")
    print(f"  ΔH2S: {delta_H2S:+.6f}")
    print(f"  ΔSO4: {delta_SO4:+.3f}")

    # Expected H2 consumption from products
    H2_from_CH4 = -4 * delta_CH4  # Methanogenesis uses 4 H2 per CH4
    H2_from_H2S = -4 * delta_H2S  # Sulfate reduction uses 4 H2 per H2S
    H2_expected = H2_from_CH4 + H2_from_H2S

    print(f"\nH2 Balance Check:")
    print(f"  Actual ΔH2:    {delta_H2:.3f} mM")
    print(f"  Expected ΔH2:  {H2_expected:.3f} mM (from 4*ΔCH4 + 4*ΔH2S)")
    print(f"  Difference:    {abs(delta_H2 - H2_expected):.6f} mM")

    if abs(delta_H2 - H2_expected) > 0.01:
        print("  ⚠️ WARNING: H2 balance violated!")
    else:
        print("  ✓ H2 balance OK")

    # CO2-CH4 balance (for methanogenesis)
    print(f"\nCO2-CH4 Balance Check (Methanogenesis):")
    print(f"  ΔCH4: {delta_CH4:.3f} mM")
    print(f"  ΔCO2: {delta_CO2:.3f} mM (should be ~ -ΔCH4)")
    balance_error = abs(delta_CH4 + delta_CO2)
    print(f"  Balance error: {balance_error:.6f} mM")

    if balance_error > 0.1:
        print("  ⚠️ WARNING: CO2-CH4 balance violated!")
    else:
        print("  ✓ CO2-CH4 balance OK")

    # SO4-H2S balance (for sulfate reduction)
    print(f"\nSO4-H2S Balance Check (Sulfate Reduction):")
    print(f"  ΔH2S: {delta_H2S:.6f} mM")
    print(f"  ΔSO4: {delta_SO4:.3f} mM (should be ~ -ΔH2S)")
    balance_error_s = abs(delta_H2S + delta_SO4)
    print(f"  Balance error: {balance_error_s:.6f} mM")

    if balance_error_s > 0.1:
        print("  ⚠️ WARNING: SO4-H2S balance violated!")
    else:
        print("  ✓ SO4-H2S balance OK")


def calculate_fit_quality(y_sim, data_exp, t_exp, t_sim):
    """Calculate goodness of fit metrics"""
    print("\n" + "="*60)
    print("FIT QUALITY METRICS")
    print("="*60)

    # Interpolate simulation to experimental time points
    from scipy.interpolate import interp1d

    species_names = ['H2', 'CO2', 'CH4', 'H2S', 'SO4']

    for i, name in enumerate(species_names):
        if y_sim.shape[0] == len(t_sim):
            # Interpolate simulation to experimental times
            interp = interp1d(t_sim, y_sim[:, i], kind='linear', fill_value='extrapolate')
            y_sim_at_exp = interp(t_exp)

            # Calculate metrics
            residuals = y_sim_at_exp - data_exp[:, i]
            mse = np.mean(residuals**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(residuals))

            # R² (coefficient of determination)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((data_exp[:, i] - np.mean(data_exp[:, i]))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf

            print(f"\n{name}:")
            print(f"  RMSE: {rmse:.6f} mM")
            print(f"  MAE:  {mae:.6f} mM")
            print(f"  R²:   {r2:.4f}")

            if r2 < 0.5:
                print(f"  ⚠️ Poor fit (R² < 0.5)")
            elif r2 < 0.8:
                print(f"  ⚡ Moderate fit")
            else:
                print(f"  ✓ Good fit")


# ==============================================================================
# PART 4: Main Optimization and Plotting
# ==============================================================================

def run_simplified_fitting(file_path):
    """Run the simplified model fitting"""

    # Load data
    t_exp, data_exp = load_experimental_data(file_path)
    if t_exp is None:
        return

    # Initial conditions (from first data point)
    y0 = data_exp[0, :]

    # Parameter setup
    # p = [k_meth, k_sulf, k_aceto, KI_H2S, k_precip, H2S_sat]
    p0 = np.array([0.1, 0.1, 0.0, 0.05, 0.1, 0.01])
    lb = np.array([1e-4, 1e-4, 0.0, 1e-3, 0.0, 0.0])
    ub = np.array([5.0, 5.0, 1.0, 1.0, 10.0, 0.5])

    print("\n" + "="*60)
    print("PARAMETER OPTIMIZATION")
    print("="*60)
    print("\nInitial guess:")
    print(f"  k_meth:   {p0[0]:.4f}")
    print(f"  k_sulf:   {p0[1]:.4f}")
    print(f"  k_aceto:  {p0[2]:.4f}")
    print(f"  KI_H2S:   {p0[3]:.4f}")
    print(f"  k_precip: {p0[4]:.4f}")
    print(f"  H2S_sat:  {p0[5]:.4f}")
    print("\nStarting optimization...")

    # Optimize
    res_opt = least_squares(
        residuals_simplified,
        p0,
        bounds=(lb, ub),
        args=(t_exp, data_exp, y0),
        method='trf',
        verbose=2,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=3000
    )

    p_fit = res_opt.x

    print("\n" + "="*60)
    print("FITTED PARAMETERS")
    print("="*60)
    print(f"  k_meth:   {p_fit[0]:.6f}")
    print(f"  k_sulf:   {p_fit[1]:.6f}")
    print(f"  k_aceto:  {p_fit[2]:.6f}")
    print(f"  KI_H2S:   {p_fit[3]:.6f}")
    print(f"  k_precip: {p_fit[4]:.6f}")
    print(f"  H2S_sat:  {p_fit[5]:.6f}")
    print(f"\nOptimization status: {res_opt.message}")
    print(f"Cost function value: {res_opt.cost:.6e}")

    # Simulate with fitted parameters
    t_sim = np.linspace(0, t_exp[-1], 1000)

    sol = solve_ivp(
        lambda t, y: simplified_ode_model(t, y, p_fit),
        [0, t_exp[-1]],
        y0,
        t_eval=t_sim,
        method='Radau',
        rtol=1e-6,
        atol=1e-9
    )

    y_sim = sol.y.T

    # Diagnostics
    diagnose_stoichiometry(y_sim, t_sim)
    calculate_fit_quality(y_sim, data_exp, t_exp, t_sim)

    # Plot results
    species_names = ['H2', 'CO2', 'CH4', 'H2S', 'SO4']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i in range(5):
        ax = axes[i]

        # Experimental data
        ax.plot(t_exp, data_exp[:, i], 'ko', markersize=8, label='Experimental', zorder=3)

        # Model prediction
        if y_sim.shape[0] == len(t_sim):
            ax.plot(t_sim, y_sim[:, i], 'b-', linewidth=2.5, label='Simplified Model', zorder=2)

        ax.set_title(f'{species_names[i]}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Concentration (mM)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)

        # Set reasonable y-limits
        if i == 3:  # H2S
            ax.set_ylim([-0.01, max(0.15, data_exp[:, i].max() * 1.2)])

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.suptitle("Simplified ChemPy-Inspired Model Fit", fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_path = file_path.replace('.txt', '_chempy_fit.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.show()

    return p_fit, y_sim, t_sim


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Update this path
    file_path = r'D:\chemical_thesis_repo\16.12_2025_calisma\Muller_2024_H2_Sandstone_at_25C.txt'

    print("="*60)
    print("CHEMPY-INSPIRED DIAGNOSTIC TOOL")
    print("Simplified model with automatic stoichiometry checking")
    print("="*60)

    p_fit, y_sim, t_sim = run_simplified_fitting(file_path)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
