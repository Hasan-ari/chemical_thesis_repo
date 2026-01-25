"""
Parameter Loading Module
========================
Loads best-fit parameters from MATLAB .mat files and experimental data.
"""

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Tuple

from .ode_model import Environment


# Experimental data for Basalt @ 25C (from Muller 2024)
# Columns: time(days), H2(umol), CO2(umol), CH4(umol), H2S(umol), pH, SO4(mM)
BASALT_25C_DATA = np.array([
    [0.0,   9074, 2464,   0,   1, 6.7, 5.7],
    [1.1,   8655, 2338,   0,   0, 6.9, 5.7],
    [5.0,   8016, 2203,  12,  20, 7.0, 3.1],
    [6.0,   7603, 2086,  25,  35, 7.2, 2.8],
    [7.0,   7362, 1949,  45,  54, 7.1, 2.2],
    [8.0,   6946, 1820,  66,  50, 7.2, 2.4],
    [9.0,   5560, 1407,  89,  49, 7.2, 2.2],
    [12.0,  3728,  766, 141,  41, 6.7, 1.9],
    [13.9,  2128,  280, 173,  42, 6.0, 1.3],
    [15.9,  1808,  189, 201,  35, 5.9, 1.2],
    [19.0,  1409,   80, 255,  30, 5.9, 1.1],
])


def load_basalt_25c_params(mat_file: Path = None) -> Tuple[np.ndarray, Environment, np.ndarray]:
    """
    Load best-fit parameters for Basalt @ 25C.

    Args:
        mat_file: Path to .mat file. If None, uses default location.

    Returns:
        p_fit: 28-element parameter vector
        env: Environment configuration with pH interpolant
        y0: Initial state vector (14 elements)
    """
    # Default paths to search
    default_paths = [
        Path(__file__).parent.parent.parent.parent.parent / "2026-W02_Lstm_development/code/matlab/Basalt/25C/best_fit_params_Basalt_25C.mat",
        Path(__file__).parent.parent.parent.parent.parent / "2026-W01_model_anlama/code/matlab/best_fit_params_Basalt_25C.mat",
    ]

    if mat_file is None:
        for path in default_paths:
            if path.exists():
                mat_file = path
                break

    if mat_file is None or not Path(mat_file).exists():
        print("Warning: .mat file not found. Using default parameters from MATLAB code.")
        return get_default_params()

    # Load .mat file
    mat_data = loadmat(str(mat_file))

    # Extract fitted parameters
    p_fit = mat_data['p_fit'].flatten()

    # Build environment
    env = build_environment(p_fit)

    # Get initial conditions
    y0 = get_initial_conditions(env)

    return p_fit, env, y0


def get_default_params() -> Tuple[np.ndarray, Environment, np.ndarray]:
    """
    Return default parameters from the MATLAB code (p0 values).
    Used when .mat file is not available.
    """
    p0 = np.array([
        0.06, 0.08, 0.03,           # k_m, k_s, k_a
        0.06, 0.05, 0.05,           # Y_m, Y_s, Y_a
        0.20, 0.20, 0.20,           # KI_m, KI_s, KI_a
        0.02, 0.10, 0.02, -12,      # k_prec, HS_sat, H2_th, DG_th
        0.50, 0.50, 0.80,           # K_H2, K_SO4, K_CO2
        10.0, 10.0, 25.0,           # kla_H2, kla_CO2, kla_H2S
        0.01, 3.0, 0.7,             # b, t_lag, w_lag
        0.12, 0.10,                 # k_diss_gyp, beta_SO4_m
        1.00, 1.00, 1.00,           # phi_H2, phi_CO2, phi_H2S
        1.00                        # alpha_H2S
    ])

    env = build_environment(p0)
    y0 = get_initial_conditions(env)

    return p0, env, y0


def build_environment(p: np.ndarray) -> Environment:
    """
    Build environment configuration with pH interpolant.

    Args:
        p: Parameter vector (28 elements)

    Returns:
        Environment with all constants set
    """
    # Extract scale factors from parameters
    phi_H2 = p[24] if len(p) > 24 else 1.0
    phi_CO2 = p[25] if len(p) > 25 else 1.0
    phi_H2S = p[26] if len(p) > 26 else 1.0

    # Create pH interpolant from experimental data
    t_exp = BASALT_25C_DATA[:, 0]
    pH_exp = BASALT_25C_DATA[:, 5]

    # Linear interpolation with extrapolation
    pH_interp = interp1d(t_exp, pH_exp, kind='linear', fill_value='extrapolate')
    pH_fun = lambda t: float(np.clip(pH_interp(t), 5.0, 8.0))

    env = Environment(
        Vg=0.14,              # headspace volume [L]
        Vl=0.015,             # liquid volume [L]
        T=298.15,             # temperature [K] (25C)
        Rgas=0.082057,        # gas constant [L*atm/(mol*K)]
        Hcp_H2_base=0.78,     # H2 Henry @ 25C [mmol/L/atm]
        Hcp_CO2_base=34.0,    # CO2 Henry @ 25C [mmol/L/atm]
        Hcp_H2S_base=90.0,    # H2S Henry @ 25C [mmol/L/atm]
        Hcp_H2_eff=phi_H2 * 0.78,
        Hcp_CO2_eff=phi_CO2 * 34.0,
        Hcp_H2S_eff=phi_H2S * 90.0,
        pKa_H2S=7.05,
        SO4_sat_gyp=36.0,     # gypsum-buffered SO4 [mM]
        pH_fun=pH_fun
    )

    return env


def get_initial_conditions(env: Environment) -> np.ndarray:
    """
    Calculate initial conditions from experimental data.

    Args:
        env: Environment configuration

    Returns:
        y0: Initial state vector (14 elements)
    """
    # Initial experimental values (convert umol to mmol)
    nH2_g_0 = BASALT_25C_DATA[0, 1] / 1000   # mmol
    nCO2_g_0 = BASALT_25C_DATA[0, 2] / 1000  # mmol
    nCH4_g_0 = BASALT_25C_DATA[0, 3] / 1000  # mmol
    nH2S_g_0 = BASALT_25C_DATA[0, 4] / 1000  # mmol
    SO4_0 = BASALT_25C_DATA[0, 6]            # mM (mmol/L)

    # Calculate initial partial pressures
    pH2 = (nH2_g_0 / 1000) * env.Rgas * env.T / env.Vg
    pCO2 = (nCO2_g_0 / 1000) * env.Rgas * env.T / env.Vg

    # Initial aqueous equilibrium
    H2_aq_0 = env.Hcp_H2_eff * pH2
    CO2_aq_0 = env.Hcp_CO2_eff * pCO2

    # Initial state vector
    # [nH2_g, nCO2_g, nCH4_g, nH2S_g, H2_aq, CO2_aq, SO4, FeS,
    #  X, Acetate, HCO3, S_tot, Lag, Fe_pool]
    y0 = np.array([
        nH2_g_0,      # nH2_g
        nCO2_g_0,     # nCO2_g
        nCH4_g_0,     # nCH4_g
        nH2S_g_0,     # nH2S_g
        H2_aq_0,      # H2_aq
        CO2_aq_0,     # CO2_aq
        SO4_0,        # SO4
        0.01,         # FeS (small initial)
        0.01,         # X (biomass seed)
        0.0,          # Acetate
        0.0,          # HCO3
        1.0,          # S_tot (tiny sulfide seed)
        0.0,          # Lag
        0.10          # Fe_pool (dissolved Fe(II))
    ])

    return y0


def get_experimental_data():
    """
    Get experimental data for validation/plotting.

    Returns:
        dict with t_exp, and experimental measurements
    """
    data = BASALT_25C_DATA
    return {
        't_exp': data[:, 0],
        'nH2_g_exp': data[:, 1] / 1000,   # mmol
        'nCO2_g_exp': data[:, 2] / 1000,  # mmol
        'nCH4_g_exp': data[:, 3] / 1000,  # mmol
        'nH2S_g_exp': data[:, 4] / 1000,  # mmol
        'pH_exp': data[:, 5],
        'SO4_exp': data[:, 6]             # mM
    }
