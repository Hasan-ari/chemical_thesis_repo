"""
============================================================================
ODE MODEL - Two-Phase Anaerobic Model (14 State Variables)
============================================================================

Ported from: 2026-W02_Lstm_development/code/python/lstm_delta_learning_v2/lstm_delta_forecast_v2.py

State Variables (14):
    0: nH2_g   - Gas phase H2 (mmol)
    1: nCO2_g  - Gas phase CO2 (mmol)
    2: nCH4_g  - Gas phase CH4 (mmol)
    3: nH2S_g  - Gas phase H2S (mmol)
    4: H2_aq   - Aqueous H2 (mM)
    5: CO2_aq  - Aqueous CO2 (mM)
    6: SO4     - Sulfate (mM)
    7: FeS     - Iron sulfide precipitate (mM)
    8: X       - Biomass (arbitrary units)
    9: Acetate - Acetate concentration (mM)
    10: HCO3   - Bicarbonate (mM)
    11: S_tot  - Total dissolved sulfide (mM)
    12: Lag    - Lag phase indicator
    13: Fe_pool - Available iron pool (mM)

Author: Chemical Thesis Project
Date: 2026-W04
============================================================================
"""

import numpy as np


def model_mixed(t, y, p, env):
    """
    Two-phase anaerobic model (v4) - Python implementation.

    Simulates coupled methanogenesis, sulfate reduction, and acetogenesis
    with gas-liquid mass transfer and FeS precipitation.

    Parameters
    ----------
    t : float
        Current time (days)
    y : array_like
        State vector (14 elements)
    p : array_like
        Parameter vector (24 elements)
    env : dict
        Environmental parameters including:
        - Vg, Vl: Gas and liquid volumes
        - T, Rgas: Temperature and gas constant
        - Hcp_*_eff: Effective Henry's constants
        - pKa_H2S: pKa for H2S dissociation
        - SO4_sat_gyp: Gypsum saturation sulfate concentration
        - pH_fun: Interpolation function for pH(t)

    Returns
    -------
    dy : ndarray
        Derivatives of state variables
    """
    # Unpack environmental parameters
    Vg, Vl, T, Rgas = env['Vg'], env['Vl'], env['T'], env['Rgas']
    Hcp_H2, Hcp_CO2, Hcp_H2S = env['Hcp_H2_eff'], env['Hcp_CO2_eff'], env['Hcp_H2S_eff']
    pKa = env['pKa_H2S']
    pH = env['pH_fun'](t)

    # Ensure non-negative concentrations (numerical stability)
    y = np.maximum(y, 1e-12)
    Fe_pool = max(y[13], 0)

    # Unpack state variables
    nH2_g, nCO2_g, nCH4_g, nH2S_g = y[0], y[1], y[2], y[3]
    H2_aq, CO2_aq, SO4, FeS = y[4], y[5], y[6], y[7]
    X, Ac, HCO3, S_tot, Lag = y[8], y[9], y[10], y[11], y[12]

    # Unpack kinetic parameters
    k_m, k_s, k_a = p[0], p[1], p[2]           # Max specific rates
    Y_m, Y_s, Y_a = p[3], p[4], p[5]           # Yield coefficients
    KI_m, KI_s, KI_a = p[6], p[7], p[8]        # Inhibition constants
    k_prec, HS_sat = p[9], p[10]               # Precipitation rate and saturation
    H2_th, DG_th = p[11], p[12]                # Threshold H2 and Gibbs energy
    K_H2, K_SO4, K_CO2 = p[13], p[14], p[15]   # Half-saturation constants
    kla_H2, kla_CO2, kla_H2S = p[16], p[17], p[18]  # Mass transfer coefficients
    b, t_lag, w_lag = p[19], p[20], p[21]      # Decay, lag time, lag width
    k_diss_gyp, beta_SO4_m = p[22], p[23]      # Gypsum dissolution, competition

    # Calculate partial pressures (atm)
    pH2 = (nH2_g / 1000) * Rgas * T / Vg
    pCO2 = (nCO2_g / 1000) * Rgas * T / Vg
    pH2S = (nH2S_g / 1000) * Rgas * T / Vg

    # Equilibrium concentrations and mass transfer fluxes
    Ceq_H2, Ceq_CO2, Ceq_H2S = Hcp_H2 * pH2, Hcp_CO2 * pCO2, Hcp_H2S * pH2S
    J_H2 = kla_H2 * (Ceq_H2 - H2_aq)
    J_CO2 = kla_CO2 * (Ceq_CO2 - CO2_aq)

    # Sulfide speciation and stripping
    frac_HS = 1 / (1 + 10**(pKa - pH))
    HS_aq = S_tot * frac_HS
    H2S_aq = S_tot * (1 - frac_HS)
    Jout_H2S = kla_H2S * (H2S_aq - Ceq_H2S)

    # Inhibition by sulfide
    f_inh_m = KI_m / (KI_m + HS_aq)
    f_inh_s = KI_s / (KI_s + HS_aq)
    f_inh_a = KI_a / (KI_a + HS_aq)

    # H2 threshold and lag phase effects
    f_H2 = H2_aq / (H2_aq + H2_th)
    f_lag = 1 / (1 + np.exp((t_lag - t) / max(w_lag, 1e-3)))
    f_act = f_H2 * f_lag

    # Substrate limitation (Monod kinetics)
    mH2 = H2_aq / (K_H2 + H2_aq)
    mSO4 = SO4 / (K_SO4 + SO4)
    mCO2 = CO2_aq / (K_CO2 + CO2_aq)

    # Thermodynamic factors
    RT = 8.314e-3 * T
    Q_a = max(Ac, 1e-12) / (max(H2_aq, 1e-12)**4 * max(CO2_aq, 1e-12)**2)

    fT_s = 1 / (1 + np.exp((-152 + RT * np.log(1) - DG_th) / RT))
    fT_m = 1 / (1 + np.exp((-130 - DG_th) / RT))
    fT_a = 1 / (1 + np.exp((-95 + RT * np.log(Q_a) - DG_th) / RT))

    # Competition factor for methanogens
    f_comp_m = 1 / (1 + beta_SO4_m * SO4)

    # Reaction rates
    r_meth = k_m * X * mH2 * mCO2 * f_inh_m * f_act * fT_m * f_comp_m  # Methanogenesis
    r_sulf = k_s * X * mH2 * mSO4 * f_inh_s * f_act * fT_s            # Sulfate reduction
    r_aceto = k_a * X * mH2 * (mCO2**2) * f_inh_a * f_act * fT_a      # Acetogenesis

    # Precipitation and dissolution
    r_prec = min(k_prec * max(0, HS_aq - HS_sat), Fe_pool)
    r_diss_gyp = k_diss_gyp * max(0, env['SO4_sat_gyp'] - SO4)

    # Mass balances (derivatives)
    dy = np.zeros(14)
    dy[0] = -J_H2 * Vl                                    # nH2_g
    dy[1] = -J_CO2 * Vl                                   # nCO2_g
    dy[2] = r_meth * Vl                                   # nCH4_g
    dy[3] = Jout_H2S * Vl                                 # nH2S_g
    dy[4] = J_H2 - 4 * (r_meth + r_sulf + r_aceto)       # H2_aq
    dy[5] = J_CO2 - r_meth - 2 * r_aceto                 # CO2_aq
    dy[6] = -r_sulf + r_diss_gyp                         # SO4
    dy[7] = r_prec                                        # FeS
    dy[8] = Y_m * r_meth + Y_s * r_sulf + Y_a * r_aceto - b * X  # X (biomass)
    dy[9] = r_aceto                                       # Acetate
    dy[10] = 0.0                                          # HCO3 (constant)
    dy[11] = r_sulf - r_prec - Jout_H2S                  # S_tot
    dy[12] = (f_lag - Lag) / max(w_lag, 1e-3)            # Lag
    dy[13] = -r_prec                                      # Fe_pool

    return dy


# State variable names for reference
STATE_NAMES = [
    'nH2_g', 'nCO2_g', 'nCH4_g', 'nH2S_g',
    'H2_aq', 'CO2_aq', 'SO4', 'FeS',
    'X', 'Acetate', 'HCO3', 'S_tot', 'Lag', 'Fe_pool'
]

STATE_UNITS = [
    'mmol', 'mmol', 'mmol', 'mmol',
    'mM', 'mM', 'mM', 'mM',
    'a.u.', 'mM', 'mM', 'mM', '-', 'mM'
]
