"""
Two-Phase Anaerobic Model (v4) - Basalt @ 25C
=============================================
Exact Python translation of the MATLAB v4 code from Professor's files.

14 State Variables:
    y[0]  nH2_g    : H2 gas moles in headspace (mmol)
    y[1]  nCO2_g   : CO2 gas moles in headspace (mmol)
    y[2]  nCH4_g   : CH4 gas moles produced (mmol)
    y[3]  nH2S_g   : H2S gas in headspace (mmol)
    y[4]  H2_aq    : Dissolved H2 concentration (mmol/L)
    y[5]  CO2_aq   : Dissolved CO2 concentration (mmol/L)
    y[6]  SO4      : Sulfate concentration (mmol/L)
    y[7]  FeS      : Precipitated iron sulfide (mmol/L)
    y[8]  X        : Biomass concentration (mmol/L equiv.)
    y[9]  Acetate  : Acetate concentration (mmol/L)
    y[10] HCO3     : Bicarbonate (mmol/L) - kept constant
    y[11] S_tot    : Total dissolved sulfide (mmol/L)
    y[12] Lag      : Lag phase activation factor (0-1)
    y[13] Fe_pool  : Dissolved Fe(II) available for FeS precip (mmol/L)

28 Parameters (p):
    p[0-2]   : k_m, k_s, k_a (max rate constants)
    p[3-5]   : Y_m, Y_s, Y_a (biomass yields)
    p[6-8]   : KI_m, KI_s, KI_a (H2S inhibition constants)
    p[9]     : k_prec (FeS precipitation rate)
    p[10]    : HS_sat (HS- saturation threshold)
    p[11]    : H2_th (H2 activation threshold)
    p[12]    : DG_th (thermodynamic threshold)
    p[13-15] : K_H2, K_SO4, K_CO2 (Monod half-saturation)
    p[16-18] : kla_H2, kla_CO2, kla_H2S (mass transfer coefficients)
    p[19]    : b (biomass decay)
    p[20-21] : t_lag, w_lag (lag phase timing & width)
    p[22]    : k_diss_gyp (gypsum dissolution rate)
    p[23]    : beta_SO4_m (SO4-methanogen competition)
    p[24-26] : phi_H2, phi_CO2, phi_H2S (Henry scale factors)
    p[27]    : alpha_H2S (H2S degassing scale)
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class Environment:
    """Environment constants for the reactor system."""
    Vg: float = 0.14          # headspace volume [L]
    Vl: float = 0.015         # liquid volume [L]
    T: float = 298.15         # temperature [K] (25C)
    Rgas: float = 0.082057    # gas constant [L*atm/(mol*K)]

    # Henry constants @ 25C [mmol/L/atm]
    Hcp_H2_base: float = 0.78
    Hcp_CO2_base: float = 34.0
    Hcp_H2S_base: float = 90.0

    # Effective Henry constants (after scaling)
    Hcp_H2_eff: float = 0.78
    Hcp_CO2_eff: float = 34.0
    Hcp_H2S_eff: float = 90.0

    pKa_H2S: float = 7.05     # pKa1 for H2S
    SO4_sat_gyp: float = 36.0 # gypsum-buffered SO4 level [mM]

    pH_fun: Callable[[float], float] = None  # pH interpolant

    def __post_init__(self):
        if self.pH_fun is None:
            # Default: constant pH = 7.0
            self.pH_fun = lambda t: 7.0


def model_mixed(t: float, y: np.ndarray, p: np.ndarray, env: Environment) -> np.ndarray:
    """
    Two-phase anaerobic model with 14 state variables.
    Exact replication of MATLAB v4 code.

    Args:
        t: Current time (days)
        y: State vector (14 elements)
        p: Parameter vector (28 elements)
        env: Environment constants

    Returns:
        dydt: Derivative vector (14 elements)
    """
    # Unpack environment
    Vg, Vl, T, Rgas = env.Vg, env.Vl, env.T, env.Rgas
    Hcp_H2 = env.Hcp_H2_eff
    Hcp_CO2 = env.Hcp_CO2_eff
    Hcp_H2S = env.Hcp_H2S_eff
    pKa = env.pKa_H2S
    pH = env.pH_fun(t)

    # Guards against negative values
    eps = 1e-12
    y = np.maximum(y, eps)
    Fe_pool = max(y[13], 0)

    # Map States
    nH2_g, nCO2_g, nCH4_g, nH2S_g = y[0], y[1], y[2], y[3]
    H2_aq, CO2_aq, SO4, FeS = y[4], y[5], y[6], y[7]
    X, Ac, HCO3, S_tot, Lag = y[8], y[9], y[10], y[11], y[12]

    # Parameters (28 total)
    k_m, k_s, k_a = p[0], p[1], p[2]
    Y_m, Y_s, Y_a = p[3], p[4], p[5]
    KI_m, KI_s, KI_a = p[6], p[7], p[8]
    k_prec, HS_sat = p[9], p[10]
    H2_th, DG_th = p[11], p[12]
    K_H2, K_SO4, K_CO2 = p[13], p[14], p[15]
    kla_H2, kla_CO2, kla_H2S = p[16], p[17], p[18]
    b, t_lag, w_lag = p[19], p[20], p[21]
    k_diss_gyp, beta_SO4_m = p[22], p[23]
    # phi_H2, phi_CO2, phi_H2S = p[24], p[25], p[26]  # already applied to env
    # alpha_H2S = p[27]  # not used directly here

    # Thermodynamic constants
    RkJ = 8.314e-3  # kJ/(mol*K)
    RT = RkJ * T
    DG0_m, DG0_s, DG0_a = -130, -152, -95

    # Partial pressures (atm) from moles (mmol)
    pH2 = (nH2_g / 1000) * Rgas * T / Vg
    pCO2 = (nCO2_g / 1000) * Rgas * T / Vg
    pH2S = (nH2S_g / 1000) * Rgas * T / Vg

    # Henry equilibria (mmol/L) @ 25C
    Ceq_H2 = Hcp_H2 * pH2
    Ceq_CO2 = Hcp_CO2 * pCO2
    Ceq_H2S = Hcp_H2S * pH2S

    # Gas-liquid transfers for H2, CO2 (liquid-side uptake positive)
    J_H2 = kla_H2 * (Ceq_H2 - H2_aq)
    J_CO2 = kla_CO2 * (Ceq_CO2 - CO2_aq)

    # Sulfide speciation
    frac_HS = 1 / (1 + 10**(pKa - pH))
    frac_H2S = 1 - frac_HS
    HS_aq = S_tot * frac_HS
    H2S_aq = S_tot * frac_H2S

    # H2S: outgassing-positive flux
    Jout_H2S = kla_H2S * (H2S_aq - Ceq_H2S)

    # Inhibitions & Activation
    f_inh_m = KI_m / (KI_m + HS_aq)
    f_inh_s = KI_s / (KI_s + HS_aq)
    f_inh_a = KI_a / (KI_a + HS_aq)
    f_H2 = H2_aq / (H2_aq + H2_th)
    f_lag = 1 / (1 + np.exp((t_lag - t) / max(w_lag, 1e-3)))
    f_act = f_H2 * f_lag

    # Monod saturations
    mH2 = H2_aq / (K_H2 + H2_aq)
    mSO4 = SO4 / (K_SO4 + SO4)
    mCO2 = CO2_aq / (K_CO2 + CO2_aq)

    # Thermo gates
    Q_a = Ac / (H2_aq**4 * CO2_aq**2)
    DG_s = DG0_s + RT * np.log(1)  # Q_s = 1
    DG_m = DG0_m
    DG_a = DG0_a + RT * np.log(Q_a)

    fT_s = 1 / (1 + np.exp((DG_s - DG_th) / RT))
    fT_m = 1 / (1 + np.exp((DG_m - DG_th) / RT))
    fT_a = 1 / (1 + np.exp((DG_a - DG_th) / RT))

    # Sulfate vs methanogen competition gate
    f_comp_m = 1 / (1 + beta_SO4_m * SO4)

    # Biomass-mediated rates (mmol/L/day)
    r_meth = k_m * X * mH2 * mCO2 * f_inh_m * f_act * fT_m * f_comp_m
    r_sulf = k_s * X * mH2 * mSO4 * f_inh_s * f_act * fT_s
    r_aceto = k_a * X * mH2 * (mCO2**2) * f_inh_a * f_act * fT_a

    # Precipitation (from HS-), limited by Fe pool
    r_prec_raw = k_prec * max(0, HS_aq - HS_sat)
    r_prec = min(r_prec_raw, Fe_pool)

    # Gypsum dissolution source
    SO4_sat = env.SO4_sat_gyp
    r_diss_gyp = k_diss_gyp * max(0, SO4_sat - SO4)

    # Gas balances (mmol/day)
    dnH2_g = -J_H2 * Vl
    dnCO2_g = -J_CO2 * Vl
    dnCH4_g = r_meth * Vl
    dnH2S_g = Jout_H2S * Vl

    # Liquid balances (mmol/L/day)
    dH2_aq = J_H2 - 4*r_meth - 4*r_sulf - 4*r_aceto
    dCO2_aq = J_CO2 - 1*r_meth - 2*r_aceto
    dSO4 = -1*r_sulf + r_diss_gyp
    dFeS = r_prec
    dX = Y_m*r_meth + Y_s*r_sulf + Y_a*r_aceto - b*X
    dAc = r_aceto
    dHCO3 = 0  # kept constant

    # Fe pool balance (mmol/L/day)
    dFe_pool = -r_prec

    # Total dissolved sulfide balance
    dS_tot = 1.0*r_sulf - r_prec - Jout_H2S

    # Lag tracker
    dLag = (f_lag - Lag) / max(w_lag, 1e-3)

    # Collect derivatives
    dydt = np.array([
        dnH2_g, dnCO2_g, dnCH4_g, dnH2S_g,
        dH2_aq, dCO2_aq, dSO4, dFeS,
        dX, dAc, dHCO3, dS_tot, dLag, dFe_pool
    ])

    return dydt


def compute_rates(t: float, y: np.ndarray, p: np.ndarray, env: Environment) -> np.ndarray:
    """
    Compute reaction rates at a given state (for diagnostics/output).

    Returns:
        rates: [r_meth, r_sulf, r_precip, r_aceto] in mmol/L/day
    """
    eps = 1e-12
    y = np.maximum(y, eps)
    Fe_pool = max(y[13], 0)

    H2_aq, CO2_aq, SO4 = y[4], y[5], y[6]
    X, S_tot = y[8], y[11]

    pH = env.pH_fun(t)
    pKa = env.pKa_H2S
    frac_HS = 1 / (1 + 10**(pKa - pH))
    HS = S_tot * frac_HS

    k_m, k_s, k_a = p[0], p[1], p[2]
    KI_m, KI_s, KI_a = p[6], p[7], p[8]
    k_prec, HS_sat = p[9], p[10]
    H2_th = p[11]
    K_H2, K_SO4, K_CO2 = p[13], p[14], p[15]
    t_lag, w_lag = p[20], p[21]
    beta_SO4_m = p[23]

    f_inh_m = KI_m / (KI_m + HS)
    f_inh_s = KI_s / (KI_s + HS)
    f_inh_a = KI_a / (KI_a + HS)
    f_H2 = H2_aq / (H2_aq + H2_th)
    f_lag = 1 / (1 + np.exp((t_lag - t) / max(w_lag, 1e-3)))
    f_act = f_H2 * f_lag

    mH2 = H2_aq / (K_H2 + H2_aq)
    mSO4 = SO4 / (K_SO4 + SO4)
    mCO2 = CO2_aq / (K_CO2 + CO2_aq)

    f_comp_m = 1 / (1 + beta_SO4_m * SO4)

    r_meth = k_m * X * mH2 * mCO2 * f_inh_m * f_act * f_comp_m
    r_sulf = k_s * X * mH2 * mSO4 * f_inh_s * f_act
    r_aceto = k_a * X * mH2 * (mCO2**2) * f_inh_a * f_act

    r_prec_raw = k_prec * max(0, HS - HS_sat)
    r_prec = min(r_prec_raw, Fe_pool)

    return np.array([r_meth, r_sulf, r_prec, r_aceto])


def speciate_sulfide(S_tot: np.ndarray, pH: np.ndarray, pKa: float = 7.05):
    """
    Calculate H2S and HS- concentrations from total sulfide.

    Returns:
        H2S_aq, HS_aq: Arrays of aqueous species concentrations
    """
    frac_HS = 1 / (1 + 10**(pKa - pH))
    HS_aq = S_tot * frac_HS
    H2S_aq = S_tot - HS_aq
    return H2S_aq, HS_aq


# State variable names for output
STATE_NAMES = [
    'nH2_g', 'nCO2_g', 'nCH4_g', 'nH2S_g',
    'H2_aq', 'CO2_aq', 'SO4', 'FeS',
    'X', 'Acetate', 'HCO3', 'S_tot', 'Lag', 'Fe_pool'
]

# Extended output names (with derived quantities and rates)
OUTPUT_NAMES = STATE_NAMES + ['H2S_aq', 'HS', 'r_meth', 'r_sulf', 'r_precip', 'r_aceto']
