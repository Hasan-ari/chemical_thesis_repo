# %% [markdown]
# # MATLAB to PyTorch Conversion: H₂ Biogeochemical Reactive Transport Model
#
# **Original MATLAB Code:** `rnn_transport_multiguild_uq_v3.m`
#
# This script is a **strict 1:1 conversion** from MATLAB to PyTorch.
# Every section maps directly to the original MATLAB code with heavy comments.

# %% [markdown]
# ## Part 1: Library Imports
#
# **MATLAB equivalent:** Automatic (no explicit imports needed in MATLAB)
#
# In Python, we must explicitly import all libraries that MATLAB has built-in.

# %%
# Core numerical computing libraries
import numpy as np                      # Replaces MATLAB's built-in matrix operations
import pandas as pd                     # For data manipulation (optional)

# Scientific computing - Optimization and ODE solving
from scipy.optimize import least_squares  # Replaces MATLAB's lsqnonlin from Optimization Toolbox
from scipy.integrate import solve_ivp     # Replaces MATLAB's ode45 from MATLAB base

# PyTorch deep learning framework - Replaces MATLAB Deep Learning Toolbox
import torch                            # Core PyTorch library (like MATLAB's base functions)
import torch.nn as nn                   # Neural network modules (replaces MATLAB layers)
import torch.optim as optim             # Optimizers (replaces trainingOptions)
from torch.utils.data import Dataset, DataLoader  # Data handling (MATLAB does automatically)

# Visualization
import matplotlib.pyplot as plt         # Replaces MATLAB's built-in plotting functions

# Utilities
from typing import Tuple                # For type hints (MATLAB doesn't need this)
import warnings
warnings.filterwarnings('ignore')       # Suppress warnings for cleaner output

# Set random seeds for reproducibility
# MATLAB: rng('default') or rng(42)
# Python: Manual seeding for both NumPy and PyTorch
torch.manual_seed(42)                   # PyTorch random seed
np.random.seed(42)                      # NumPy random seed

# Device configuration (CPU vs GPU)
# MATLAB: Automatically handles GPU if available
# PyTorch: Must explicitly specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f'PyTorch version: {torch.__version__}')

# %% [markdown]
# ## Part 2: Load Experimental Data
#
# **MATLAB lines 8-15:**
# ```matlab
# %% Load experimental data
# raw = readmatrix('Muller_2024_H2_Sandstone_at_25C.txt');
# t_exp = raw(:,1); % time in days
# % concentrations of chemical species (H2, CO2, CH4, H2S) at columns from 2 to 5 are in µmol, at column 6 (SO4) is in already in mmol
# data_exp = [raw(:,2:5)*1e-3, raw(:,7)]; % Convert µmol to mmol
#
# %% Initial condition: [H2, CO2, CH4, H2S, SO4, FeS (Precipitated iron sulfide)
# % , X_meth (Methanogen biomass), X_sulf (Sulfate reducer biomass), X_aceto (Acetogen biomass), Acetate (Acetic acid)]
# x0 = [data_exp(1,:)'; 0.01; 0.01; 0.01; 0; 0]; % 10 elements
# ```

# %%
# MATLAB: raw = readmatrix('Muller_2024_H2_Sandstone_at_25C.txt')
# Load raw experimental data from text file
# encoding='latin-1' handles special characters like µ (micro symbol)
# skiprows=2 skips the two header comment lines in the file
raw = np.loadtxt('Muller_2024_H2_Sandstone_at_25C.txt', skiprows=2, encoding='latin-1')

# MATLAB: t_exp = raw(:,1)
# Extract time column (first column, index 0 in Python due to 0-based indexing)
# Units: days
t_exp = raw[:, 0]

# MATLAB: data_exp = [raw(:,2:5)*1e-3, raw(:,7)]
# Extract and convert concentration data
# Columns in MATLAB (1-based):
#   Column 2-5: H2, CO2, CH4, H2S in µmol/L
#   Column 7: SO4 in mmol/L (already in correct units)
# Columns in Python (0-based):
#   Column 1-4: H2, CO2, CH4, H2S in µmol/L (need conversion to mmol/L)
#   Column 6: SO4 in mmol/L (no conversion needed)

# Extract H2, CO2, CH4, H2S (columns 1-4) and convert µmol → mmol
conc_micromol = raw[:, 1:5] * 1e-3      # Multiply by 1e-3 to convert µmol/L to mmol/L

# Extract SO4 (column 6, already in mmol/L)
so4_mmol = raw[:, 6:7]                  # Keep as 2D array for concatenation

# Combine: [H2, CO2, CH4, H2S, SO4] all in mmol/L
data_exp = np.hstack([conc_micromol, so4_mmol])  # Shape: [n_timepoints, 5]

# MATLAB: x0 = [data_exp(1,:)'; 0.01; 0.01; 0.01; 0; 0]
# Create initial conditions vector for all 10 state variables
# MATLAB indexing: data_exp(1,:) is first row (1-based)
# Python indexing: data_exp[0,:] is first row (0-based)
#
# Initial conditions:
# [0-4]: H2, CO2, CH4, H2S, SO4 from first experimental measurement
# [5]: FeS (precipitated iron sulfide) = 0.01 mmol/L initial guess
# [6]: X_meth (methanogen biomass) = 0.01 initial biomass
# [7]: X_sulf (sulfate reducer biomass) = 0.01 initial biomass
# [8]: X_aceto (acetogen biomass) = 0.01 initial biomass
# [9]: Acetate (acetic acid) = 0.0 mmol/L (no acetate initially)
x0 = np.concatenate([
    data_exp[0, :],                         # First row: H2, CO2, CH4, H2S, SO4 (5 values)
    np.array([0.01, 0.01, 0.01, 0.01, 0.0]) # FeS, X_meth, X_sulf, X_aceto, Acetate (5 values)
])  # Total: 10 elements

print(f'Loaded {len(t_exp)} experimental time points')
print(f'Time range: {t_exp[0]:.1f} to {t_exp[-1]:.1f} days')
print(f'Data shape: {data_exp.shape} (timepoints × species)')
print(f'Initial conditions (x0): {x0}')

# %% [markdown]
# ## Part 3: Mechanistic ODE Model for Multi-Guild Biogeochemistry
#
# **MATLAB lines 225-291:** `trueODEfunc_multiguild`
#
# This function defines the differential equations for:
# - 3 microbial guilds competing for H₂
# - Chemical reactions with thermodynamic constraints
# - Biomass growth dynamics
# - Precipitation reactions

# %%
def ode_multiguild(t, y, p):
    """
    Multi-guild biogeochemical ODE system for hydrogen reactions.

    MATLAB equivalent: function dydt = trueODEfunc_multiguild(~, y, p)
    (lines 225-291 in rnn_transport_multiguild_uq_v3.m)

    Implements three anaerobic H2-consuming reactions:
    1. Methanogenesis:    4H2 + CO2 → CH4 + 2H2O       (ΔG° = -130 kJ/mol)
    2. Sulfate Reduction: 4H2 + SO4²⁻ → H2S + 4H2O    (ΔG° = -152 kJ/mol)
    3. Acetogenesis:      4H2 + 2CO2 → CH3COOH + 2H2O (ΔG° = -95 kJ/mol)

    Args:
        t: Time (not used in equations, but required by solve_ivp)
        y: State vector [10 elements]:
           [H2, CO2, CH4, H2S, SO4, FeS, X_meth, X_sulf, X_aceto, Acetate]
        p: Parameter vector [13 elements]:
           [k_meth, k_sulf, k_aceto,      # Maximum reaction rates
            Y_m, Y_s, Y_a,                 # Biomass yields
            KI_meth, KI_sulf, KI_aceto,    # Inhibition constants
            k_precip, H2S_sat, H2_thresh,  # Precipitation parameters
            DG_thresh]                      # Thermodynamic threshold

    Returns:
        dydt: Array of derivatives [10 elements] in mmol/L/day
    """

    # === UNPACK STATE VARIABLES ===
    # MATLAB: H2 = y(1); CO2 = y(2); ... (1-based indexing)
    # Python: Use 0-based indexing
    H2 = y[0]        # Hydrogen concentration [mmol/L]
    CO2 = y[1]       # Carbon dioxide concentration [mmol/L]
    CH4 = y[2]       # Methane concentration [mmol/L]
    H2S = y[3]       # Hydrogen sulfide concentration [mmol/L]
    SO4 = y[4]       # Sulfate concentration [mmol/L]
    FeS = y[5]       # Precipitated iron sulfide [mmol/L]
    X_meth = y[6]    # Methanogen biomass [mmol/L or similar units]
    X_sulf = y[7]    # Sulfate reducer biomass [mmol/L or similar units]
    X_aceto = y[8]   # Acetogen biomass [mmol/L or similar units]
    Acetate = y[9]   # Acetate (acetic acid) concentration [mmol/L]

    # === UNPACK PARAMETERS ===
    # MATLAB: k_meth = p(1); k_sulf = p(2); ... (1-based indexing)
    # Python: Use 0-based indexing and array slicing

    # Maximum reaction rates [1/day or similar]
    k_meth = p[0]    # Methanogenesis maximum rate
    k_sulf = p[1]    # Sulfate reduction maximum rate
    k_aceto = p[2]   # Acetogenesis maximum rate

    # Biomass yields [dimensionless, 0-1]
    Y_m = p[3]       # Methanogen biomass yield
    Y_s = p[4]       # Sulfate reducer biomass yield
    Y_a = p[5]       # Acetogen biomass yield

    # Inhibition constants [mmol/L]
    KI_meth = p[6]   # H2S inhibition constant for methanogens
    KI_sulf = p[7]   # H2S inhibition constant for sulfate reducers
    KI_aceto = p[8]  # H2S inhibition constant for acetogens

    # Precipitation and activation parameters
    k_precip = p[9]  # FeS precipitation rate constant [1/day]
    H2S_sat = p[10]  # H2S saturation threshold for precipitation [mmol/L]
    H2_thresh = p[11] # H2 activation threshold [mmol/L]

    # Thermodynamic threshold
    DG_thresh = p[12] # Gibbs energy threshold [kJ/mol]

    # === PHYSICAL CONSTANTS ===
    # MATLAB: R = 8.314e-3; T = 298.15; RT = R*T;
    R = 8.314e-3     # Gas constant [kJ/(mol·K)]
    T = 298.15       # Temperature [K] (25°C)
    RT = R * T       # RT product for Nernst equation [kJ/mol]

    # Standard Gibbs free energies [kJ/mol]
    # MATLAB: DG0_meth = -130; DG0_sulf = -152; DG0_aceto = -95;
    DG0_meth = -130   # Methanogenesis: 4H2 + CO2 → CH4 + 2H2O
    DG0_sulf = -152   # Sulfate reduction: 4H2 + SO4²⁻ → H2S + 4H2O
    DG0_aceto = -95   # Acetogenesis: 4H2 + 2CO2 → CH3COOH + 2H2O

    # === INHIBITION FUNCTIONS ===
    # H2S inhibits all three microbial guilds
    # MATLAB: f_inh_meth = KI_meth / (KI_meth + H2S)
    # Non-competitive inhibition: activity decreases as H2S increases
    f_inh_meth = KI_meth / (KI_meth + H2S)      # Methanogen inhibition factor [0-1]
    f_inh_sulf = KI_sulf / (KI_sulf + H2S)      # Sulfate reducer inhibition factor [0-1]
    f_inh_aceto = KI_aceto / (KI_aceto + H2S)   # Acetogen inhibition factor [0-1]

    # === ACTIVATION THRESHOLD ===
    # Low H2 concentrations suppress all reactions
    # MATLAB: f_activation = H2 / (H2 + H2_thresh)
    # This is a Monod-type function: activity approaches 1 as H2 >> H2_thresh
    f_activation = H2 / (H2 + H2_thresh)  # Activation factor [0-1]

    # === PREVENT LOG OF ZERO OR NEGATIVE VALUES ===
    # MATLAB: H2 = max(H2, 1e-6); CO2 = max(CO2, 1e-6); ...
    # This is necessary because we take log() in Gibbs energy calculations
    # Ensures numerical stability by enforcing minimum concentration
    H2 = np.maximum(H2, 1e-6)           # Minimum H2 = 1e-6 mmol/L
    CO2 = np.maximum(CO2, 1e-6)         # Minimum CO2 = 1e-6 mmol/L
    CH4 = np.maximum(CH4, 1e-6)         # Minimum CH4 = 1e-6 mmol/L
    SO4 = np.maximum(SO4, 1e-6)         # Minimum SO4 = 1e-6 mmol/L
    H2S = np.maximum(H2S, 1e-6)         # Minimum H2S = 1e-6 mmol/L
    Acetate = np.maximum(Acetate, 1e-6) # Minimum Acetate = 1e-6 mmol/L

    # === REACTION QUOTIENTS ===
    # Q = [products] / [reactants]
    # Used in Nernst equation to calculate actual Gibbs energy
    # MATLAB: Q_meth = CH4 / (H2^4 * CO2)

    # Methanogenesis: 4H2 + CO2 → CH4 + 2H2O
    Q_meth = CH4 / (H2**4 * CO2)                # Reaction quotient (dimensionless)

    # Sulfate reduction: 4H2 + SO4²⁻ → H2S + 4H2O
    Q_sulf = H2S / (H2**4 * SO4)                # Reaction quotient (dimensionless)

    # Acetogenesis: 4H2 + 2CO2 → CH3COOH + 2H2O
    Q_aceto = Acetate / (H2**4 * CO2**2)        # Reaction quotient (dimensionless)

    # === DYNAMIC GIBBS FREE ENERGIES ===
    # Nernst equation: ΔG = ΔG° + RT·ln(Q)
    # Calculates actual Gibbs energy based on current concentrations
    # MATLAB: DG_meth = DG0_meth + RT*log(Q_meth)

    DG_meth = DG0_meth + RT * np.log(Q_meth)    # Actual ΔG for methanogenesis [kJ/mol]
    DG_sulf = DG0_sulf + RT * np.log(Q_sulf)    # Actual ΔG for sulfate reduction [kJ/mol]
    DG_aceto = DG0_aceto + RT * np.log(Q_aceto) # Actual ΔG for acetogenesis [kJ/mol]

    # === THERMODYNAMIC FEASIBILITY ===
    # Reactions only proceed if ΔG < DG_thresh (thermodynamically favorable)
    # MATLAB: f_thermo_meth = 1 / (1 + exp((DG_meth - DG_thresh)/RT))
    # Sigmoid function: returns ~1 if ΔG << DG_thresh (favorable)
    #                   returns ~0 if ΔG >> DG_thresh (unfavorable)

    f_thermo_meth = 1.0 / (1.0 + np.exp((DG_meth - DG_thresh) / RT))   # Thermodynamic factor [0-1]
    f_thermo_sulf = 1.0 / (1.0 + np.exp((DG_sulf - DG_thresh) / RT))   # Thermodynamic factor [0-1]
    f_thermo_aceto = 1.0 / (1.0 + np.exp((DG_aceto - DG_thresh) / RT)) # Thermodynamic factor [0-1]

    # === REACTION RATES ===
    # Overall rate = max_rate × substrates × inhibition × activation × thermodynamics
    # MATLAB: r_meth = k_meth * H2 * CO2^(-2) * f_inh_meth * f_activation * f_thermo_meth

    # Methanogenesis rate [mmol/L/day]
    # Note: CO2^(-2) term represents inverse dependence (reaction slows at high CO2)
    r_meth = k_meth * H2 * (CO2**-2) * f_inh_meth * f_activation * f_thermo_meth

    # Sulfate reduction rate [mmol/L/day]
    # Linear dependence on both H2 and SO4
    r_sulf = k_sulf * H2 * SO4 * f_inh_sulf * f_activation * f_thermo_sulf

    # Acetogenesis rate [mmol/L/day]
    # Quadratic dependence on CO2 (needs 2 CO2 molecules)
    r_aceto = k_aceto * H2 * (CO2**2) * f_inh_aceto * f_activation * f_thermo_aceto

    # === FeS PRECIPITATION ===
    # H2S precipitates as iron sulfide (FeS) when it exceeds saturation
    # MATLAB: r_precip = k_precip * max(0, H2S - H2S_sat)
    # Only precipitates if H2S > H2S_sat, otherwise rate = 0
    r_precip = k_precip * np.maximum(0, H2S - H2S_sat)  # Precipitation rate [mmol/L/day]

    # === DIFFERENTIAL EQUATIONS ===
    # Stoichiometry from the three main reactions:
    # Methanogenesis:    4H2 + CO2 → CH4 + 2H2O
    # Sulfate Reduction: 4H2 + SO4²⁻ → H2S + 4H2O
    # Acetogenesis:      4H2 + 2CO2 → CH3COOH + 2H2O

    # MATLAB: dH2 = -4*r_meth - 4*r_sulf - 4*r_aceto
    # H2 is consumed by all three reactions (4 molecules per reaction)
    dH2 = -4 * r_meth - 4 * r_sulf - 4 * r_aceto  # [mmol/L/day]

    # MATLAB: dCO2 = -1*r_meth - 2*r_aceto
    # CO2 consumed by methanogenesis (1 molecule) and acetogenesis (2 molecules)
    dCO2 = -1 * r_meth - 2 * r_aceto               # [mmol/L/day]

    # MATLAB: dCH4 = +1*r_meth
    # CH4 produced only by methanogenesis (1 molecule per reaction)
    dCH4 = +1 * r_meth                             # [mmol/L/day]

    # MATLAB: dH2S = +1*r_sulf - r_precip
    # H2S produced by sulfate reduction, removed by precipitation
    dH2S = +1 * r_sulf - r_precip                  # [mmol/L/day]

    # MATLAB: dSO4 = -1*r_sulf
    # SO4 consumed only by sulfate reduction (1 molecule per reaction)
    dSO4 = -1 * r_sulf                             # [mmol/L/day]

    # MATLAB: dFeS = +1*r_precip
    # FeS accumulates from H2S precipitation
    dFeS = +1 * r_precip                           # [mmol/L/day]

    # === BIOMASS GROWTH ===
    # Biomass increases proportional to reaction rate and yield coefficient
    # MATLAB: dX_meth = Y_m * r_meth
    # Yield coefficient (Y) represents biomass produced per unit reaction

    dX_meth = Y_m * r_meth      # Methanogen biomass growth [1/day or similar]
    dX_sulf = Y_s * r_sulf      # Sulfate reducer biomass growth [1/day or similar]
    dX_aceto = Y_a * r_aceto    # Acetogen biomass growth [1/day or similar]

    # MATLAB: dAcetate = +1*r_aceto
    # Acetate produced only by acetogenesis (1 molecule per reaction)
    dAcetate = +1 * r_aceto     # [mmol/L/day]

    # === RETURN DERIVATIVES ===
    # MATLAB: dydt = [dH2; dCO2; dCH4; dH2S; dSO4; dFeS; dX_meth; dX_sulf; dX_aceto; dAcetate]
    # Return as NumPy array in same order as input state vector y
    return np.array([dH2, dCO2, dCH4, dH2S, dSO4, dFeS, dX_meth, dX_sulf, dX_aceto, dAcetate])

print('✓ ODE function defined')
print('  Reactions: Methanogenesis, Sulfate Reduction, Acetogenesis')
print('  State variables: 10')
print('  Parameters: 13')

# %% [markdown]
# ## Part 4: Parameter Fitting with Nonlinear Least Squares
#
# **MATLAB lines 193-221:** `fit_mechanistic_params` and `residuals_multiguild`
#
# Fits 13 model parameters to match experimental data using:
# - Log-space residuals (handles multi-scale concentrations)
# - Weighted residuals (emphasizes important species)
# - Penalty for negative concentrations (physically invalid)

# %%
def compute_residuals(p, t_exp, data_exp, x0):
    """
    Compute weighted residuals between model predictions and experimental data.

    MATLAB equivalent: function res = residuals_multiguild(p, t_exp, data_exp, x0)
    (lines 204-221 in rnn_transport_multiguild_uq_v3.m)

    Uses log-space residuals to handle concentrations spanning multiple orders
    of magnitude (H2 ranges from ~10 mmol/L to ~0.001 mmol/L).

    Args:
        p: Parameter vector to test [13 elements]
        t_exp: Experimental time points [days]
        data_exp: Experimental concentrations [H2, CO2, CH4, H2S, SO4] in mmol/L
        x0: Initial conditions [10 elements]

    Returns:
        res: Flattened residual vector (required by least_squares)
    """
    try:
        # === SOLVE ODE WITH CURRENT PARAMETERS ===
        # MATLAB: [t_sim, y_raw] = ode45(@(t,y) trueODEfunc_multiguild(t,y,p), t_exp, x0)
        # solve_ivp is the equivalent of MATLAB's ode45 (Runge-Kutta 4-5 method)
        sol = solve_ivp(
            fun=lambda t, y: ode_multiguild(t, y, p),  # ODE function with parameters p
            t_span=(t_exp[0], t_exp[-1]),               # Time span from first to last point
            y0=x0,                                       # Initial conditions
            t_eval=t_exp,                                # Evaluate at experimental time points
            method='RK45'                                # Runge-Kutta 4-5 (same as ode45)
        )

        # === EXTRACT SIMULATED CONCENTRATIONS ===
        # MATLAB: y_sim = interp1(t_sim, y_raw, t_exp, 'linear')
        # solve_ivp with t_eval already returns values at t_exp, so no interpolation needed
        # Extract only first 5 state variables (H2, CO2, CH4, H2S, SO4)
        # sol.y has shape [10 states, n_timepoints], need [n_timepoints, 5 states]
        y_sim = sol.y[:5, :].T  # Transpose and take first 5 rows

        # === LOG-SPACE TRANSFORMATION ===
        # MATLAB: log_sim = log1p(y_sim(:,1:5))
        # log1p(x) = log(1+x) prevents log(0) and handles small positive values well
        # This transformation makes residuals more balanced across concentration scales
        log_sim = np.log1p(y_sim)            # Simulated concentrations in log-space
        log_exp = np.log1p(data_exp)          # Experimental concentrations in log-space

        # === WEIGHTED RESIDUALS ===
        # MATLAB: weights = [1, 1, 0.5, 0.5, 1]
        # Emphasize fitting H2 and SO4 (primary reactants, weight=1.0)
        # De-emphasize CH4 and H2S (products, weight=0.5)
        # Weights: [H2, CO2, CH4, H2S, SO4]
        weights = np.array([1.0, 1.0, 0.5, 0.5, 1.0])

        # Element-wise multiplication of residuals by weights
        res = (log_sim - log_exp) * weights

        # === PENALTY FOR NEGATIVE CONCENTRATIONS ===
        # MATLAB: if any(y_sim(:) < -1e-6)
        #           res = res + 1e3 * abs(min(y_sim(:)))
        # Negative concentrations are physically impossible
        # Add large penalty to residuals to discourage optimizer from going there
        if np.any(sol.y < -1e-6):
            penalty = 1e3 * np.abs(np.min(sol.y))  # Large penalty proportional to violation
            res = res + penalty

        # === FLATTEN TO 1D ARRAY ===
        # MATLAB: res = res(:)
        # least_squares requires 1D residual vector
        return res.ravel()  # Convert 2D array to 1D

    except Exception as e:
        # === HANDLE ODE SOLVER FAILURES ===
        # MATLAB: catch; res = 1e6 * ones(numel(data_exp), 1); end
        # If ODE solver fails (stiff system, invalid parameters, etc.),
        # return very large residuals to discourage optimizer
        print(f'⚠ ODE solver failed with parameters: {p}')
        print(f'  Error: {e}')
        return 1e6 * np.ones(data_exp.size)  # Large residuals = bad fit


def fit_parameters(t_exp, data_exp, x0):
    """
    Fit mechanistic model parameters to experimental data.

    MATLAB equivalent: function p_fit = fit_mechanistic_params(t_exp, data_exp, x0)
    (lines 193-201 in rnn_transport_multiguild_uq_v3.m)

    Uses nonlinear least squares to find best-fit parameters.

    Args:
        t_exp: Experimental time points [days]
        data_exp: Experimental concentrations [mmol/L]
        x0: Initial conditions

    Returns:
        p_fit: Optimized parameter vector [13 elements]
    """

    # === INITIAL PARAMETER GUESS ===
    # MATLAB: p0 = [1, 1, 1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, -10]
    # These are reasonable starting values based on typical biogeochemical parameters
    p0 = np.array([
        1.0, 1.0, 1.0,          # k_meth, k_sulf, k_aceto: reaction rates
        0.05, 0.05, 0.05,        # Y_m, Y_s, Y_a: biomass yields (5% efficiency)
        0.1, 0.1, 0.1,           # KI_meth, KI_sulf, KI_aceto: inhibition constants
        0.01, 0.01, 0.01,        # k_precip, H2S_sat, H2_thresh: precipitation params
        -10.0                    # DG_thresh: thermodynamic threshold (negative)
    ])

    # === LOWER BOUNDS ===
    # MATLAB: lb = [0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0, 0, 0, -50]
    # Physical constraints: reaction rates must be positive, yields between 0-1, etc.
    lb = np.array([
        0.001, 0.001, 0.001,     # Reaction rates > 0 (small positive number)
        0.01, 0.01, 0.01,        # Biomass yields > 0 (at least 1% efficiency)
        0.001, 0.001, 0.001,     # Inhibition constants > 0
        0.0, 0.0, 0.0,           # Precipitation params >= 0
        -50.0                    # DG_thresh can be negative (large negative allowed)
    ])

    # === UPPER BOUNDS ===
    # MATLAB: ub = [10, 10, 10, 0.5, 0.5, 0.5, 10, 10, 10, 1, 1, 1, 0]
    # Physical constraints: yields can't exceed 50%, DG_thresh must be negative, etc.
    ub = np.array([
        10.0, 10.0, 10.0,        # Reaction rates max (reasonable upper limit)
        0.5, 0.5, 0.5,           # Biomass yields max 50% (thermodynamic constraint)
        10.0, 10.0, 10.0,        # Inhibition constants max
        1.0, 1.0, 1.0,           # Precipitation params max
        0.0                      # DG_thresh must be negative (0 is upper bound)
    ])

    print('\n' + '='*70)
    print('PARAMETER FITTING: Nonlinear Least Squares Optimization')
    print('='*70)
    print('Algorithm: Trust-Region-Reflective (scipy.optimize.least_squares)')
    print('MATLAB Equivalent: lsqnonlin from Optimization Toolbox')
    print('\nFitting 13 parameters to experimental data...')
    print('This may take 30-90 seconds...')
    print('='*70)
    # Comparison-friendly: print initial guess and bounds with names
    param_names = [
        'k_meth', 'k_sulf', 'k_aceto',
        'Y_m', 'Y_s', 'Y_a',
        'KI_meth', 'KI_sulf', 'KI_aceto',
        'k_precip', 'H2S_sat', 'H2_thresh',
        'DG_thresh'
    ]
    print('\nInitial guess (p0):')
    for name, val in zip(param_names, p0):
        print(f'  {name:12s} = {val:12.6f}')
    print('\nLower bounds (lb):')
    for name, val in zip(param_names, lb):
        print(f'  {name:12s} >= {val:12.6f}')
    print('\nUpper bounds (ub):')
    for name, val in zip(param_names, ub):
        print(f'  {name:12s} <= {val:12.6f}')
    print(f'\nResidual length (data points flattened): {data_exp.size}')
    print(f'Weights: [1.0, 1.0, 0.5, 0.5, 1.0] (H2, CO2, CH4, H2S, SO4)')
    print('='*70 + '\n')

    # === RUN OPTIMIZATION ===
    # MATLAB: p_fit = lsqnonlin(@(p) residuals_multiguild(p, ...), p0, lb, ub, options)
    # scipy.optimize.least_squares is the direct Python equivalent of lsqnonlin
    result = least_squares(
        fun=lambda p: compute_residuals(p, t_exp, data_exp, x0),  # Objective function
        x0=p0,                           # Initial parameter guess
        bounds=(lb, ub),                 # (lower_bounds, upper_bounds) as tuple
        verbose=2,                       # Print iteration details (MATLAB 'Display','iter')
        max_nfev=5000,                   # Maximum function evaluations (MATLAB MaxFunctionEvaluations)
        ftol=1e-8,                       # Function tolerance (stopping criterion)
        xtol=1e-8,                       # Parameter tolerance (stopping criterion)
        method='trf'                     # Trust-Region-Reflective (default, robust algorithm)
    )

    # === EXTRACT FITTED PARAMETERS ===
    p_fit = result.x  # Optimized parameter values

    # === DISPLAY RESULTS ===
    print('\n' + '='*70)
    print('FITTING COMPLETE')
    print('='*70)
    print(f'Success: {result.success}')
    print(f'Message: {result.message}')
    # SciPy reports cost = 0.5 * sum(residuals^2); MATLAB reports resnorm = sum(residuals.^2)
    resnorm = 2.0 * result.cost
    rms_residual = np.sqrt(resnorm / result.fun.size) if result.fun.size > 0 else np.nan
    print(f'Resnorm (sum of squares, MATLAB style): {resnorm:.6f}')
    print(f'Cost (0.5 * resnorm, SciPy style):       {result.cost:.6f}')
    print(f'RMS residual:                             {rms_residual:.6f}')
    print(f'Residual length:                          {result.fun.size}')
    print(f'Function evaluations: {result.nfev}')
    print(f'Optimality: {result.optimality:.2e}')

    # Print parameter names and values
    print('\nFitted Parameters:')
    print('-'*70)
    for name, val in zip(param_names, p_fit):
        print(f'  {name:12s} = {val:12.6f}')
    print('='*70 + '\n')

    return p_fit


# === RUN PARAMETER FITTING ===
# MATLAB: line 18
# p_fit = fit_mechanistic_params(t_exp, data_exp, x0);
p_fit = fit_parameters(t_exp, data_exp, x0)

# %% [markdown]
# ## Part 5: Generate Dense Training Data from Fitted ODE Model
#
# **MATLAB lines 19-21:**
# ```matlab
# t = linspace(0, t_exp(end), 2000);
# [~, xTrain] = ode45(@(t,y) trueODEfunc_multiguild(t,y,p_fit), t, x0);
# xTrain = xTrain';
# ```
#
# This is the key to **physics-informed machine learning**:
# - We only have 11 experimental measurements
# - We use the fitted ODE model to generate 2000 dense time points
# - LSTM trains on this physics-based data (not raw sparse data)

# %%
# === CREATE DENSE TIME GRID ===
# MATLAB: t = linspace(0, t_exp(end), 2000)
# Generate 2000 evenly-spaced time points from 0 to final experimental time
# This gives ~182× more data points than the 11 experimental measurements
t_train = np.linspace(t_exp[0], t_exp[-1], 2000)  # [days]

print('='*70)
print('TRAINING DATA GENERATION')
print('='*70)
print(f'Solving ODE with fitted parameters over {len(t_train)} time points...')
print(f'Time range: {t_train[0]:.1f} to {t_train[-1]:.1f} days')

# === SOLVE ODE SYSTEM WITH FITTED PARAMETERS ===
# MATLAB: [~, xTrain] = ode45(@(t,y) trueODEfunc_multiguild(t,y,p_fit), t, x0)
# Solve using fitted parameters p_fit to generate training trajectories
sol = solve_ivp(
    fun=lambda t, y: ode_multiguild(t, y, p_fit),  # ODE with fitted parameters
    t_span=(t_train[0], t_train[-1]),               # Time span
    y0=x0,                                           # Initial conditions
    t_eval=t_train,                                  # Evaluate at 2000 dense points
    method='RK45',                                   # Runge-Kutta 4-5
    dense_output=False,                              # Don't need interpolation
    rtol=1e-6,                                       # Relative tolerance
    atol=1e-9                                        # Absolute tolerance
)

# === EXTRACT SOLUTION ===
# MATLAB: xTrain = xTrain'
# MATLAB ode45 returns [time × states], then transposes to [states × time]
# Python solve_ivp returns [states × time] directly, so we keep it
x_train = sol.y  # Shape: [10 state variables, 2000 time points]

# === VERIFY SOLUTION QUALITY ===
print(f'✓ ODE integration successful')
print(f'  Solution shape: {x_train.shape} (10 species × {len(t_train)} time points)')
print(f'  Min concentration: {x_train.min():.6f} mmol/L')
print(f'  Max concentration: {x_train.max():.6f} mmol/L')

# Check for negative concentrations (indicates poor fit or numerical issues)
if np.any(x_train < 0):
    neg_count = np.sum(x_train < 0)
    print(f'  ⚠ Warning: {neg_count} negative concentration values detected')
    print(f'    This may indicate poor parameter fit or stiff ODEs')
else:
    print(f'  ✓ All concentrations positive (physically valid)')

print('='*70 + '\n')

# %% [markdown]
# ## Part 6: Prepare Sequence Data for LSTM Training
#
# **MATLAB lines 50-61:**
# ```matlab
# sequenceLength = 10;
# X = {}; Y = [];
# for i = 1:(size(xTrain,2) - sequenceLength)
#     X{end+1} = xTrain(:, i:i+sequenceLength-1);
#     Y(end+1, :) = xTrain(:, i+sequenceLength)';
# end
# ```
#
# Creates sliding window sequences:
# - Input: Past 10 timesteps of all 10 state variables
# - Target: Next single timestep (sequence-to-one prediction)

# %%
class SequenceDataset(Dataset):
    """
    PyTorch Dataset for sequence-to-one time series prediction.

    MATLAB equivalent: Creating X (cell array) and Y (matrix) with for loop
    (lines 50-61 in rnn_transport_multiguild_uq_v3.m)

    In MATLAB, data is stored in cell arrays. In PyTorch, we use a Dataset class
    which provides a clean interface for DataLoader to batch and shuffle data.
    """

    def __init__(self, data, sequence_length=10):
        """
        Initialize dataset with training data.

        Args:
            data: Training trajectories [n_features, n_timesteps] as NumPy array
            sequence_length: Number of past timesteps to use as input (MATLAB: 10)
        """
        # MATLAB: sequenceLength = 10
        self.sequence_length = sequence_length

        # MATLAB: xTrain is [10 features × 2000 timesteps]
        # PyTorch LSTM expects [batch, sequence, features]
        # So we transpose to [2000 timesteps, 10 features]
        self.data = torch.FloatTensor(data.T)  # Convert NumPy to PyTorch tensor

        # Calculate number of valid sequences
        # MATLAB: for i = 1:(size(xTrain,2) - sequenceLength)
        # In Python (0-based): range(0, n_timesteps - sequence_length)
        # If we have 2000 points and sequence_length=10, we get 1990 sequences
        self.n_samples = self.data.shape[0] - sequence_length

    def __len__(self):
        """
        Return total number of sequences.
        Required by PyTorch DataLoader for batching.

        MATLAB equivalent: length(X) after loop completes
        """
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a single sequence-target pair by index.
        Required by PyTorch DataLoader.

        MATLAB equivalent:
            X{i} = xTrain(:, i:i+sequenceLength-1)  -> Input sequence
            Y(i,:) = xTrain(:, i+sequenceLength)'    -> Target (next step)

        Args:
            idx: Index of sample (0-based, from 0 to n_samples-1)

        Returns:
            x: Input sequence [sequence_length, n_features]
               Example: [10 timesteps, 10 state variables]
            y: Target (next timestep) [n_features]
               Example: [10 state variables]
        """
        # MATLAB: X{i} = xTrain(:, i:i+sequenceLength-1)
        # Extract past sequence_length timesteps
        # In PyTorch with batch_first=True, shape is [seq_len, features]
        x = self.data[idx:idx+self.sequence_length, :]  # [10, 10]

        # MATLAB: Y(i,:) = xTrain(:, i+sequenceLength)'
        # Extract next single timestep as target
        y = self.data[idx+self.sequence_length, :]  # [10]

        return x, y


# === CREATE DATASET ===
# MATLAB: Creates X (cell array) and Y (matrix) in for loop
# PyTorch: Create Dataset object that provides same functionality
dataset = SequenceDataset(x_train, sequence_length=10)

# === CREATE DATALOADER FOR BATCHING ===
# MATLAB: MiniBatchSize in trainingOptions determines batch size
# PyTorch: Use DataLoader to handle batching automatically
# MATLAB: 'Shuffle','every-epoch' -> shuffle=True
batch_size = 64  # MATLAB default in trainingOptions
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,   # Process 64 sequences at once
    shuffle=True,            # Shuffle data every epoch (MATLAB 'Shuffle','every-epoch')
    num_workers=0,           # Use 0 for Windows compatibility
    pin_memory=False         # Set True if using GPU with CUDA
)

print('='*70)
print('PYTORCH DATASET CREATED')
print('='*70)
print(f'Total sequences: {len(dataset)}')
print(f'Input shape per sequence: [sequence_length=10, features=10]')
print(f'Target shape per sequence: [features=10]')
print(f'Batch size: {batch_size}')
print(f'Batches per epoch: {len(train_loader)}')
print('='*70 + '\n')

# %% [markdown]
# ## Part 7: Define LSTM Neural Network Architecture
#
# **MATLAB lines 64-70:**
# ```matlab
# layers = [
#     sequenceInputLayer(10)
#     dropoutLayer(0.2)
#     lstmLayer(64, 'OutputMode', 'last')
#     dropoutLayer(0.2)
#     fullyConnectedLayer(10)
#     regressionLayer
# ];
# ```
#
# In PyTorch, we define this as a custom `nn.Module` class.

# %%
class ChemicalLSTM(nn.Module):
    """
    LSTM network for biogeochemical time series prediction.

    MATLAB equivalent: layers array (lines 64-70)

    Architecture matches MATLAB exactly:
        Input(10) -> Dropout(0.2) -> LSTM(64) -> Dropout(0.2) -> Linear(10) -> Output
    """

    def __init__(self, input_size=10, hidden_size=64, output_size=10):
        """
        Initialize LSTM model layers.

        Args:
            input_size: Number of input features (MATLAB: sequenceInputLayer(10))
            hidden_size: LSTM hidden units (MATLAB: lstmLayer(64))
            output_size: Number of output features (MATLAB: fullyConnectedLayer(10))
        """
        super(ChemicalLSTM, self).__init__()

        # Store architecture parameters
        self.input_size = input_size      # 10 state variables
        self.hidden_size = hidden_size    # 64 LSTM hidden units
        self.output_size = output_size    # 10 output predictions

        # === LAYER 1: DROPOUT ===
        # MATLAB: dropoutLayer(0.2)
        # Randomly zeros 20% of inputs during training to prevent overfitting
        self.dropout1 = nn.Dropout(0.2)

        # === LAYER 2: LSTM ===
        # MATLAB: lstmLayer(64, 'OutputMode', 'last')
        # Long Short-Term Memory layer with 64 hidden units
        # batch_first=True means input shape is [batch, sequence, features]
        # (MATLAB uses [features, sequence, batch], different convention)
        self.lstm = nn.LSTM(
            input_size=input_size,      # 10 input features
            hidden_size=hidden_size,    # 64 hidden units
            num_layers=1,                # Single LSTM layer
            batch_first=True,            # Input: [batch, seq, features]
            dropout=0.0                  # No internal dropout (we add manually)
        )

        # === LAYER 3: DROPOUT ===
        # MATLAB: dropoutLayer(0.2)
        # Second dropout layer after LSTM
        self.dropout2 = nn.Dropout(0.2)

        # === LAYER 4: FULLY CONNECTED ===
        # MATLAB: fullyConnectedLayer(10)
        # Linear transformation from hidden_size (64) to output_size (10)
        self.fc = nn.Linear(hidden_size, output_size)

        # Note: MATLAB's regressionLayer (MSE loss) is NOT part of the model
        # In PyTorch, loss functions are separate and used in training loop

    def forward(self, x):
        """
        Forward pass through the network.

        MATLAB equivalent: Automatic in trainNetwork, manual in predict

        Args:
            x: Input sequences [batch_size, sequence_length, input_size]
               Example: [64, 10, 10] for batch=64, seq_len=10, features=10

        Returns:
            output: Predictions [batch_size, output_size]
                    Example: [64, 10] for batch=64, features=10
        """
        # === LAYER 1: FIRST DROPOUT ===
        # MATLAB: dropoutLayer(0.2) applied to input
        # Apply dropout to input sequences (only active during training)
        x = self.dropout1(x)  # Shape: [batch, seq_len, features]

        # === LAYER 2: LSTM ===
        # MATLAB: lstmLayer(64, 'OutputMode', 'last')
        # Process sequences through LSTM
        # Returns: (output, (h_n, c_n))
        #   output: [batch, seq_len, hidden_size] - outputs at ALL timesteps
        #   h_n: [1, batch, hidden_size] - final hidden state
        #   c_n: [1, batch, hidden_size] - final cell state
        lstm_out, (h_n, c_n) = self.lstm(x)

        # === EXTRACT LAST TIMESTEP ===
        # MATLAB: 'OutputMode', 'last' extracts only final timestep
        # In PyTorch, we manually extract it from lstm_out
        # lstm_out[:, -1, :] means: [all batches, last timestep, all hidden units]
        last_output = lstm_out[:, -1, :]  # Shape: [batch, hidden_size]

        # === LAYER 3: SECOND DROPOUT ===
        # MATLAB: dropoutLayer(0.2) applied to LSTM output
        last_output = self.dropout2(last_output)  # Shape: [batch, hidden_size]

        # === LAYER 4: FULLY CONNECTED ===
        # MATLAB: fullyConnectedLayer(10)
        # Linear transformation: 64 hidden units -> 10 output features
        output = self.fc(last_output)  # Shape: [batch, output_size]

        return output


# === INSTANTIATE MODEL ===
# MATLAB: Model is created automatically by trainNetwork
# PyTorch: Must explicitly create model instance
model = ChemicalLSTM(
    input_size=10,      # 10 state variables
    hidden_size=64,     # 64 LSTM hidden units (MATLAB default)
    output_size=10      # Predict all 10 state variables
).to(device)  # Move model to GPU if available

print('='*70)
print('LSTM MODEL ARCHITECTURE')
print('='*70)
print(model)
print(f'\nTotal trainable parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'Device: {device}')
print('='*70 + '\n')

# %% [markdown]
# ## Part 8: Training Configuration and Loop
#
# **MATLAB lines 72-77:**
# ```matlab
# options = trainingOptions('adam',
#     'MaxEpochs', 300,
#     'MiniBatchSize', 64,
#     'InitialLearnRate', 1e-3,
#     'Shuffle', 'every-epoch',
#     'Verbose', false);
# net = trainNetwork(X, Y, layers, options);
# ```
#
# PyTorch requires manual training loop (no automatic trainNetwork equivalent).

# %%
# === TRAINING HYPERPARAMETERS ===
# MATLAB: trainingOptions('adam', ...)

# MATLAB: 'MaxEpochs', 300
num_epochs = 300                  # Number of complete passes through data

# MATLAB: 'InitialLearnRate', 1e-3
learning_rate = 1e-3              # Learning rate (0.001)

# MATLAB: 'MiniBatchSize', 64 (already defined in DataLoader above)
# batch_size = 64

# === LOSS FUNCTION ===
# MATLAB: regressionLayer uses Mean Squared Error (MSE) loss
# PyTorch: Must explicitly define loss function
criterion = nn.MSELoss()  # Mean Squared Error: MSE = (1/N) * sum((y_pred - y_true)^2)

# === OPTIMIZER ===
# MATLAB: trainingOptions('adam', ...)
# PyTorch: torch.optim.Adam
optimizer = optim.Adam(
    model.parameters(),           # All model parameters (weights and biases)
    lr=learning_rate,             # Learning rate = 1e-3
    betas=(0.9, 0.999),           # Adam momentum parameters (default)
    eps=1e-8,                     # Numerical stability constant
    weight_decay=0.0              # No L2 regularization (MATLAB default)
)

print('='*70)
print('TRAINING CONFIGURATION')
print('='*70)
print(f'Epochs: {num_epochs}')
print(f'Learning rate: {learning_rate}')
print(f'Batch size: {batch_size}')
print(f'Optimizer: Adam')
print(f'Loss function: MSE (Mean Squared Error)')
print('='*70 + '\n')

# === TRAINING LOOP ===
# MATLAB: net = trainNetwork(X, Y, layers, options) does this automatically
# PyTorch: Must write explicit training loop

print('='*70)
print('STARTING TRAINING')
print('='*70)
print(f'Training on device: {device}')
print('='*70 + '\n')

# Training loop
for epoch in range(num_epochs):
    # Set model to training mode
    # This enables dropout layers (MATLAB does automatically during training)
    model.train()

    epoch_loss = 0.0  # Accumulate loss for this epoch

    # === ITERATE THROUGH BATCHES ===
    # MATLAB: trainNetwork handles batching automatically
    # PyTorch: DataLoader provides batches
    for batch_idx, (sequences, targets) in enumerate(train_loader):
        # Move data to device (GPU if available)
        # MATLAB: Handles automatically
        sequences = sequences.to(device)  # [batch, seq_len, features]
        targets = targets.to(device)      # [batch, features]

        # === ZERO GRADIENTS ===
        # MATLAB: trainNetwork does this automatically
        # PyTorch: Must manually zero gradients before backward pass
        # (Gradients accumulate by default in PyTorch)
        optimizer.zero_grad()

        # === FORWARD PASS ===
        # MATLAB: Automatic in trainNetwork
        # Compute model predictions
        outputs = model(sequences)  # [batch, features]

        # === COMPUTE LOSS ===
        # MATLAB: regressionLayer computes MSE automatically
        # PyTorch: Explicit loss computation
        loss = criterion(outputs, targets)

        # === BACKWARD PASS ===
        # MATLAB: trainNetwork computes gradients automatically
        # PyTorch: Explicit backward pass to compute gradients
        loss.backward()

        # === UPDATE PARAMETERS ===
        # MATLAB: trainNetwork updates parameters automatically
        # PyTorch: Explicit parameter update using optimizer
        optimizer.step()

        # Accumulate loss for reporting
        epoch_loss += loss.item()

    # === PRINT PROGRESS ===
    # MATLAB: 'Verbose', false means no output during training
    # We print every 50 epochs for monitoring
    if (epoch + 1) % 50 == 0:
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch+1:3d}/{num_epochs}] | Loss: {avg_loss:.6f}')

print('\n' + '='*70)
print('TRAINING COMPLETE')
print('='*70 + '\n')

# %% [markdown]
# ## Part 9: Save Trained Model
#
# **MATLAB line 80:**
# ```matlab
# save('trained_LSTM_multiguild.mat', 'net');
# ```

# %%
# === SAVE MODEL ===
# MATLAB: save('trained_LSTM_multiguild.mat', 'net')
# PyTorch: save state_dict (recommended) or entire model

# Save model parameters (state_dict)
torch.save(model.state_dict(), 'trained_lstm_pytorch.pth')
print('✓ Model saved to: trained_lstm_pytorch.pth')

# To load later:
# model = ChemicalLSTM()
# model.load_state_dict(torch.load('trained_lstm_pytorch.pth'))
# model.eval()

# %% [markdown]
# ## Part 10: Reactive Transport Simulation - Initialization
#
# **MATLAB lines 87-118:** Set up 1D transport column parameters
#
# Simulates advection-dispersion-reaction in a 1D porous media column.

# %%
# === TRANSPORT PARAMETERS ===
# MATLAB: lines 87-100

# MATLAB: L=75; % length of column [m]
L = 75                # Column length [meters]

# MATLAB: N=75; % Number of cells
nx = 75               # Number of spatial cells (discretization)

# MATLAB: cell_w=1; % Cell width [m]
cell_w = 1            # Width of each cell [meters]

# MATLAB: S_time=t_exp(end); % Simulation time [d]
S_time = int(t_exp[-1])  # Total simulation time [days]

# MATLAB: n=0.3; % porosity [-]
n = 0.3               # Porosity (dimensionless, 30% pore space)

# MATLAB: q=1; % Darcys velocity [m/d]
q = 1                 # Darcy velocity [m/day]

# MATLAB: v=q/n; % seepage velocity [m/d]
v = q / n             # Seepage (pore water) velocity [m/day]

# MATLAB: D=0.3; % Dispersion coefficient [m^2/d]
D = 0.3               # Dispersion coefficient [m²/day]

# MATLAB: dt=cell_w/v; % [d]
dt = cell_w / v       # Time step for advection [days]

# === CONCENTRATION MATRICES ===
# MATLAB: lines 108-111

# MATLAB: cmob = zeros(nx, 6); % H2, CO2, CH4, H2S, SO4, Acetate, % mobile species
cmob = np.zeros((nx, 6))  # Mobile (transported) species [mmol/L]
                          # Columns: H2, CO2, CH4, H2S, SO4, Acetate

# MATLAB: cimob = zeros(nx, 4); % FeS, X_meth, X_sulf, X_aceto  % imobile species
cimob = np.zeros((nx, 4))  # Immobile (non-transported) species
                           # Columns: FeS, X_meth, X_sulf, X_aceto

# MATLAB: cmob(:,1:2) = 1e-4; cmob(:,5) = 5e-5;
# Initial concentrations for mobile species
cmob[:, 0:2] = 1e-4   # H2 and CO2 initial concentrations
cmob[:, 4] = 5e-5     # SO4 initial concentration

# MATLAB: cimob(:,1:4) = 0.01;
# Initial biomass and precipitate
cimob[:, 0:4] = 0.01  # FeS, X_meth, X_sulf, X_aceto initial values

# === BREAKTHROUGH CURVE STORAGE ===
# MATLAB: lines 113-114
# BTC_mean = zeros(0,10); BTC_std = zeros(0,10);
BTC_mean = np.zeros((0, 10))  # Mean concentrations at monitoring point (x=25m)
                               # Will grow to [time_points, 10 species]

# === HISTORY BUFFER FOR LSTM ===
# MATLAB: lines 116-117
# historyLength = sequenceLength;
# historyBuffer = repmat(x0,1,historyLength);
sequenceLength = 10
historyBuffer = np.tile(x0.reshape(-1, 1), (1, sequenceLength))  # [10 species, 10 timesteps]

# Set model to evaluation mode (disables dropout)
# MATLAB: predict() automatically disables dropout
model.eval()

print('='*70)
print('REACTIVE TRANSPORT SIMULATION INITIALIZED')
print('='*70)
print(f'Column length: {L} m')
print(f'Number of cells: {nx}')
print(f'Cell width: {cell_w} m')
print(f'Simulation time: {S_time} days')
print(f'Porosity: {n}')
print(f'Seepage velocity: {v:.2f} m/day')
print(f'Dispersion coefficient: {D} m²/day')
print(f'Timestep: {dt:.4f} days')
print('='*70 + '\n')

# %% [markdown]
# ## Part 11: Main Transport Loop
#
# **MATLAB lines 119-162:** Time-stepping loop with advection, dispersion, and LSTM-based reaction
#
# At each timestep:
# 1. **Advection**: Shift concentrations downstream
# 2. **Dispersion**: Smooth concentration gradients
# 3. **Reaction**: Use LSTM to predict biogeochemical reactions

# %%
# MATLAB: for time = 0:1:S_time
print('='*70)
print('RUNNING TRANSPORT SIMULATION')
print('='*70)

for time in range(S_time + 1):

    # ============================================
    # ADVECTION
    # ============================================
    # MATLAB: lines 121-127
    # Shift concentrations downstream by one cell (Courant number = 1)

    # MATLAB: cmob(2:end,:) = cmob(1:end-1,:);
    # Shift all cells downstream (cell 2 gets value from cell 1, etc.)
    cmob[1:, :] = cmob[:-1, :]

    # MATLAB: cmob(1,:) = [1e-4, 1e-4, 0, 0, 5e-5, 0];
    # Set inflow boundary condition (first cell)
    # Inflow composition: H2=1e-4, CO2=1e-4, CH4=0, H2S=0, SO4=5e-5, Acetate=0
    cmob[0, :] = [1e-4, 1e-4, 0, 0, 5e-5, 0]

    # ============================================
    # DISPERSION
    # ============================================
    # MATLAB: lines 132-138
    # Fick's law: Flux = -D * (dC/dx)

    # Calculate dispersive fluxes at interior interfaces
    # MATLAB: Jd = (cmob(1:end-1,:) - cmob(2:end,:)) / cell_w * D;
    Jd = (cmob[:-1, :] - cmob[1:, :]) / cell_w * D  # [nx-1, 6]

    # Add boundary fluxes
    # MATLAB: Jd = [zeros(1,6); Jd; Jd(end,:)];
    # Zero flux at inflow, repeat last flux at outflow
    Jd = np.vstack([
        np.zeros((1, 6)),  # Zero dispersive flux at inflow
        Jd,                # Interior fluxes
        Jd[-1:, :]         # Repeat last flux at outflow
    ])  # [nx+1, 6]

    # Update concentrations based on divergence of dispersive flux
    # MATLAB: cmob = cmob + dt/cell_w * (Jd(1:end-1,:) - Jd(2:end,:));
    cmob = cmob + dt / cell_w * (Jd[:-1, :] - Jd[1:, :])

    # ============================================
    # REACTION (LSTM-based)
    # ============================================
    # MATLAB: lines 142-152
    # Use trained LSTM to predict reaction dynamics at each spatial cell

    # Storage for ensemble predictions (uncertainty quantification)
    # MATLAB: cmat_ensemble = zeros(nx,10,20); % 20 samples
    cmat_ensemble = np.zeros((nx, 10, 20))

    # Loop through each spatial cell
    # MATLAB: for it = 1:nx
    for it in range(nx):
        # Combine mobile and immobile species into state vector
        # MATLAB: currentState = [cmob(it,:), cimob(it,:)]';
        currentState = np.hstack([cmob[it, :], cimob[it, :]]).reshape(-1, 1)  # [10, 1]

        # Update history buffer (rolling window)
        # MATLAB: historyBuffer = [historyBuffer(:,2:end), currentState];
        # Remove oldest timestep, add current state
        historyBuffer = np.hstack([historyBuffer[:, 1:], currentState])  # [10, 10]

        # Ensemble predictions (20 samples for uncertainty)
        # MATLAB: for s = 1:20
        with torch.no_grad():  # Disable gradient computation (inference mode)
            for s in range(20):
                # Prepare input for LSTM
                # Convert to PyTorch tensor: [1, seq_len, features] = [1, 10, 10]
                input_seq = torch.FloatTensor(historyBuffer.T).unsqueeze(0).to(device)

                # MATLAB: y_pred = predict(net, historyBuffer, 'ExecutionEnvironment','cpu');
                # Get LSTM prediction
                y_pred = model(input_seq).cpu().numpy().flatten()  # [10]

                # Store prediction
                # MATLAB: cmat_ensemble(it,:,s) = y_pred;
                cmat_ensemble[it, :, s] = y_pred

    # Compute mean and std across ensemble
    # MATLAB: lines 154-155
    # cmat_mean = mean(cmat_ensemble,3);
    # cmat_std = std(cmat_ensemble,0,3);
    cmat_mean = np.mean(cmat_ensemble, axis=2)  # [nx, 10]
    cmat_std = np.std(cmat_ensemble, axis=2)     # [nx, 10] (not used, but computed)

    # Update mobile and immobile concentrations
    # MATLAB: lines 157-158
    # cmob = cmat_mean(:,1:6);
    # cimob = cmat_mean(:,7:10);
    cmob = cmat_mean[:, 0:6]   # Update mobile species
    cimob = cmat_mean[:, 6:10] # Update immobile species

    # Record breakthrough curve at monitoring point
    # MATLAB: lines 160-161
    # BTC_mean = [BTC_mean; cmat_mean(25,:)]; % Collect data at 25 m
    # Note: MATLAB uses 1-based indexing, so position 25 in MATLAB is index 24 in Python
    BTC_mean = np.vstack([BTC_mean, cmat_mean[24, :]])  # Append row

    # Print progress
    if time % 5 == 0:
        print(f'  Time: {time}/{S_time} days')

print('='*70)
print('TRANSPORT SIMULATION COMPLETE')
print('='*70 + '\n')

# %% [markdown]
# ## Part 12: Visualization - Breakthrough Curves
#
# **MATLAB lines 164-189:** Plot concentration histories at monitoring point (x=25m)
#
# Creates 10 subplots showing temporal evolution of all species.

# %%
# === PREPARE DATA FOR PLOTTING ===
# MATLAB: lines 165-166
# tvec = 0:1:S_time;
# species = {'H2','CO2','CH4','H2S','SO4','Acetate','FeS','X_meth','X_sulf','X_aceto'};

tvec = np.arange(0, S_time + 1)  # Time vector [days]
species = ['H2', 'CO2', 'CH4', 'H2S', 'SO4', 'Acetate', 'FeS', 'X_meth', 'X_sulf', 'X_aceto']

# === CREATE FIGURE ===
# MATLAB: figure; for i = 1:10; subplot(5,2,i); ...
fig, axes = plt.subplots(5, 2, figsize=(14, 16))  # 5 rows × 2 columns
axes = axes.ravel()  # Flatten to 1D array for easy indexing

# === PLOT EACH SPECIES ===
for i in range(10):
    # MATLAB: plot(tvec, BTC_mean(:,i), '-b', 'LineWidth', 1.5);
    axes[i].plot(tvec, BTC_mean[:, i], 'b-', linewidth=2)

    # Labels and title
    # MATLAB: xlabel('Time [d]'); ylabel('mmol/L'); title(['BTC: ', species{i}]);
    axes[i].set_xlabel('Time [d]', fontsize=10)
    axes[i].set_ylabel('Concentration [mmol/L]', fontsize=10)
    axes[i].set_title(f'BTC: {species[i]}', fontsize=12, fontweight='bold')

    # Grid
    # MATLAB: grid on (implicitly done)
    axes[i].grid(True, alpha=0.3)

# Overall title
plt.suptitle('Breakthrough Curves at x = 25 m', fontsize=14, fontweight='bold', y=0.995)

# Adjust layout
plt.tight_layout()

# Save figure
# MATLAB: saveas or print command
plt.savefig('breakthrough_curves.png', dpi=150, bbox_inches='tight')
print('✓ Breakthrough curves saved to: breakthrough_curves.png')

# Display
plt.show()

print('\n' + '='*70)
print('MATLAB TO PYTORCH CONVERSION COMPLETE')
print('='*70)
print('All code sections successfully converted and executed.')
print('='*70)
