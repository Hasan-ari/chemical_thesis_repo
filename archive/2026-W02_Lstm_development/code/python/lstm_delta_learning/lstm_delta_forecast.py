"""
============================================================================
LSTM DELTA LEARNING FORECAST - Week 03
============================================================================

1. Delta Learning: Predict (y_{t+h} - y_t) instead of y_{t+h}
2. Physics-Informed: Explicitly adds pH as an input feature (Input size = 15)
3. Horizon: 30 steps (Fixed based on W02 findings)
4. Recursive Integration: current_{t+h} = current_t + predicted_delta

Author: Chemical Thesis Project
Date: 2026-W03
============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Data Generation
    T_START = 0.0
    T_END = 20.0
    N_POINTS = 2500
    
    # Train/Test Split
    TRAIN_SIZE = 2000
    TEST_SIZE = 500
    
    # LSTM Architecture
    SEQ_LEN = 100
    HIDDEN_1 = 128
    HIDDEN_2 = 64
    N_STATES = 14       # Original state variables
    N_FEATURES = 15     # States + pH
    
    # Delta Learning Settings
    PRED_HORIZON = 30   # Fixed horizon (W02 winner)
    
    # Training
    EPOCHS = 500
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-4
    
    # Forecast
    FORECAST_STEPS = 150
    
    # Preprocessing
    LOG_COLS = [3, 7, 9, 12, 13]  # nH2S_g, FeS, Acetate, Lag, Fe_pool
    
    # Paths (Default for local Windows env, will be overwritten if Colab)
    BASE_DIR = r'd:\chemical_thesis_repo\2026-W02_Lstm_development\code\matlab\Basalt\25C'

# ============================================================================
# ODE MODEL (Reference for Data Generation)
# ============================================================================
def model_mixed(t, y, p, env):
    """Refactored ODE model for data generation."""
    Vg, Vl, T, Rgas = env['Vg'], env['Vl'], env['T'], env['Rgas']
    Hcp_H2, Hcp_CO2, Hcp_H2S = env['Hcp_H2_eff'], env['Hcp_CO2_eff'], env['Hcp_H2S_eff']
    pKa = env['pKa_H2S']
    pH = env['pH_fun'](t)  # pH is time-dependent

    y = np.maximum(y, 1e-12)
    Fe_pool = max(y[13], 0)

    nH2_g, nCO2_g, nCH4_g, nH2S_g = y[0], y[1], y[2], y[3]
    H2_aq, CO2_aq, SO4, FeS = y[4], y[5], y[6], y[7]
    X, Ac, HCO3, S_tot, Lag = y[8], y[9], y[10], y[11], y[12]

    # Unpack parameters (shortened for brevity, assumes standard p vector)
    k_m, k_s, k_a = p[0], p[1], p[2]
    Y_m, Y_s, Y_a = p[3], p[4], p[5]
    KI_m, KI_s, KI_a = p[6], p[7], p[8]
    k_prec, HS_sat = p[9], p[10]
    H2_th, DG_th = p[11], p[12]
    K_H2, K_SO4, K_CO2 = p[13], p[14], p[15]
    kla_H2, kla_CO2, kla_H2S = p[16], p[17], p[18]
    b, t_lag, w_lag = p[19], p[20], p[21]
    k_diss_gyp, beta_SO4_m = p[22], p[23]

    # Partial pressures
    pH2 = (nH2_g / 1000) * Rgas * T / Vg
    pCO2 = (nCO2_g / 1000) * Rgas * T / Vg
    pH2S = (nH2S_g / 1000) * Rgas * T / Vg

    # Gas-liquid transfers
    Ceq_H2, Ceq_CO2, Ceq_H2S = Hcp_H2*pH2, Hcp_CO2*pCO2, Hcp_H2S*pH2S
    J_H2 = kla_H2 * (Ceq_H2 - H2_aq)
    J_CO2 = kla_CO2 * (Ceq_CO2 - CO2_aq)

    # Speciation
    frac_HS = 1 / (1 + 10**(pKa - pH))
    HS_aq = S_tot * frac_HS
    H2S_aq = S_tot * (1 - frac_HS)
    Jout_H2S = kla_H2S * (H2S_aq - Ceq_H2S)

    # Rates
    f_inh_m, f_inh_s, f_inh_a = KI_m/(KI_m+HS_aq), KI_s/(KI_s+HS_aq), KI_a/(KI_a+HS_aq)
    f_H2 = H2_aq / (H2_aq + H2_th)
    f_lag = 1 / (1 + np.exp((t_lag - t) / max(w_lag, 1e-3)))
    f_act = f_H2 * f_lag
    
    mH2, mSO4, mCO2 = H2_aq/(K_H2+H2_aq), SO4/(K_SO4+SO4), CO2_aq/(K_CO2+CO2_aq)
    
    # Thermo gates
    RkJ, RT = 8.314e-3, 8.314e-3 * T
    DG0_m, DG0_s, DG0_a = -130, -152, -95
    Q_a = Ac / (H2_aq**4 * CO2_aq**2)
    
    fT_s = 1 / (1 + np.exp((-152 + RT * np.log(1) - DG_th) / RT))
    fT_m = 1 / (1 + np.exp((-130 - DG_th) / RT))
    fT_a = 1 / (1 + np.exp((-95 + RT * np.log(Q_a) - DG_th) / RT))
    f_comp_m = 1 / (1 + beta_SO4_m * SO4)

    r_meth = k_m * X * mH2 * mCO2 * f_inh_m * f_act * fT_m * f_comp_m
    r_sulf = k_s * X * mH2 * mSO4 * f_inh_s * f_act * fT_s
    r_aceto = k_a * X * mH2 * (mCO2**2) * f_inh_a * f_act * fT_a
    
    r_prec = min(k_prec * max(0, HS_aq - HS_sat), Fe_pool)
    r_diss_gyp = k_diss_gyp * max(0, env['SO4_sat_gyp'] - SO4)

    # Derivatives
    dy = np.zeros(14)
    dy[0] = -J_H2 * Vl
    dy[1] = -J_CO2 * Vl
    dy[2] = r_meth * Vl
    dy[3] = Jout_H2S * Vl
    dy[4] = J_H2 - 4*(r_meth + r_sulf + r_aceto)
    dy[5] = J_CO2 - r_meth - 2*r_aceto
    dy[6] = -r_sulf + r_diss_gyp
    dy[7] = r_prec
    dy[8] = Y_m*r_meth + Y_s*r_sulf + Y_a*r_aceto - b*X
    dy[9] = r_aceto
    dy[10] = 0.0
    dy[11] = r_sulf - r_prec - Jout_H2S
    dy[12] = (f_lag - Lag) / max(w_lag, 1e-3)
    dy[13] = -r_prec
    
    return dy

# ============================================================================
# DATA UTILS
# ============================================================================
def load_and_generate_data(config):
    """Load MATLAB params, generate ODE data, and append pH feature."""
    import scipy.io as sio
    import pandas as pd
    
    print("[1/6] Loading resources and generating ODE data...")
    
    # Load .mat parameters
    mat_file = os.path.join(config.BASE_DIR, 'best_fit_params_Basalt_25C.mat')
    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"Missing: {mat_file}")
        
    mat = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    p_fit = mat['p_fit']
    env_struct = mat['env']
    
    # Load experimental pH for interpolation
    txt_file = os.path.join(config.BASE_DIR, 'Muller_2024_H2_Basalt_at_25C.txt')
    df = pd.read_csv(txt_file, sep=r'\s+', comment='%', header=None, encoding='latin1')
    t_exp, pH_exp = df.values[:, 0], df.values[:, 5]
    pH_fun = interp1d(t_exp, pH_exp, kind='linear', fill_value='extrapolate')
    
    # Reconstruct env dict
    env = {
        'Vg': float(env_struct.Vg), 'Vl': float(env_struct.Vl),
        'T': float(env_struct.T), 'Rgas': float(env_struct.Rgas),
        'Hcp_H2_eff': float(env_struct.Hcp_H2_eff),
        'Hcp_CO2_eff': float(env_struct.Hcp_CO2_eff),
        'Hcp_H2S_eff': float(env_struct.Hcp_H2S_eff),
        'pKa_H2S': float(env_struct.pKa_H2S),
        'SO4_sat_gyp': float(env_struct.SO4_sat_gyp),
        'pH_fun': pH_fun
    }
    
    # Initial conditions (from file)
    nH2_g_0 = df.values[0, 1] / 1000.0
    y0 = np.array([
        nH2_g_0, df.values[0, 2]/1000, df.values[0, 3]/1000, df.values[0, 4]/1000,
        env['Hcp_H2_eff'] * (nH2_g_0/1000)*env['Rgas']*env['T']/env['Vg'], 
        env['Hcp_CO2_eff'] * (df.values[0, 2]/1e6)*env['Rgas']*env['T']/env['Vg'], # Approx
        df.values[0, 6], 0.01, 0.01, 0.0, 0.0, 1.0, 0.0, 0.10
    ])
    
    # Generate ODE Data
    t_eval = np.linspace(config.T_START, config.T_END, config.N_POINTS)
    sol = solve_ivp(lambda t, y: model_mixed(t, y, p_fit, env), 
                   [config.T_START, config.T_END], y0, t_eval=t_eval, method='BDF')
    
    if not sol.success: raise RuntimeError("ODE Solver failed")
    
    data_states = sol.y.T # (N, 14)
    
    # Augment with pH
    pH_vals = pH_fun(t_eval).reshape(-1, 1)
    data_full = np.hstack([data_states, pH_vals]) # (N, 15)
    
    print(f"   Data shape: {data_full.shape} (14 states + 1 pH)")
    return t_eval, data_full

def preprocess_delta(data, config):
    """
    1. Log transform select columns.
    2. Split Train/Test.
    3. Calculate Deltas (Targets).
    4. Normalize Features and Deltas separately.
    """
    print("[2/6] Preprocessing for Delta Learning...")
    
    data_proc = data.copy()
    # Log transform only state variables (0-13), not pH (14)
    log_indices = [c for c in config.LOG_COLS if c < config.N_STATES]
    data_proc[:, log_indices] = np.log1p(data_proc[:, log_indices])
    
    # Split
    train_data = data_proc[:config.TRAIN_SIZE]
    
    # Scaler for Features (Input)
    feat_scaler = StandardScaler()
    train_feat_norm = feat_scaler.fit_transform(train_data)
    all_feat_norm = feat_scaler.transform(data_proc)
    
    # Create Targets: Delta_y = y(t+h) - y(t)
    # We only predict deltas for the 14 STATES, not pH
    horizon = config.PRED_HORIZON
    
    X_list, Y_delta_list = [], []
    
    # Sliding window for training data
    # Input: [t-seq...t] (Features including pH)
    # Target: [State(t+h) - State(t)] (Only 14 states)
    
    for i in range(config.SEQ_LEN, config.TRAIN_SIZE - horizon):
        # Input sequence: (SEQ_LEN, 15)
        seq = train_feat_norm[i-config.SEQ_LEN : i]
        X_list.append(seq)
        
        # Calculate Delta using NORMALIZED values to keep scales consistent for loss
        # Note: We are predicting the jump in the NORMALIZED space
        current_state = train_feat_norm[i-1, :config.N_STATES]
        future_state  = train_feat_norm[i+horizon-1, :config.N_STATES]
        delta = future_state - current_state
        Y_delta_list.append(delta)
        
    X_train = np.array(X_list)
    Y_train = np.array(Y_delta_list)
    
    # Scaler for Deltas? 
    # Actually, if we use normalized states, deltas are already scaled roughly ~N(0, 1).
    # But a separate scaler can help fine-tune.
    delta_scaler = StandardScaler()
    Y_train_norm = delta_scaler.fit_transform(Y_train)
    
    print(f"   X_train: {X_train.shape}, Y_train (Delta): {Y_train_norm.shape}")
    
    return all_feat_norm, X_train, Y_train_norm, feat_scaler, delta_scaler

# ============================================================================
# MODEL & TRAINING
# ============================================================================
class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc = nn.Linear(hidden2, output_size)
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        return self.fc(out[:, -1, :])

def train_delta_model(model, X, Y, config, device):
    print(f"[3/6] Training Delta Model ({config.EPOCHS} epochs)...")
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(config.EPOCHS):
        epoch_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)
        
        if (epoch+1) % 50 == 0:
            print(f"   Epoch {epoch+1}: Loss = {avg_loss:.6f}")
            
    return model

# ============================================================================
# RECURSIVE DELTA INTEGRATOR
# ============================================================================
def recursive_forecast_delta(model, initial_context, n_steps, delta_scaler, config, device):
    """
    Integrates the delta predictions: y_{t+h} = y_t + predicted_delta
    """
    model.eval()
    
    # context: (SEQ_LEN, 15) - Normalized
    context = initial_context.copy()
    
    # We need to track the current "head" state (normalized 14 vars)
    current_state_norm = context[-1, :config.N_STATES].copy()
    
    predictions = [] # Will store normalized states
    
    horizon = config.PRED_HORIZON
    n_chains = (n_steps + horizon - 1) // horizon
    
    # Pre-calculate future pH values for the context update?
    # In a real forecast, we assume we know pH (controlled) or project it.
    # For validation, we can 'cheat' and use the known pH from the dataset (feature 15)
    # But strictly recursive means we should probably repeat the last pH or use a model for it.
    # Here: We will assume pH is an EXTERNAL CONTROL variable (known).
    
    with torch.no_grad():
        for i in range(n_chains):
            # Predict Delta
            ctx_tensor = torch.FloatTensor(context).unsqueeze(0).to(device)
            pred_delta_norm = model(ctx_tensor).cpu().numpy()[0]
            
            # Inverse scale delta to get "Normalized State Delta"
            # (Remember we trained on scaled deltas of normalized states)
            pred_delta_state = delta_scaler.inverse_transform(pred_delta_norm.reshape(1, -1))[0]
            
            # Integrate
            next_state_norm = current_state_norm + pred_delta_state
            
            # Store prediction (replicate for horizon steps for dense output)
            # Simplification: Linear interp could be better, but step holds for now
            for _ in range(horizon):
                predictions.append(next_state_norm)
                
            # Update Context
            # We need to append 'horizon' new rows. 
            # For the state part, we use the predicted 'next_state_norm'.
            # For the pH part (index 14), we simply repeat the last known pH (constant assumption for short term)
            # OR better: use the pH from the "Next State" if we had a pH model.
            # Fix: Repeat the LAST prediction as the new context input
            
            last_pH = context[-1, 14] 
            new_row = np.hstack([next_state_norm, last_pH])
            
            # Slide context
            context = np.vstack([context[horizon:], np.tile(new_row, (horizon, 1))])
            current_state_norm = next_state_norm
            
    return np.array(predictions[:n_steps])

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=== LSTM Delta Learning Experiment ===")
    config = Config()
    
    # 1. Load Data
    t_eval, data_full = load_and_generate_data(config)
    
    # 2. Preprocess
    all_feat_norm, X_train, Y_train, feat_scaler, delta_scaler = preprocess_delta(data_full, config)
    
    # 3. Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StackedLSTM(config.N_FEATURES, config.HIDDEN_1, config.HIDDEN_2, config.N_STATES).to(device)
    model = train_delta_model(model, X_train, Y_train, config, device)
    
    # 4. Forecast (Test Set)
    # Start from random point in test set
    np.random.seed(42)
    start_idx = 2102 # Fixed from previous exp
    
    initial_context = all_feat_norm[start_idx-config.SEQ_LEN : start_idx]
    
    print("[4/6] Running Recursive Delta Forecast...")
    preds_norm = recursive_forecast_delta(model, initial_context, config.FORECAST_STEPS, delta_scaler, config, device)
    
    # 5. Inverse Transform
    # Create dummy array for inverse transform (needs 15 cols, we have 14 predicted states)
    # We pad the 15th col (pH) with zeros as we don't evaluate pH error
    preds_padded = np.hstack([preds_norm, np.zeros((len(preds_norm), 1))])
    preds_orig = feat_scaler.inverse_transform(preds_padded)
    
    # Handle Log transform inverse
    preds_final = preds_orig.copy()
    log_indices = [c for c in config.LOG_COLS if c < config.N_STATES]
    preds_final[:, log_indices] = np.expm1(preds_final[:, log_indices])
    preds_final = np.maximum(preds_final, 0)
    
    # Ground Truth
    gt_orig = data_full[start_idx : start_idx+config.FORECAST_STEPS, :config.N_STATES]
    
    # 6. Evaluate
    mse_h2 = np.mean((preds_final[:, 0] - gt_orig[:, 0])**2)
    print(f"\nRESULTS:")
    print(f"MSE H2 (Gas): {mse_h2:.6f}")
    
    # Save dummy plot for verification
    plt.figure()
    plt.plot(gt_orig[:, 0], label='Truth')
    plt.plot(preds_final[:, 0], '--', label='Delta Prediction')
    plt.title("H2 Gas - Delta Learning Check")
    plt.legend()
    plt.savefig('delta_check.png')
    print("Saved delta_check.png")

if __name__ == "__main__":
    main()
