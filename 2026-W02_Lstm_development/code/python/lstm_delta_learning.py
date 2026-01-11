"""
============================================================================
LSTM RECURSIVE FORECAST with DELTA LEARNING (DIFFERENCING)
============================================================================
Updates from previous version:
1. DELTA LEARNING: Model predicts (y_t+1 - y_t) instead of y_t+1.
   - Forces model to learn dynamics/derivatives.
   - Solves "Identity Mapping" (flat line) problem.
2. DELTA SCALING: Separate scaler for deltas to handle small dt changes.
3. STRIDE: Added optional stride to learn non-microscopic changes.

Author: Chemical Thesis Project
Date: 2026-W02 (Delta Fix)
Framework: PyTorch (v2.x)
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
    SEQ_LEN = 50           # Reduced slightly to focus on recent dynamics
    HIDDEN_1 = 128
    HIDDEN_2 = 64
    N_FEATURES = 14
    
    # Delta Learning Config
    PRED_HORIZON = 1       # Predict 1 step ahead (t+1)
    
    # Training
    EPOCHS = 300           # Delta learning converges faster usually
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3   # Slightly higher LR for deltas
    
    # Forecast
    FORECAST_STEPS = 150
    
    # Preprocessing
    LOG_COLS = [3, 7, 9, 12, 13]
    
    # Paths
    BASE_DIR = None

# ============================================================================
# ODE MODEL & DATA LOADING (Same as before)
# ============================================================================
def model_mixed(t, y, p, env):
    # Unpack environment
    Vg, Vl, T, Rgas = env['Vg'], env['Vl'], env['T'], env['Rgas']
    Hcp_H2, Hcp_CO2, Hcp_H2S = env['Hcp_H2_eff'], env['Hcp_CO2_eff'], env['Hcp_H2S_eff']
    pKa, pH = env['pKa_H2S'], env['pH_fun'](t)

    y = np.maximum(y, 1e-12)
    Fe_pool = max(y[13], 0)

    nH2_g, nCO2_g, nCH4_g, nH2S_g = y[0], y[1], y[2], y[3]
    H2_aq, CO2_aq, SO4, FeS = y[4], y[5], y[6], y[7]
    X, Ac, HCO3, S_tot, Lag = y[8], y[9], y[10], y[11], y[12]

    k_m, k_s, k_a = p[0], p[1], p[2]
    Y_m, Y_s, Y_a = p[3], p[4], p[5]
    KI_m, KI_s, KI_a = p[6], p[7], p[8]
    k_prec, HS_sat = p[9], p[10]
    H2_th, DG_th = p[11], p[12]
    K_H2, K_SO4, K_CO2 = p[13], p[14], p[15]
    kla_H2, kla_CO2, kla_H2S = p[16], p[17], p[18]
    b, t_lag, w_lag = p[19], p[20], p[21]
    k_diss_gyp, beta_SO4_m = p[22], p[23]

    pH2 = (nH2_g / 1000) * Rgas * T / Vg
    pCO2 = (nCO2_g / 1000) * Rgas * T / Vg
    pH2S = (nH2S_g / 1000) * Rgas * T / Vg

    Ceq_H2 = Hcp_H2 * pH2
    Ceq_CO2 = Hcp_CO2 * pCO2
    Ceq_H2S = Hcp_H2S * pH2S

    J_H2 = kla_H2 * (Ceq_H2 - H2_aq)
    J_CO2 = kla_CO2 * (Ceq_CO2 - CO2_aq)

    frac_HS = 1 / (1 + 10**(pKa - pH))
    HS_aq = S_tot * frac_HS
    H2S_aq = S_tot * (1 - frac_HS)
    Jout_H2S = kla_H2S * (H2S_aq - Ceq_H2S)

    f_inh_m = KI_m / (KI_m + HS_aq)
    f_inh_s = KI_s / (KI_s + HS_aq)
    f_inh_a = KI_a / (KI_a + HS_aq)
    f_H2 = H2_aq / (H2_aq + H2_th)
    f_lag = 1 / (1 + np.exp((t_lag - t) / max(w_lag, 1e-3)))
    f_act = f_H2 * f_lag

    mH2 = H2_aq / (K_H2 + H2_aq)
    mSO4 = SO4 / (K_SO4 + SO4)
    mCO2 = CO2_aq / (K_CO2 + CO2_aq)

    RkJ = 8.314e-3
    RT = RkJ * T
    DG0_m, DG0_s, DG0_a = -130, -152, -95
    
    Q_a = Ac / (H2_aq**4 * CO2_aq**2)
    fT_s = 1 / (1 + np.exp((DG0_s + RT * np.log(1) - DG_th) / RT))
    fT_m = 1 / (1 + np.exp((DG0_m - DG_th) / RT))
    fT_a = 1 / (1 + np.exp((DG0_a + RT * np.log(Q_a) - DG_th) / RT))

    f_comp_m = 1 / (1 + beta_SO4_m * SO4)

    r_meth = k_m * X * mH2 * mCO2 * f_inh_m * f_act * fT_m * f_comp_m
    r_sulf = k_s * X * mH2 * mSO4 * f_inh_s * f_act * fT_s
    r_aceto = k_a * X * mH2 * (mCO2**2) * f_inh_a * f_act * fT_a

    r_prec = min(k_prec * max(0, HS_aq - HS_sat), Fe_pool)
    r_diss_gyp = k_diss_gyp * max(0, env['SO4_sat_gyp'] - SO4)

    dy = np.zeros(14)
    dy[0] = -J_H2 * Vl
    dy[1] = -J_CO2 * Vl
    dy[2] = r_meth * Vl
    dy[3] = Jout_H2S * Vl
    dy[4] = J_H2 - 4 * (r_meth + r_sulf + r_aceto)
    dy[5] = J_CO2 - r_meth - 2 * r_aceto
    dy[6] = -r_sulf + r_diss_gyp
    dy[7] = r_prec
    dy[8] = Y_m * r_meth + Y_s * r_sulf + Y_a * r_aceto - b * X
    dy[9] = r_aceto
    dy[10] = 0.0
    dy[11] = r_sulf - r_prec - Jout_H2S
    dy[12] = (f_lag - Lag) / max(w_lag, 1e-3)
    dy[13] = -r_prec

    return dy

def load_matlab_resources(base_dir):
    import scipy.io as sio
    import pandas as pd
    
    mat_file = os.path.join(base_dir, 'best_fit_params_Basalt_25C.mat')
    txt_file = os.path.join(base_dir, 'Muller_2024_H2_Basalt_at_25C.txt')
    
    if not os.path.exists(mat_file): raise FileNotFoundError(f"Missing: {mat_file}")
    if not os.path.exists(txt_file): raise FileNotFoundError(f"Missing: {txt_file}")
    
    mat = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    p_fit = mat['p_fit']
    env_struct = mat['env']
    
    df = pd.read_csv(txt_file, sep=r'\s+', comment='%', header=None, encoding='latin1')
    raw_data = df.values
    
    t_exp = raw_data[:, 0]
    pH_exp = raw_data[:, 5]
    pH_fun = interp1d(t_exp, pH_exp, kind='linear', fill_value='extrapolate')
    
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
    
    nH2_g_0 = raw_data[0, 1] / 1000.0
    nCO2_g_0 = raw_data[0, 2] / 1000.0
    nCH4_g_0 = raw_data[0, 3] / 1000.0
    nH2S_g_0 = raw_data[0, 4] / 1000.0
    SO4_0 = raw_data[0, 6]
    
    pH2 = (nH2_g_0 / 1000.0) * env['Rgas'] * env['T'] / env['Vg']
    pCO2 = (nCO2_g_0 / 1000.0) * env['Rgas'] * env['T'] / env['Vg']
    
    y0 = np.array([
        nH2_g_0, nCO2_g_0, nCH4_g_0, nH2S_g_0,
        env['Hcp_H2_eff'] * pH2, env['Hcp_CO2_eff'] * pCO2, SO4_0,
        0.01, 0.01, 0.0, 0.0, 1.0, 0.0, 0.10
    ])
    return p_fit, y0, env

def generate_ode_data(p_fit, y0, env, config):
    t_eval = np.linspace(config.T_START, config.T_END, config.N_POINTS)
    sol = solve_ivp(
        lambda t, y: model_mixed(t, y, p_fit, env),
        [config.T_START, config.T_END], y0, t_eval=t_eval, method='BDF',
        rtol=1e-8, atol=1e-10
    )
    return t_eval, sol.y.T

# ============================================================================
# PREPROCESSING
# ============================================================================
def preprocess_data(data, config, fit_scaler=True, scaler=None):
    data_processed = data.copy()
    data_processed[:, config.LOG_COLS] = np.log1p(data[:, config.LOG_COLS])
    
    if fit_scaler:
        scaler = StandardScaler()
        data_norm = scaler.fit_transform(data_processed)
    else:
        data_norm = scaler.transform(data_processed)
    
    return data_norm, scaler

def inverse_preprocess(data_norm, scaler, config):
    data_processed = scaler.inverse_transform(data_norm)
    data_processed[:, config.LOG_COLS] = np.expm1(data_processed[:, config.LOG_COLS])
    data_processed = np.maximum(data_processed, 0)
    return data_processed

# ============================================================================
# SEQUENCE CREATION WITH DELTA
# ============================================================================
def create_delta_sequences(data, seq_len, horizon, fit_delta_scaler=True, delta_scaler=None):
    """
    Creates sequences X and targets Y (Deltas).
    X[t] = [x_{t-seq}, ..., x_t]
    Y[t] = x_{t+horizon} - x_t  (The change over horizon)
    """
    X, Y_deltas = [], []
    
    # Calculate raw deltas first
    raw_deltas = []
    
    valid_len = len(data) - seq_len - horizon
    for i in range(valid_len):
        # Current window
        curr_window = data[i : i + seq_len]
        
        # Current state (last element of window)
        curr_state = curr_window[-1]
        
        # Future state
        future_state = data[i + seq_len + horizon - 1] # e.g. next step
        
        # Delta
        delta = future_state - curr_state
        
        X.append(curr_window)
        raw_deltas.append(delta)
    
    X = np.array(X)
    raw_deltas = np.array(raw_deltas)
    
    # Scale Deltas
    # Since deltas are small, scaling them to N(0,1) is crucial for Gradient Descent
    if fit_delta_scaler:
        delta_scaler = StandardScaler()
        Y_scaled = delta_scaler.fit_transform(raw_deltas)
    else:
        Y_scaled = delta_scaler.transform(raw_deltas)
        
    return X, Y_scaled, delta_scaler

# ============================================================================
# PYTORCH HELPERS
# ============================================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(StackedLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, 1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, 1, batch_first=True)
        self.fc = nn.Linear(hidden2, output_size)
    
    def forward(self, x):
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        out = self.fc(out2[:, -1, :])
        return out

# ============================================================================
# TRAINING
# ============================================================================
def train_model(model, X_train, Y_train, config, device):
    dataset = TimeSeriesDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20)
    
    best_loss = float('inf')
    model.train()
    
    for epoch in range(config.EPOCHS):
        epoch_loss = 0
        for X_b, Y_b in dataloader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, Y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save best state if needed
        
        if (epoch+1) % 50 == 0:
            print(f"   Epoch {epoch+1}/{config.EPOCHS} - Delta Loss: {avg_loss:.6f}")
            
    return model

# ============================================================================
# RECURSIVE FORECAST (DELTA INTEGRATION)
# ============================================================================
def recursive_forecast_delta(model, initial_context, n_steps, delta_scaler, config, device):
    """
    Performs forecast by summing predicted deltas.
    y_{t+1} = y_t + InverseScale(Predicted_Delta)
    """
    model.eval()
    predictions = []
    
    # We work in the "feature scaled" space (Z-scores of raw variables)
    context = initial_context.copy()
    
    # Last known state (normalized)
    current_state = context[-1].copy()
    
    with torch.no_grad():
        for _ in range(n_steps):
            # Prepare input
            inp = torch.FloatTensor(context).unsqueeze(0).to(device)
            
            # Predict Scaled Delta
            delta_scaled_pred = model(inp).cpu().numpy()[0]
            
            # Inverse Scale Delta to get "Real Normalized Delta"
            # (The change in Z-score units)
            delta_pred = delta_scaler.inverse_transform(delta_scaled_pred.reshape(1, -1))[0]
            
            # Update State
            next_state = current_state + delta_pred
            
            # Store
            predictions.append(next_state)
            
            # Update Context (sliding window)
            context = np.vstack([context[1:], next_state])
            current_state = next_state
            
    return np.array(predictions)

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*60)
    print("LSTM DELTA LEARNING (DIFFERENCING) - Week 02 Fix")
    print("="*60)
    
    config = Config()
    
    # Environment Setup
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        config.BASE_DIR = '/content/drive/MyDrive/chemical_thesis_repo/2026-W01_model_anlama/code/matlab'
        out_dir = '/content/drive/MyDrive/chemical_thesis_repo/2026-W02_Lstm_development/results'
    except:
        config.BASE_DIR = r'd:\chemical_thesis_repo\2026-W01_model_anlama\code\matlab'
        out_dir = r'd:\chemical_thesis_repo\2026-W02_Lstm_development\results'
    
    os.makedirs(os.path.join(out_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'data'), exist_ok=True)
    
    # 1. Load & Generate Data
    p_fit, y0, env = load_matlab_resources(config.BASE_DIR)
    _, data_raw = generate_ode_data(p_fit, y0, env, config)
    
    # 2. Split & Feature Scaling
    train_data = data_raw[:config.TRAIN_SIZE]
    test_data = data_raw[config.TRAIN_SIZE:]
    
    train_norm, feat_scaler = preprocess_data(train_data, config, fit_scaler=True)
    # Note: We create 'test_norm' just for initial context extraction
    full_norm, _ = preprocess_data(data_raw, config, fit_scaler=False, scaler=feat_scaler)
    
    # 3. Create Sequences (Delta Learning)
    print(f"[Info] creating delta sequences (Horizon={config.PRED_HORIZON})...")
    X_train, Y_train_deltas, delta_scaler = create_delta_sequences(
        train_norm, config.SEQ_LEN, config.PRED_HORIZON, fit_delta_scaler=True
    )
    print(f"   X shape: {X_train.shape}")
    print(f"   Y (Delta) shape: {Y_train_deltas.shape}")
    
    # 4. Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StackedLSTM(config.N_FEATURES, config.HIDDEN_1, config.HIDDEN_2, config.N_FEATURES).to(device)
    model = train_model(model, X_train, Y_train_deltas, config, device)
    
    # 5. Recursive Forecast Check
    start_idx = 2102 # Same index as before for fair comparison
    
    # Context must be normalized features
    initial_context = full_norm[start_idx - config.SEQ_LEN : start_idx]
    
    # Ground Truth (Original Scale)
    ground_truth_orig = data_raw[start_idx : start_idx + config.FORECAST_STEPS]
    
    # Run Delta Forecast
    print("[Info] Running Delta Recursive Forecast...")
    preds_norm = recursive_forecast_delta(model, initial_context, config.FORECAST_STEPS, delta_scaler, config, device)
    
    # Inverse Feature Scaling
    preds_orig = inverse_preprocess(preds_norm, feat_scaler, config)
    
    # 6. Plotting
    key_vars = [0, 3, 6] # nH2_g, nH2S_g, SO4
    names = ['nH2_g', 'nH2S_g', 'SO4']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    t_axis = np.arange(config.FORECAST_STEPS)
    
    for i, (ax, var_idx) in enumerate(zip(axes, key_vars)):
        # Ground Truth
        ax.plot(t_axis, ground_truth_orig[:, var_idx], 'b-', label='Ground Truth', linewidth=2)
        # Prediction
        ax.plot(t_axis, preds_orig[:, var_idx], 'r--', label='Delta LSTM', linewidth=2)
        
        ax.set_title(f"{names[i]} - Delta Learning")
        ax.grid(True, alpha=0.3)
        if i==0: ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figures', 'delta_test.png'))
    print("   Saved delta_test.png")
    
    # Save Scalers
    with open(os.path.join(out_dir, 'data', 'delta_scaler.pkl'), 'wb') as f:
        pickle.dump(delta_scaler, f)

if __name__ == "__main__":
    main()