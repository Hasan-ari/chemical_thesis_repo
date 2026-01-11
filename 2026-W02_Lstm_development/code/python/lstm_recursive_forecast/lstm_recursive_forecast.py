"""
============================================================================
LSTM RECURSIVE FORECAST VALIDATION - Week 02 (v2 - Multi-Step Horizon)
============================================================================

1. Chain correction ile recursive forecast
2. Multi-step horizon: 10, 20, 30 step atlayarak git
3. SEQ_LEN = 100 
4. FORECAST = 150 step
5. Fixed step size
6. Random başlangıç noktası
7. Divergence analizi: 50, 100, 150 step'lerde

Author: Chemical Thesis Project
Date: 2026-W02
Framework: PyTorch (v2.x)
============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
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
    N_POINTS = 2500  # Fixed step size
    
    # Train/Test Split (STRICT 80/20)
    TRAIN_SIZE = 2000
    TEST_SIZE = 500
    
    # LSTM Architecture
    SEQ_LEN = 100          # Hocam: 100 adım sequence length
    HIDDEN_1 = 128
    HIDDEN_2 = 64
    N_FEATURES = 14
    
    # ========== HOCAMIN İSTEDİĞİ: Multi-Step Horizon ==========
    # "10 step gidelim, 20 step" - 1 step yerine N step atlayarak öğren
    PRED_HORIZON = 10      # Kaç step sonrasını tahmin etsin
    
    # Test edilecek horizon değerleri (3 senaryo)
    HORIZONS_TO_TEST = [10, 20, 30]
    # ==========================================================
    
    # Training
    EPOCHS = 500
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-4
    
    # Recursive Forecast
    FORECAST_STEPS = 150   # Hocam: 100-150 tahmin yapsın
    
    # Preprocessing
    LOG_COLS = [3, 7, 9, 12, 13]  # nH2S_g, FeS, Acetate, Lag, Fe_pool
    
    # Paths
    BASE_DIR = None

# ============================================================================
# ODE MODEL (v4 Two-Phase)
# ============================================================================
def model_mixed(t, y, p, env):
    """
    Two-phase anaerobic model with 14 state variables.
    Exact replication of MATLAB v4 code.
    """
    # Unpack environment
    Vg, Vl, T, Rgas = env['Vg'], env['Vl'], env['T'], env['Rgas']
    Hcp_H2 = env['Hcp_H2_eff']
    Hcp_CO2 = env['Hcp_CO2_eff']
    Hcp_H2S = env['Hcp_H2S_eff']
    pKa = env['pKa_H2S']
    pH = env['pH_fun'](t)

    # Guard against negatives
    y = np.maximum(y, 1e-12)
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

    # Partial pressures
    pH2 = (nH2_g / 1000) * Rgas * T / Vg
    pCO2 = (nCO2_g / 1000) * Rgas * T / Vg
    pH2S = (nH2S_g / 1000) * Rgas * T / Vg

    # Henry equilibria
    Ceq_H2 = Hcp_H2 * pH2
    Ceq_CO2 = Hcp_CO2 * pCO2
    Ceq_H2S = Hcp_H2S * pH2S

    # Gas-liquid transfers
    J_H2 = kla_H2 * (Ceq_H2 - H2_aq)
    J_CO2 = kla_CO2 * (Ceq_CO2 - CO2_aq)

    # Sulfide speciation
    frac_HS = 1 / (1 + 10**(pKa - pH))
    HS_aq = S_tot * frac_HS
    H2S_aq = S_tot * (1 - frac_HS)
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

    # Thermodynamic gates
    RkJ = 8.314e-3
    RT = RkJ * T
    DG0_m, DG0_s, DG0_a = -130, -152, -95
    
    Q_a = Ac / (H2_aq**4 * CO2_aq**2)
    fT_s = 1 / (1 + np.exp((DG0_s + RT * np.log(1) - DG_th) / RT))
    fT_m = 1 / (1 + np.exp((DG0_m - DG_th) / RT))
    fT_a = 1 / (1 + np.exp((DG0_a + RT * np.log(Q_a) - DG_th) / RT))

    # Sulfate competition
    f_comp_m = 1 / (1 + beta_SO4_m * SO4)

    # Reaction rates
    r_meth = k_m * X * mH2 * mCO2 * f_inh_m * f_act * fT_m * f_comp_m
    r_sulf = k_s * X * mH2 * mSO4 * f_inh_s * f_act * fT_s
    r_aceto = k_a * X * mH2 * (mCO2**2) * f_inh_a * f_act * fT_a

    # Precipitation
    r_prec = min(k_prec * max(0, HS_aq - HS_sat), Fe_pool)
    r_diss_gyp = k_diss_gyp * max(0, env['SO4_sat_gyp'] - SO4)

    # Derivatives (14 ODEs)
    dy = np.zeros(14)
    dy[0] = -J_H2 * Vl                                    # nH2_g
    dy[1] = -J_CO2 * Vl                                   # nCO2_g
    dy[2] = r_meth * Vl                                   # nCH4_g
    dy[3] = Jout_H2S * Vl                                 # nH2S_g
    dy[4] = J_H2 - 4 * (r_meth + r_sulf + r_aceto)       # H2_aq
    dy[5] = J_CO2 - r_meth - 2 * r_aceto                 # CO2_aq
    dy[6] = -r_sulf + r_diss_gyp                         # SO4
    dy[7] = r_prec                                        # FeS
    dy[8] = Y_m * r_meth + Y_s * r_sulf + Y_a * r_aceto - b * X  # X
    dy[9] = r_aceto                                       # Ac
    dy[10] = 0.0                                          # HCO3
    dy[11] = r_sulf - r_prec - Jout_H2S                  # S_tot
    dy[12] = (f_lag - Lag) / max(w_lag, 1e-3)            # Lag
    dy[13] = -r_prec                                      # Fe_pool

    return dy

# ============================================================================
# DATA LOADING
# ============================================================================
def load_matlab_resources(base_dir):
    """
    Load parameters and experimental data from MATLAB files.
    """
    import scipy.io as sio
    
    mat_file = os.path.join(base_dir, 'best_fit_params_Basalt_25C.mat')
    txt_file = os.path.join(base_dir, 'Muller_2024_H2_Basalt_at_25C.txt')
    
    print(f"[1/7] Loading resources from {base_dir}...")
    
    # Load .mat file
    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"Missing: {mat_file}")
    
    mat = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    p_fit = mat['p_fit']
    env_struct = mat['env']
    
    # Load experimental data for pH interpolation
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Missing: {txt_file}")
    
    import pandas as pd
    df = pd.read_csv(txt_file, sep=r'\s+', comment='%', header=None, encoding='latin1')
    raw_data = df.values
    
    t_exp = raw_data[:, 0]
    pH_exp = raw_data[:, 5]
    
    # Create pH interpolator
    pH_fun = interp1d(t_exp, pH_exp, kind='linear', fill_value='extrapolate')
    
    # Build environment dictionary
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
    
    # Calculate initial state (y0) - exact MATLAB replication
    nH2_g_0 = raw_data[0, 1] / 1000.0   # umol -> mmol
    nCO2_g_0 = raw_data[0, 2] / 1000.0
    nCH4_g_0 = raw_data[0, 3] / 1000.0
    nH2S_g_0 = raw_data[0, 4] / 1000.0
    SO4_0 = raw_data[0, 6]
    
    # Henry's Law for initial aqueous concentrations
    pH2 = (nH2_g_0 / 1000.0) * env['Rgas'] * env['T'] / env['Vg']
    pCO2 = (nCO2_g_0 / 1000.0) * env['Rgas'] * env['T'] / env['Vg']
    
    H2_aq0 = env['Hcp_H2_eff'] * pH2
    CO2_aq0 = env['Hcp_CO2_eff'] * pCO2
    
    # Initial state vector
    y0 = np.array([
        nH2_g_0, nCO2_g_0, nCH4_g_0, nH2S_g_0,
        H2_aq0, CO2_aq0, SO4_0,
        0.01,   # FeS
        0.01,   # X (biomass)
        0.0,    # Acetate
        0.0,    # HCO3
        1.0,    # S_tot (initial sulfide seed)
        0.0,    # Lag
        0.10    # Fe_pool
    ])
    
    print("   [OK] Parameters and y0 loaded successfully.")
    return p_fit, y0, env

# ============================================================================
# DATA GENERATION (FIXED STEP SIZE)
# ============================================================================
def generate_ode_data(p_fit, y0, env, config):
    """
    Generate ODE solution with FIXED step size using linspace.
    This is CRITICAL for proper sequence learning.
    """
    print(f"[2/7] Generating ODE data (FIXED step: {config.N_POINTS} points)...")
    
    # CRITICAL: Use linspace for uniform time steps
    t_eval = np.linspace(config.T_START, config.T_END, config.N_POINTS)
    dt = t_eval[1] - t_eval[0]
    print(f"   Time step dt = {dt:.6f} days")
    
    # Solve ODE
    sol = solve_ivp(
        lambda t, y: model_mixed(t, y, p_fit, env),
        [config.T_START, config.T_END],
        y0,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-8,
        atol=1e-10
    )
    
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    
    data = sol.y.T  # Shape: (N_POINTS, 14)
    print(f"   [OK] ODE solved. Data shape: {data.shape}")
    
    return t_eval, data

# ============================================================================
# PREPROCESSING
# ============================================================================
def preprocess_data(data, config, fit_scaler=True, scaler=None):
    """
    Apply log1p transform and Z-score normalization.
    """
    data_processed = data.copy()
    
    # Log transform for small-valued columns
    data_processed[:, config.LOG_COLS] = np.log1p(data[:, config.LOG_COLS])
    
    # Z-score normalization
    if fit_scaler:
        scaler = StandardScaler()
        data_norm = scaler.fit_transform(data_processed)
    else:
        data_norm = scaler.transform(data_processed)
    
    return data_norm, scaler

def inverse_preprocess(data_norm, scaler, config):
    """
    Inverse transform: Z-score -> original scale -> expm1 for log columns.
    """
    data_processed = scaler.inverse_transform(data_norm)
    data_processed[:, config.LOG_COLS] = np.expm1(data_processed[:, config.LOG_COLS])
    
    # Ensure non-negative
    data_processed = np.maximum(data_processed, 0)
    
    return data_processed

# ============================================================================
# SEQUENCE CREATION (Multi-Step Horizon)
# ============================================================================
def create_sequences(data, seq_len, horizon=1):
    """
    Create sequences for LSTM training with multi-step horizon.
    
    Hocamın isteği: "10 step gidelim, 20 step"
    - horizon=1:  t+1 tahmin et (eski yöntem - identity mapping riski)
    - horizon=10: t+10 tahmin et (daha büyük değişim - öğrenmesi kolay)
    - horizon=20: t+20 tahmin et
    
    X: (N, SEQ_LEN, 14) - Son SEQ_LEN adım
    Y: (N, 14) - horizon adım sonrası
    """
    X, Y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i + seq_len])
        Y.append(data[i + seq_len + horizon - 1])  # horizon adım sonrası
    
    return np.array(X), np.array(Y)

# ============================================================================
# PYTORCH DATASET
# ============================================================================
class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series sequences.
    """
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ============================================================================
# LSTM MODEL (PyTorch)
# ============================================================================
class StackedLSTM(nn.Module):
    """
    Stacked LSTM model for time series forecasting.
    Architecture: LSTM(128) -> LSTM(64) -> Linear(14)
    """
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(StackedLSTM, self).__init__()
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True
        )
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden1,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True
        )
        
        # Output layer (no activation for regression)
        self.fc = nn.Linear(hidden2, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        # First LSTM: returns all hidden states
        out1, _ = self.lstm1(x)  # (batch, seq_len, hidden1)
        
        # Second LSTM: returns all hidden states
        out2, _ = self.lstm2(out1)  # (batch, seq_len, hidden2)
        
        # Take only the last time step
        out = self.fc(out2[:, -1, :])  # (batch, output_size)
        
        return out

def build_lstm_model(config, device):
    """
    Build Stacked LSTM model with PyTorch.
    """
    model = StackedLSTM(
        input_size=config.N_FEATURES,
        hidden1=config.HIDDEN_1,
        hidden2=config.HIDDEN_2,
        output_size=config.N_FEATURES
    )
    model = model.to(device)
    
    return model

# ============================================================================
# TRAINING (PyTorch)
# ============================================================================
def train_model(model, X_train, Y_train, config, device):
    """
    Train the LSTM model using PyTorch.
    Includes learning rate scheduling and early stopping logic.
    """
    print(f"[5/7] Training LSTM (SEQ_LEN={config.SEQ_LEN}, EPOCHS={config.EPOCHS})...")
    print(f"   Training samples: {len(X_train)}")
    
    # Create DataLoader
    dataset = TimeSeriesDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )
    
    # Learning rate scheduler (reduce on plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=30,
        min_lr=1e-7
    )
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 100
    best_state = None
    
    history = {'loss': [], 'mae': []}
    
    model.train()
    for epoch in range(config.EPOCHS):
        epoch_loss = 0.0
        epoch_mae = 0.0
        n_batches = 0
        
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
            optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_mae += torch.mean(torch.abs(predictions - Y_batch)).item()
            n_batches += 1
        
        # Average loss for epoch
        avg_loss = epoch_loss / n_batches
        avg_mae = epoch_mae / n_batches
        history['loss'].append(avg_loss)
        history['mae'].append(avg_mae)
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch+1}/{config.EPOCHS} - Loss: {avg_loss:.6f}, MAE: {avg_mae:.6f}, LR: {current_lr:.2e}")
        
        # Early stopping
        if patience_counter >= patience_limit:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print("   [OK] Training complete.")
    return history

# ============================================================================
# RECURSIVE FORECAST (Multi-Step Chain) - PyTorch
# ============================================================================
def recursive_forecast(model, initial_context, n_steps, horizon, config, device):
    """
    Perform recursive (free-running) forecast with multi-step horizon.
    
    Hocamın isteği: "Chain correction yaparak gidelim, 3 seferde gibi"
    
    Çalışma mantığı:
    - Model her seferinde 'horizon' adım sonrasını tahmin eder
    - Tahmin context'e eklenir
    - Tekrar tahmin yapılır
    - Bu şekilde chain devam eder
    
    Örnek (horizon=10, n_steps=150):
    - Adım 1: [t-99...t] -> t+10 tahmin
    - Adım 2: [t-89...t+10] -> t+20 tahmin
    - ...
    - 15 chain adımda 150 step tamamlanır
    
    Args:
        model: Trained PyTorch LSTM model
        initial_context: Array of shape (SEQ_LEN, 14) - normalized
        n_steps: Total number of steps to forecast
        horizon: Steps to predict at each chain iteration
        config: Configuration object
        device: torch device
    
    Returns:
        predictions: Array of shape (n_steps, 14) - normalized
    """
    model.eval()
    predictions = []
    context = initial_context.copy()
    
    # Kaç chain adımı gerekli?
    n_chains = (n_steps + horizon - 1) // horizon
    
    with torch.no_grad():
        for chain_idx in range(n_chains):
            # Convert to tensor: (1, seq_len, features)
            context_tensor = torch.FloatTensor(context).unsqueeze(0).to(device)
            
            # Predict horizon steps ahead
            pred = model(context_tensor)
            pred = pred.cpu().numpy()[0]  # Back to numpy, remove batch dim
            
            # Store prediction
            predictions.append(pred)
            
            # Update context: slide window by 1 and add prediction
            # Not: Context'i horizon kadar kaydırmak yerine 1 kaydırıyoruz
            # çünkü model horizon sonrasını tahmin ediyor
            context = np.vstack([context[1:], pred])
    
    # Tam n_steps döndür
    predictions = np.array(predictions)
    
    # Eğer chain adımları n_steps'ten fazlaysa, kes
    if len(predictions) > n_steps:
        predictions = predictions[:n_steps]
    
    return predictions


def recursive_forecast_dense(model, initial_context, n_steps, horizon, config, device):
    """
    Dense recursive forecast - her adımı tahmin et.
    
    Hocamın isteği: "Uzadıkça kopma olacak mı" kontrolü için
    
    Bu fonksiyon:
    - Model horizon adım sonrasını öğrenmiş
    - Ama biz her adımı görmek istiyoruz
    - Lineer interpolasyon ile ara değerleri doldurabiliriz
    - VEYA: model'i her adımda çalıştırıp context'i 1'er kaydırırız
    
    Args:
        Same as recursive_forecast
    
    Returns:
        predictions: Array of shape (n_steps, 14) - tüm adımlar
    """
    model.eval()
    predictions = []
    context = initial_context.copy()
    
    with torch.no_grad():
        for step in range(n_steps):
            # Convert to tensor
            context_tensor = torch.FloatTensor(context).unsqueeze(0).to(device)
            
            # Predict (model horizon adım sonrasını tahmin ediyor)
            pred = model(context_tensor)
            pred = pred.cpu().numpy()[0]
            
            # Store prediction
            predictions.append(pred)
            
            # Context'i 1 adım kaydır ve tahmini ekle
            context = np.vstack([context[1:], pred])
    
    return np.array(predictions)

# ============================================================================
# DIVERGENCE ANALYSIS
# ============================================================================
def analyze_divergence(predictions, ground_truth, config):
    """
    Calculate error at steps 50, 100, 150.
    Determine if error grows exponentially (divergence) or stays bounded (stable).
    """
    checkpoints = [50, 100, 150]
    state_names = ['nH2_g', 'nCO2_g', 'nCH4_g', 'nH2S_g', 'H2_aq', 'CO2_aq',
                   'SO4', 'FeS', 'X', 'Acetate', 'HCO3', 'S_tot', 'Lag', 'Fe_pool']
    
    # Key variables to report
    key_vars = [0, 3, 6]  # nH2_g, nH2S_g, SO4
    
    report = []
    report.append("=" * 70)
    report.append("DIVERGENCE ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")
    
    errors_at_checkpoints = {}
    
    for cp in checkpoints:
        if cp > len(predictions):
            continue
        
        errors = np.abs(predictions[cp-1] - ground_truth[cp-1])
        errors_at_checkpoints[cp] = errors
        
        report.append(f"--- Step {cp} ---")
        for idx in key_vars:
            report.append(f"  {state_names[idx]}: Error = {errors[idx]:.6f}")
        report.append("")
    
    # Divergence check: Is error at 150 > 10x error at 50?
    report.append("--- STABILITY CHECK ---")
    if 50 in errors_at_checkpoints and 150 in errors_at_checkpoints:
        for idx in key_vars:
            ratio = errors_at_checkpoints[150][idx] / (errors_at_checkpoints[50][idx] + 1e-10)
            status = "DIVERGING" if ratio > 10 else "STABLE"
            report.append(f"  {state_names[idx]}: Error ratio (150/50) = {ratio:.2f} -> {status}")
    
    report.append("")
    report.append("=" * 70)
    
    return "\n".join(report), errors_at_checkpoints

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_chain_test(predictions, ground_truth, start_idx, output_path, config):
    """
    Generate chain_test.png showing recursive prediction vs ground truth.
    """
    state_names = ['nH2_g', 'nCO2_g', 'nCH4_g', 'nH2S_g', 'H2_aq', 'CO2_aq',
                   'SO4', 'FeS', 'X', 'Acetate', 'HCO3', 'S_tot', 'Lag', 'Fe_pool']
    
    # Key variables to plot
    key_vars = [0, 3, 6]  # nH2_g, nH2S_g, SO4
    key_names = [state_names[i] for i in key_vars]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    steps = np.arange(1, len(predictions) + 1)
    
    for ax, var_idx, var_name in zip(axes, key_vars, key_names):
        ax.plot(steps, ground_truth[:, var_idx], 'b-', linewidth=2, label='Ground Truth (ODE)')
        ax.plot(steps, predictions[:, var_idx], 'r--', linewidth=2, label='Recursive Prediction (LSTM)')
        
        # Mark checkpoints
        for cp in [50, 100, 150]:
            if cp <= len(steps):
                ax.axvline(x=cp, color='gray', linestyle=':', alpha=0.5)
                ax.text(cp, ax.get_ylim()[1], f'Step {cp}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Forecast Step')
        ax.set_ylabel(f'{var_name} (mmol or mmol/L)')
        ax.set_title(f'{var_name} - Recursive Forecast')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'LSTM Recursive Forecast Test (Start Index: {start_idx})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"   [OK] Saved: {output_path}")

# ============================================================================
# MAIN EXECUTION (Multi-Horizon Test)
# ============================================================================
def main():
    """
    Main execution pipeline - Hocamın istekleri:
    1. Multi-step horizon testi (10, 20, 30)
    2. Chain correction ile recursive forecast
    3. Divergence analizi
    """
    print("=" * 70)
    print("LSTM RECURSIVE FORECAST - Multi-Step Horizon Test")
    print("Hocamın isteği: 10 step, 20 step, 30 step karşılaştırması")
    print("=" * 70)
    print("")
    
    # Configuration
    config = Config()
    
    # Detect environment and set paths
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        config.BASE_DIR = '/content/drive/MyDrive/chemical_thesis_repo/2026-W01_model_anlama/code/matlab'
        output_dir = '/content/drive/MyDrive/chemical_thesis_repo/2026-W02_Lstm_development/results'
        print("[ENV] Running on Google Colab")
    except ImportError:
        config.BASE_DIR = r'd:\chemical_thesis_repo\2026-W01_model_anlama\code\matlab'
        output_dir = r'd:\chemical_thesis_repo\2026-W02_Lstm_development\results'
        print("[ENV] Running locally")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    
    # Step 1: Load resources
    p_fit, y0, env = load_matlab_resources(config.BASE_DIR)
    
    # Step 2: Generate ODE data with FIXED step size
    t_eval, data_raw = generate_ode_data(p_fit, y0, env, config)
    dt = t_eval[1] - t_eval[0]
    
    # Step 3: STRICT Train/Test Split
    print(f"[3/7] Splitting data (STRICT 80/20)...")
    train_data = data_raw[:config.TRAIN_SIZE]
    print(f"   Train: {len(train_data)} points")
    print(f"   dt = {dt:.6f} days per step")
    
    # Step 4: Preprocess
    print(f"[4/7] Preprocessing (log1p + Z-score)...")
    train_norm, scaler = preprocess_data(train_data, config, fit_scaler=True)
    full_norm, _ = preprocess_data(data_raw, config, fit_scaler=False, scaler=scaler)
    
    # Fixed random seed for reproducibility
    np.random.seed(42)
    
    # Fixed start point for fair comparison
    min_start = config.TRAIN_SIZE
    max_start = config.N_POINTS - config.FORECAST_STEPS
    start_idx = np.random.randint(min_start, max_start)
    print(f"   Test start index: {start_idx} (fixed for all horizons)")
    
    # Get context and ground truth (same for all tests)
    context_start = start_idx - config.SEQ_LEN
    initial_context = full_norm[context_start:start_idx]
    ground_truth_norm = full_norm[start_idx:start_idx + config.FORECAST_STEPS]
    ground_truth_orig = inverse_preprocess(ground_truth_norm, scaler, config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[GPU] Device: {device}")
    
    # ========== HOCAMIN İSTEĞİ: 3 Farklı Horizon Testi ==========
    horizons = config.HORIZONS_TO_TEST  # [10, 20, 30]
    all_results = {}
    
    print("")
    print("=" * 70)
    print(f"TESTING {len(horizons)} HORIZONS: {horizons}")
    print("=" * 70)
    
    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"HORIZON = {horizon} steps ({horizon * dt:.4f} days)")
        print(f"{'='*60}")
        
        # Create sequences with this horizon
        X_train, Y_train = create_sequences(train_norm, config.SEQ_LEN, horizon)
        print(f"   Training sequences: {X_train.shape}")
        
        # Build fresh model for each horizon
        model = build_lstm_model(config, device)
        
        # Train
        print(f"[5/7] Training LSTM for horizon={horizon}...")
        history = train_model(model, X_train, Y_train, config, device)
        
        # Recursive forecast
        print(f"[6/7] Running recursive forecast...")
        predictions_norm = recursive_forecast_dense(
            model, initial_context.copy(), 
            config.FORECAST_STEPS, horizon, config, device
        )
        
        # Inverse transform
        predictions_orig = inverse_preprocess(predictions_norm, scaler, config)
        
        # Store results
        all_results[horizon] = {
            'predictions': predictions_orig,
            'model': model,
            'history': history
        }
        
        # Quick divergence check
        rmse = np.sqrt(np.mean((predictions_orig - ground_truth_orig)**2))
        print(f"   Overall RMSE: {rmse:.6f}")
    
    # ========== Karşılaştırmalı Analiz ==========
    print("\n" + "=" * 70)
    print("KARŞILAŞTIRMALI ANALİZ")
    print("=" * 70)
    
    # State names
    state_names = ['nH2_g', 'nCO2_g', 'nCH4_g', 'nH2S_g', 'H2_aq', 'CO2_aq',
                   'SO4', 'FeS', 'X', 'Acetate', 'HCO3', 'S_tot', 'Lag', 'Fe_pool']
    key_vars = [0, 3, 6]  # nH2_g, nH2S_g, SO4
    
    # Error comparison table
    print(f"\n{'Variable':<12}", end="")
    for h in horizons:
        print(f"H={h:<8}", end="")
    print("Best")
    print("-" * 50)
    
    best_horizons = {}
    for var_idx in range(14):
        print(f"{state_names[var_idx]:<12}", end="")
        errors = []
        for h in horizons:
            pred = all_results[h]['predictions'][:, var_idx]
            true = ground_truth_orig[:, var_idx]
            rmse = np.sqrt(np.mean((pred - true)**2))
            errors.append(rmse)
            print(f"{rmse:<10.4f}", end="")
        
        best_h = horizons[np.argmin(errors)]
        best_horizons[state_names[var_idx]] = best_h
        print(f"H={best_h}")
    
    # ========== Plot Comparison ==========
    print(f"\n[7/7] Generating comparison plots...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    steps = np.arange(config.FORECAST_STEPS)
    colors = ['red', 'green', 'purple']
    
    for ax, var_idx, var_name in zip(axes, key_vars, [state_names[i] for i in key_vars]):
        # Ground truth
        ax.plot(steps, ground_truth_orig[:, var_idx], 'b-', linewidth=2.5, 
                label='Ground Truth (ODE)')
        
        # Each horizon
        for h, color in zip(horizons, colors):
            pred = all_results[h]['predictions'][:, var_idx]
            ax.plot(steps, pred, '--', color=color, linewidth=1.5, 
                    label=f'Horizon={h}')
        
        # Checkpoints
        for cp in [50, 100, 150]:
            if cp <= len(steps):
                ax.axvline(x=cp, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_title(f'{var_name} - Multi-Horizon Comparison', fontsize=12)
        ax.set_xlabel('Forecast Step')
        ax.set_ylabel(var_name)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'LSTM Recursive Forecast - Horizon Comparison (Start: {start_idx})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'figures', 'horizon_comparison.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"   Saved: {plot_path}")
    
    # ========== Detailed Report ==========
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("MULTI-HORIZON COMPARISON REPORT")
    report_lines.append(f"Horizons tested: {horizons}")
    report_lines.append(f"Start index: {start_idx}")
    report_lines.append(f"Forecast steps: {config.FORECAST_STEPS}")
    report_lines.append(f"dt: {dt:.6f} days/step")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Divergence at checkpoints for each horizon
    for h in horizons:
        report_lines.append(f"\n--- HORIZON = {h} ---")
        pred = all_results[h]['predictions']
        
        for cp in [50, 100, 150]:
            if cp <= len(pred):
                report_lines.append(f"\nStep {cp}:")
                for var_idx in key_vars:
                    err = abs(pred[cp-1, var_idx] - ground_truth_orig[cp-1, var_idx])
                    report_lines.append(f"  {state_names[var_idx]}: Error = {err:.6f}")
    
    report_lines.append("\n" + "=" * 70)
    report_lines.append("BEST HORIZON FOR EACH VARIABLE:")
    report_lines.append("=" * 70)
    for var, best_h in best_horizons.items():
        report_lines.append(f"  {var}: Horizon = {best_h}")
    
    report = "\n".join(report_lines)
    print(report)
    
    report_path = os.path.join(output_dir, 'data', 'horizon_comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"   Saved: {report_path}")
    
    # Save best model
    best_overall = max(set(best_horizons.values()), key=list(best_horizons.values()).count)
    print(f"\n   Best overall horizon: {best_overall}")
    
    model_path = os.path.join(output_dir, 'data', f'lstm_horizon_{best_overall}.pt')
    torch.save({
        'model_state_dict': all_results[best_overall]['model'].state_dict(),
        'config': {
            'input_size': config.N_FEATURES,
            'hidden1': config.HIDDEN_1,
            'hidden2': config.HIDDEN_2,
            'output_size': config.N_FEATURES,
            'seq_len': config.SEQ_LEN,
            'horizon': best_overall
        }
    }, model_path)
    print(f"   Saved: {model_path}")
    
    # Save scaler
    import pickle
    scaler_path = os.path.join(output_dir, 'data', 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print("")
    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"Outputs in: {output_dir}")
    print("  - figures/horizon_comparison.png")
    print("  - data/horizon_comparison_report.txt")
    print(f"  - data/lstm_horizon_{best_overall}.pt")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()
