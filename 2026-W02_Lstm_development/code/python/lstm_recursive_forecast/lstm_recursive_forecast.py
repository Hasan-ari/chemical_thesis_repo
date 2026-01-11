"""
============================================================================
LSTM RECURSIVE FORECAST VALIDATION - Week 02
============================================================================
This script:
1. Generates ODE data with FIXED step size (np.linspace)
2. Strict 80/20 train/test split (model never sees test data)
3. Trains Stacked LSTM with SEQ_LEN=100 using PyTorch
4. Validates using RECURSIVE forecast (no teacher forcing)
5. Generates chain_test.png and divergence report

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
    SEQ_LEN = 100          # Increased from 50
    HIDDEN_1 = 128
    HIDDEN_2 = 64
    N_FEATURES = 14
    
    # Training
    EPOCHS = 500
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-4
    
    # Recursive Forecast
    FORECAST_STEPS = 150
    
    # Preprocessing
    LOG_COLS = [3, 7, 9, 12, 13]  # nH2S_g, FeS, Acetate, Lag, Fe_pool
    
    # Paths (UPDATE THESE FOR YOUR ENVIRONMENT)
    # For Colab: '/content/drive/MyDrive/chemical_thesis_repo/...'
    # For Local: 'd:/chemical_thesis_repo/...'
    BASE_DIR = None  # Will be set based on environment

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
# SEQUENCE CREATION
# ============================================================================
def create_sequences(data, seq_len):
    """
    Create sequences for LSTM training.
    X: (N, SEQ_LEN, 14)
    Y: (N, 14)
    """
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        Y.append(data[i + seq_len])
    
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
# RECURSIVE FORECAST (THE CORE TEST) - PyTorch
# ============================================================================
def recursive_forecast(model, initial_context, n_steps, config, device):
    """
    Perform recursive (free-running) forecast using PyTorch.
    
    This is the CRITICAL test:
    - Model receives its own predictions as input
    - No teacher forcing (no ground truth during prediction)
    - Tests stability and error accumulation
    
    Args:
        model: Trained PyTorch LSTM model
        initial_context: Array of shape (SEQ_LEN, 14) - normalized (numpy)
        n_steps: Number of steps to forecast
        config: Configuration object
        device: torch device
    
    Returns:
        predictions: Array of shape (n_steps, 14) - normalized (numpy)
    """
    model.eval()
    predictions = []
    context = initial_context.copy()
    
    with torch.no_grad():
        for step in range(n_steps):
            # Convert to tensor: (1, seq_len, features)
            context_tensor = torch.FloatTensor(context).unsqueeze(0).to(device)
            
            # Predict next step
            pred = model(context_tensor)
            pred = pred.cpu().numpy()[0]  # Back to numpy, remove batch dim
            
            # Store prediction
            predictions.append(pred)
            
            # Update context: remove oldest, add newest prediction
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
# MAIN EXECUTION
# ============================================================================
def main():
    """
    Main execution pipeline.
    """
    print("=" * 70)
    print("LSTM RECURSIVE FORECAST VALIDATION - Week 02")
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
    
    # Step 3: STRICT Train/Test Split
    print(f"[3/7] Splitting data (STRICT 80/20)...")
    train_data = data_raw[:config.TRAIN_SIZE]
    test_data = data_raw[config.TRAIN_SIZE:]
    print(f"   Train: {len(train_data)} points (indices 0-{config.TRAIN_SIZE-1})")
    print(f"   Test:  {len(test_data)} points (indices {config.TRAIN_SIZE}-{config.N_POINTS-1})")
    
    # Step 4: Preprocess (fit scaler ONLY on training data!)
    print(f"[4/7] Preprocessing (log1p + Z-score)...")
    train_norm, scaler = preprocess_data(train_data, config, fit_scaler=True)
    test_norm, _ = preprocess_data(test_data, config, fit_scaler=False, scaler=scaler)
    
    # Also preprocess full data for ground truth comparison
    full_norm, _ = preprocess_data(data_raw, config, fit_scaler=False, scaler=scaler)
    
    # Create sequences (ONLY from training data)
    X_train, Y_train = create_sequences(train_norm, config.SEQ_LEN)
    print(f"   Training sequences: {X_train.shape}")
    
    # Step 5: Build and train model (PyTorch)
    # Set device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[GPU] PyTorch device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    
    model = build_lstm_model(config, device)
    print(model)
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    history = train_model(model, X_train, Y_train, config, device)
    
    # Step 6: Recursive Forecast Test
    print(f"[6/7] Running RECURSIVE FORECAST test...")
    
    # Pick a random starting point from TEST set
    np.random.seed(42)  # Reproducibility
    
    # The test set starts at index TRAIN_SIZE
    # We need SEQ_LEN steps BEFORE the starting point for context
    # So valid start indices are: TRAIN_SIZE to (N_POINTS - FORECAST_STEPS)
    min_start = config.TRAIN_SIZE
    max_start = config.N_POINTS - config.FORECAST_STEPS
    
    start_idx = np.random.randint(min_start, max_start)
    print(f"   Random start index: {start_idx} (from test set)")
    
    # Get context: SEQ_LEN steps before start_idx
    context_start = start_idx - config.SEQ_LEN
    initial_context = full_norm[context_start:start_idx]
    print(f"   Context: indices {context_start} to {start_idx-1}")
    
    # Get ground truth for comparison
    ground_truth_norm = full_norm[start_idx:start_idx + config.FORECAST_STEPS]
    
    # Run recursive forecast
    predictions_norm = recursive_forecast(model, initial_context, config.FORECAST_STEPS, config, device)
    
    # Inverse transform to original scale
    predictions_orig = inverse_preprocess(predictions_norm, scaler, config)
    ground_truth_orig = inverse_preprocess(ground_truth_norm, scaler, config)
    
    # Step 7: Analysis and Visualization
    print(f"[7/7] Generating outputs...")
    
    # Divergence report
    report, errors = analyze_divergence(predictions_orig, ground_truth_orig, config)
    print(report)
    
    # Save report
    report_path = os.path.join(output_dir, 'data', 'divergence_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"   [OK] Saved: {report_path}")
    
    # Plot
    plot_path = os.path.join(output_dir, 'figures', 'chain_test.png')
    plot_chain_test(predictions_orig, ground_truth_orig, start_idx, plot_path, config)
    
    # Save model (PyTorch format)
    model_path = os.path.join(output_dir, 'data', 'lstm_recursive_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_size': config.N_FEATURES,
            'hidden1': config.HIDDEN_1,
            'hidden2': config.HIDDEN_2,
            'output_size': config.N_FEATURES,
            'seq_len': config.SEQ_LEN
        }
    }, model_path)
    print(f"   [OK] Saved: {model_path}")
    
    # Save scaler for future use
    import pickle
    scaler_path = os.path.join(output_dir, 'data', 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   [OK] Saved: {scaler_path}")
    
    print("")
    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"Outputs in: {output_dir}")
    print("  - figures/chain_test.png")
    print("  - data/divergence_report.txt")
    print("  - data/lstm_recursive_model.pt")
    print("  - data/scaler.pkl")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()
