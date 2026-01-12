"""
============================================================================
LSTM DELTA LEARNING FORECAST V2 - Interpolation Fix
============================================================================

Changes from V1:
- Fixed staircase pattern by adding LINEAR INTERPOLATION
- Instead of repeating same state for horizon steps, interpolate smoothly

V1 Problem:
    for _ in range(horizon):
        predictions.append(next_state_norm)  # Same value repeated!

V2 Solution:
    for step in range(horizon):
        alpha = (step + 1) / horizon
        interp_state = current_state + alpha * pred_delta
        predictions.append(interp_state)

Author: Chemical Thesis Project
Date: 2026-W02 (V2)
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
    EPOCHS = 300
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-4
    PATIENCE = 30

    # Forecast
    FORECAST_STEPS = 150

    # Preprocessing
    LOG_COLS = [3, 7, 9, 12, 13]  # nH2S_g, FeS, Acetate, Lag, Fe_pool

    # Paths (Default for local Windows env, will be overwritten if Colab)
    BASE_DIR = r'd:\chemical_thesis_repo\2026-W02_Lstm_development\code\matlab\Basalt\25C'

# ============================================================================
# ODE MODEL (Same as V1)
# ============================================================================
def model_mixed(t, y, p, env):
    """Two-phase anaerobic model (v4) - Python implementation."""
    Vg, Vl, T, Rgas = env['Vg'], env['Vl'], env['T'], env['Rgas']
    Hcp_H2, Hcp_CO2, Hcp_H2S = env['Hcp_H2_eff'], env['Hcp_CO2_eff'], env['Hcp_H2S_eff']
    pKa = env['pKa_H2S']
    pH = env['pH_fun'](t)

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

    Ceq_H2, Ceq_CO2, Ceq_H2S = Hcp_H2*pH2, Hcp_CO2*pCO2, Hcp_H2S*pH2S
    J_H2 = kla_H2 * (Ceq_H2 - H2_aq)
    J_CO2 = kla_CO2 * (Ceq_CO2 - CO2_aq)

    frac_HS = 1 / (1 + 10**(pKa - pH))
    HS_aq = S_tot * frac_HS
    H2S_aq = S_tot * (1 - frac_HS)
    Jout_H2S = kla_H2S * (H2S_aq - Ceq_H2S)

    f_inh_m, f_inh_s, f_inh_a = KI_m/(KI_m+HS_aq), KI_s/(KI_s+HS_aq), KI_a/(KI_a+HS_aq)
    f_H2 = H2_aq / (H2_aq + H2_th)
    f_lag = 1 / (1 + np.exp((t_lag - t) / max(w_lag, 1e-3)))
    f_act = f_H2 * f_lag

    mH2, mSO4, mCO2 = H2_aq/(K_H2+H2_aq), SO4/(K_SO4+SO4), CO2_aq/(K_CO2+CO2_aq)

    RT = 8.314e-3 * T
    Q_a = max(Ac, 1e-12) / (max(H2_aq, 1e-12)**4 * max(CO2_aq, 1e-12)**2)

    fT_s = 1 / (1 + np.exp((-152 + RT * np.log(1) - DG_th) / RT))
    fT_m = 1 / (1 + np.exp((-130 - DG_th) / RT))
    fT_a = 1 / (1 + np.exp((-95 + RT * np.log(Q_a) - DG_th) / RT))
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
# DATA UTILS (Same as V1)
# ============================================================================
def load_and_generate_data(config):
    """Load MATLAB params, generate ODE data, and append pH feature."""
    import scipy.io as sio
    import pandas as pd

    print("[1/6] Loading resources and generating ODE data...")

    mat_file = os.path.join(config.BASE_DIR, 'best_fit_params_Basalt_25C.mat')
    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"Missing: {mat_file}")

    mat = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    p_fit = mat['p_fit']
    env_struct = mat['env']

    txt_file = os.path.join(config.BASE_DIR, 'Muller_2024_H2_Basalt_at_25C.txt')
    df = pd.read_csv(txt_file, sep=r'\s+', comment='%', header=None, encoding='latin1')
    t_exp, pH_exp = df.values[:, 0], df.values[:, 5]
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

    nH2_g_0 = df.values[0, 1] / 1000.0
    pH2_0 = (nH2_g_0 / 1000) * env['Rgas'] * env['T'] / env['Vg']
    pCO2_0 = (df.values[0, 2] / 1e6) * env['Rgas'] * env['T'] / env['Vg']

    y0 = np.array([
        nH2_g_0, df.values[0, 2]/1000, df.values[0, 3]/1000, df.values[0, 4]/1000,
        env['Hcp_H2_eff'] * pH2_0,
        env['Hcp_CO2_eff'] * pCO2_0,
        df.values[0, 6], 0.01, 0.01, 0.0, 0.0, 1.0, 0.0, 0.10
    ])

    t_eval = np.linspace(config.T_START, config.T_END, config.N_POINTS)
    sol = solve_ivp(lambda t, y: model_mixed(t, y, p_fit, env),
                   [config.T_START, config.T_END], y0, t_eval=t_eval, method='BDF')

    if not sol.success: raise RuntimeError("ODE Solver failed")

    data_states = sol.y.T
    pH_vals = pH_fun(t_eval).reshape(-1, 1)
    data_full = np.hstack([data_states, pH_vals])

    print(f"   Data shape: {data_full.shape} (14 states + 1 pH)")
    return t_eval, data_full, env, p_fit

def preprocess_delta(data, config):
    """Preprocess for delta learning."""
    print("[2/6] Preprocessing for Delta Learning...")

    data_proc = data.copy()
    log_indices = [c for c in config.LOG_COLS if c < config.N_STATES]
    data_proc[:, log_indices] = np.log1p(data_proc[:, log_indices])

    train_data = data_proc[:config.TRAIN_SIZE]

    feat_scaler = StandardScaler()
    train_feat_norm = feat_scaler.fit_transform(train_data)
    all_feat_norm = feat_scaler.transform(data_proc)

    horizon = config.PRED_HORIZON
    X_list, Y_delta_list = [], []

    for i in range(config.SEQ_LEN, config.TRAIN_SIZE - horizon):
        seq = train_feat_norm[i-config.SEQ_LEN : i]
        X_list.append(seq)

        current_state = train_feat_norm[i-1, :config.N_STATES]
        future_state  = train_feat_norm[i+horizon-1, :config.N_STATES]
        delta = future_state - current_state
        Y_delta_list.append(delta)

    X_train = np.array(X_list)
    Y_train = np.array(Y_delta_list)

    delta_scaler = StandardScaler()
    Y_train_norm = delta_scaler.fit_transform(Y_train)

    print(f"   X_train: {X_train.shape}, Y_train (Delta): {Y_train_norm.shape}")

    return all_feat_norm, X_train, Y_train_norm, feat_scaler, delta_scaler

# ============================================================================
# MODEL (Same as V1)
# ============================================================================
class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size, dropout=0.1):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden2, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        return self.fc(out[:, -1, :])

def train_delta_model(model, X, Y, config, device):
    print(f"[3/6] Training Delta Model ({config.EPOCHS} max epochs)...")
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = {'loss': []}

    model.train()
    for epoch in range(config.EPOCHS):
        epoch_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        history['loss'].append(avg_loss)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch+1) % 25 == 0:
            print(f"   Epoch {epoch+1}: Loss = {avg_loss:.6f}")

        if patience_counter >= config.PATIENCE:
            print(f"   Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    print(f"   Best loss: {best_loss:.6f}")
    return model, history

# ============================================================================
# RECURSIVE DELTA INTEGRATOR V2 - WITH INTERPOLATION
# ============================================================================
def recursive_forecast_delta_v2(model, initial_context, n_steps, all_feat_norm,
                                 start_idx, delta_scaler, config, device):
    """
    V2: Recursive forecasting with LINEAR INTERPOLATION.

    Instead of repeating same state for all horizon steps,
    interpolate smoothly between current and predicted next state.

    y_interp(k) = y_current + (k/H) * delta_pred,  k = 1, 2, ..., H
    """
    print("[4/6] Recursive Delta Forecast V2 (with interpolation)...")

    model.eval()
    context = initial_context.copy()
    current_state_norm = context[-1, :config.N_STATES].copy()

    predictions = []
    horizon = config.PRED_HORIZON
    n_chains = (n_steps + horizon - 1) // horizon

    print(f"   Forecasting {n_steps} steps ({n_chains} chains of {horizon})")
    print(f"   Using LINEAR INTERPOLATION (V2 fix)")

    with torch.no_grad():
        for chain in range(n_chains):
            # Predict delta
            ctx_tensor = torch.FloatTensor(context).unsqueeze(0).to(device)
            pred_delta_scaled = model(ctx_tensor).cpu().numpy()[0]

            # Inverse scale delta
            pred_delta = delta_scaler.inverse_transform(pred_delta_scaled.reshape(1, -1))[0]

            # ============================================================
            # V2 FIX: LINEAR INTERPOLATION instead of repeating
            # ============================================================
            for step in range(horizon):
                if len(predictions) >= n_steps:
                    break

                # Interpolation factor: 0.033, 0.067, ..., 1.0
                alpha = (step + 1) / horizon

                # Interpolated state
                interp_state = current_state_norm + alpha * pred_delta
                predictions.append(interp_state.copy())
            # ============================================================

            # Final state after this horizon
            next_state_norm = current_state_norm + pred_delta

            # Update context for next chain
            future_idx = start_idx + (chain + 1) * horizon
            if future_idx < len(all_feat_norm):
                future_pH = all_feat_norm[future_idx, 14]
            else:
                future_pH = context[-1, 14]

            new_row = np.hstack([next_state_norm, future_pH])
            context = np.vstack([context[horizon:], np.tile(new_row, (horizon, 1))])
            current_state_norm = next_state_norm

    print(f"   Generated {len(predictions)} predictions (smooth)")
    return np.array(predictions[:n_steps])

# ============================================================================
# INVERSE TRANSFORM
# ============================================================================
def inverse_transform_predictions(preds_norm, feat_scaler, config):
    """Inverse transform predictions back to original scale."""
    print("[5/6] Inverse Transform...")

    preds_padded = np.hstack([preds_norm, np.zeros((len(preds_norm), 1))])
    preds_orig = feat_scaler.inverse_transform(preds_padded)

    preds_final = preds_orig.copy()
    log_indices = [c for c in config.LOG_COLS if c < config.N_STATES]
    preds_final[:, log_indices] = np.expm1(preds_final[:, log_indices])
    preds_final = np.maximum(preds_final, 0)

    return preds_final[:, :config.N_STATES]

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("LSTM Delta Learning V2 - Interpolation Fix")
    print("=" * 60)

    config = Config()

    # 1. Load Data
    t_eval, data_full, env, p_fit = load_and_generate_data(config)

    # 2. Preprocess
    all_feat_norm, X_train, Y_train, feat_scaler, delta_scaler = preprocess_delta(data_full, config)

    # 3. Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")

    model = StackedLSTM(config.N_FEATURES, config.HIDDEN_1, config.HIDDEN_2, config.N_STATES).to(device)
    model, history = train_delta_model(model, X_train, Y_train, config, device)

    # 4. Forecast (Test Set) - V2 with interpolation
    start_idx = config.TRAIN_SIZE + 50
    initial_context = all_feat_norm[start_idx - config.SEQ_LEN : start_idx]

    preds_norm = recursive_forecast_delta_v2(
        model, initial_context, config.FORECAST_STEPS,
        all_feat_norm, start_idx, delta_scaler, config, device
    )

    # 5. Inverse Transform
    preds_final = inverse_transform_predictions(preds_norm, feat_scaler, config)

    # Ground Truth
    gt_orig = data_full[start_idx : start_idx + config.FORECAST_STEPS, :config.N_STATES]
    t_forecast = t_eval[start_idx : start_idx + config.FORECAST_STEPS]

    # 6. Evaluate
    print("\n" + "=" * 60)
    print("[6/6] Evaluation Results (V2)")
    print("=" * 60)

    state_names = ['nH2_g', 'nCO2_g', 'nCH4_g', 'nH2S_g', 'H2_aq', 'CO2_aq',
                   'SO4', 'FeS', 'X', 'Acetate', 'HCO3', 'S_tot', 'Lag', 'Fe_pool']

    print("\nRMSE per state variable:")
    print("-" * 50)
    results = []
    for i, name in enumerate(state_names):
        rmse = np.sqrt(np.mean((preds_final[:, i] - gt_orig[:, i])**2))
        rel_error = rmse / (np.mean(np.abs(gt_orig[:, i])) + 1e-10) * 100
        results.append((name, rmse, rel_error))
        print(f"  {name:12s}: RMSE = {rmse:.6f}, RelErr = {rel_error:.2f}%")

    # Save results
    print("\nSaving results...")

    # Save model and scalers
    torch.save(model.state_dict(), 'results/delta_lstm_model_v2.pt')
    with open('results/delta_scalers_v2.pkl', 'wb') as f:
        pickle.dump({'feat_scaler': feat_scaler, 'delta_scaler': delta_scaler}, f)

    # Save evaluation results
    with open('results/evaluation_results_v2.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("[6/6] Evaluation Results (V2 - With Interpolation)\n")
        f.write("=" * 60 + "\n\n")
        f.write("RMSE per state variable:\n")
        f.write("-" * 50 + "\n")
        for name, rmse, rel_err in results:
            f.write(f"  {name:12s}: RMSE = {rmse:.6f}, RelErr = {rel_err:.2f}%\n")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    plot_vars = [
        (0, 'nH2_g (mmol)', 'Gas H2'),
        (1, 'nCO2_g (mmol)', 'Gas CO2'),
        (2, 'nCH4_g (mmol)', 'Gas CH4'),
        (3, 'nH2S_g (mmol)', 'Gas H2S'),
        (6, 'SO4 (mmol/L)', 'Sulfate'),
        (8, 'X (mmol/L)', 'Biomass')
    ]

    for ax, (idx, ylabel, title) in zip(axes.flat, plot_vars):
        ax.plot(t_forecast, gt_orig[:, idx], 'b-', linewidth=2, label='Truth')
        ax.plot(t_forecast, preds_final[:, idx], 'r--', linewidth=2, label='Delta LSTM V2')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Delta Learning V2 Forecast (with Interpolation)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/Delta_learning_forecast_v2.png', dpi=150)
    print("Saved: results/Delta_learning_forecast_v2.png")

    # H2 detailed plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(t_forecast, gt_orig[:, 0], 'b-', linewidth=2, label='Ground Truth')
    ax1.plot(t_forecast, preds_final[:, 0], 'r--', linewidth=2, label='Delta LSTM V2')
    ax1.fill_between(t_forecast, gt_orig[:, 0], preds_final[:, 0], alpha=0.3, color='gray')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('nH2_g (mmol)')
    ax1.set_title('H2 Gas: Delta Learning V2 (with Interpolation)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    error = np.abs(preds_final[:, 0] - gt_orig[:, 0])
    ax2.plot(t_forecast, error, 'k-', linewidth=1.5)
    ax2.fill_between(t_forecast, 0, error, alpha=0.3)
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Absolute Error (mmol)')
    ax2.set_title('H2 Gas: Prediction Error Over Time')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/H2_gas_delta_learning_v2.png', dpi=150)
    print("Saved: results/H2_gas_delta_learning_v2.png")

    plt.show()
    print("\nDone!")

if __name__ == "__main__":
    main()
