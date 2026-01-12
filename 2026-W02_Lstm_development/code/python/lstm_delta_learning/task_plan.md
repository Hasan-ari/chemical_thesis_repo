# Task Plan: LSTM Delta Learning Experiment

## User Intent
The user wants to implement "Delta Learning" (predicting change $\Delta y$ instead of absolute $y$) to solve the "drift" and "lag" issues observed in the previous LSTM model. This experiment also involves adding `pH` as an explicit input feature to help the model capture speciation-dependent kinetics.

---

## 1. Setup & Planning ✅ COMPLETED
- [x] Create experiment directory: `2026-W02_Lstm_development/code/python/lstm_delta_learning/`
- [x] Initialize `findings.md` and `progress.md` in the experiment directory.
- [x] Copy necessary resources (MATLAB data files) or define paths to existing ones.
- [x] Analyze MATLAB v4 model structure (14 states, 28 parameters)
- [x] Document data source decision (uniform grid vs .dat file)

---

## 2. Implementation ✅ COMPLETED

### `lstm_delta_forecast.py`
- [x] **Data Loading:** Load Basalt 25°C data via ODE solver with uniform grid
- [x] **Feature Engineering:**
    - [x] Add `pH` as the 15th feature (interpolated from experimental data)
    - [x] Log-transform small/zero-prone columns (nH2S_g, FeS, Acetate, Lag, Fe_pool)
- [x] **Target Engineering:**
    - [x] Calculate `dy = y(t+horizon) - y(t)` as the training target
    - [x] Create `StandardScaler` for delta normalization
- [x] **Model Architecture:**
    - [x] Stacked LSTM (128 -> 64) with dropout
    - [x] Input size = 15 (14 states + pH)
    - [x] Output size = 14 (predicted deltas)
- [x] **Recursive Loop:**
    - [x] Integration: `current = current + pred_delta`

### `run_in_colab.ipynb`
- [x] Environment setup (Drive mount, GPU detection)
- [x] Full ODE model (Python port of MATLAB v4)
- [x] Training with early stopping and LR scheduling
- [x] Recursive delta forecasting
- [x] Visualization and model saving

---

## 3. Execution & Validation 🔄 IN PROGRESS
- [ ] Upload data files to Colab (`.mat` and `.txt`)
- [ ] Train the model on GPU (300 epochs max, early stopping)
- [ ] Run Recursive Forecast (150 steps)
- [ ] Compare "Delta Model" vs "Absolute Model" (previous results)

---

## 4. Reporting 📋 PENDING
- [ ] Generate comparison plots
- [ ] Update `findings.md` with RMSE comparison
- [ ] Update `progress.md` with final results
- [ ] Save trained model and scalers

---

## Key Design Decisions

### 1. Uniform Grid (Critical for LSTM)
**Problem:** MATLAB .dat file has adaptive timesteps (non-uniform)
**Solution:** Re-generate ODE solution with `t_eval=np.linspace(0, 20, 2500)`
**Rationale:** LSTM hidden state assumes uniform time intervals

### 2. Delta Learning
**Problem:** Standard LSTM learns "identity mapping" during slow phases
**Solution:** Predict Δy instead of y, then integrate recursively
**Expected Benefit:** Model forced to learn actual dynamics/derivatives

### 3. pH as Input Feature
**Problem:** Sulfide speciation depends on pH (H2S ↔ HS⁻)
**Solution:** Include pH as 15th input feature
**Note:** pH treated as known/external during forecasting

---

## Files

| File | Status | Description |
|------|--------|-------------|
| `lstm_delta_forecast.py` | ✅ | Main Python module |
| `run_in_colab.ipynb` | ✅ | Colab notebook |
| `findings.md` | ✅ | MATLAB analysis + design decisions |
| `progress.md` | ✅ | Session log |
| `task_plan.md` | ✅ | This file |

---

## Related Memories (Serena)
- `matlab_v4_model_analysis` - Model structure
- `lstm_data_source_decision` - Uniform grid rationale
- `lstm_delta_learning_experiment` - Experiment overview
