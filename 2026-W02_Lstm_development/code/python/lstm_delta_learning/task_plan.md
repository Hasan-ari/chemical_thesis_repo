# Task Plan: LSTM Delta Learning Experiment

## User Intent
The user wants to implement "Delta Learning" (predicting change $\Delta y$ instead of absolute $y$) to solve the "drift" and "lag" issues observed in the previous LSTM model. This experiment also involves adding `pH` as an explicit input feature to help the model capture speciation-dependent kinetics.

## 1. Setup & Planning
- [ ] Create experiment directory: `2026-W02_Lstm_development/code/python/lstm_delta_learning/`
- [ ] Initialize `findings.md` and `progress.md` in the experiment directory.
- [ ] Copy necessary resources (MATLAB data files) or define paths to existing ones.

## 2. Implementation (`lstm_delta_forecast.py`)
- [ ] **Data Loading:** Load Basalt 25°C data (same as before).
- [ ] **Feature Engineering (New):**
    - [ ] Explicitly add `pH` as the 15th feature (interpolate from experimental data if needed, or use the `pH_fun` logic).
- [ ] **Target Engineering (New):**
    - [ ] Calculate `dy = y(t+horizon) - y(t)` as the training target.
    - [ ] Create a specific `DeltaScaler` to normalize these small delta values properly.
- [ ] **Model Architecture:**
    - [ ] Keep Stacked LSTM (128 -> 64).
    - [ ] Input size = 15 (14 states + pH).
    - [ ] Output size = 14 (predicted deltas).
- [ ] **Recursive Loop (Modified):**
    - [ ] Instead of `current = pred`, use `current = current + pred_delta`.

## 3. Execution & Validation
- [ ] Train the model (using 30-step horizon as established).
- [ ] Run Recursive Forecast (150 steps).
- [ ] Compare "Delta Model" vs "Absolute Model" (previous results).

## 4. Reporting
- [ ] Generate `delta_comparison.png`.
- [ ] Update `findings.md` with RMSE comparison.
- [ ] Update `progress.md`.
