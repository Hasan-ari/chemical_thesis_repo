# Week 2 (2026-W02): LSTM Recursive Forecast Validation

**Date Range:** 06 Jan - 12 Jan 2026  
**Main Goal:** Validate LSTM stability in free-running (recursive) forecast mode

---

## Goals

- [ ] Fixed-step ODE data generation (2500 points, uniform spacing)
- [ ] Strict 80/20 temporal train/test split
- [ ] Stacked LSTM with SEQ_LEN=100
- [ ] Train for 500+ epochs
- [ ] Implement recursive forecast (chain prediction)
- [ ] Divergence analysis at steps 50, 100, 150
- [ ] Generate chain_test.png deliverable

---

## Week 01 Foundation

Last week we established:
- ODE model (v4 two-phase) works in both MATLAB and Python
- LSTM one-step prediction achieves good R2 scores
- Preprocessing: Z-score + log1p for small-valued variables
- Architecture: Stacked LSTM (128 -> 64) performs well

**Key files from Week 01:**
- `2026-W01_model_anlama/lstm_training_in_matlab/lstm_train_v4.m`
- `2026-W01_model_anlama/lstm_training_in_matlab/notebook/6_01_2026_yeni_kod_ile_lstm_egitimi.ipynb`

---

## This Week's Critical Test

### The Problem with One-Step Validation

One-step prediction (teacher forcing) hides a critical issue: the model always receives *perfect* ground truth as input. In real ODE surrogates, we need the model to:

1. Start from initial conditions
2. Predict the next state
3. Use that prediction as input for the next step
4. Continue indefinitely without ground truth

### Recursive Forecast Algorithm

```
1. Select random index t from TEST SET (steps 2001-2500)
2. Context = states[t-100 : t]  # 100 steps of history
3. For i = 1 to 150:
   - pred = model.predict(Context)
   - Append pred to trajectory
   - Context = Context[1:] + [pred]  # Slide window
4. Compare predicted trajectory with ground truth
5. Check: Does error grow exponentially or stay bounded?
```

---

## Technical Specifications

### Data Generation
- Time span: 0 to 20 days
- Points: 2500 (fixed step = 0.008 days)
- Method: `np.linspace(0, 20, 2500)`
- ODE solver: solve_ivp with BDF method

### Train/Test Split
- Train: Steps 0-1999 (first 2000 points)
- Test: Steps 2000-2499 (last 500 points)
- STRICT: Model never sees test data during training

### LSTM Architecture
```
Input (14 features, 100 timesteps)
    |
LSTM(128, return_sequences=True)
    |
LSTM(64, return_sequences=False)
    |
Dense(14)
    |
Output (14 features, 1 timestep)
```

### Training
- Optimizer: Adam (lr=5e-4)
- Loss: MSE
- Epochs: 500+
- Batch size: 32
- Gradient clipping: 0.5

---

## Deliverables

### 1. chain_test.png
- 3 subplots: nH2_g, nH2S_g, SO4
- Blue solid line: Ground truth
- Red dashed line: Recursive prediction
- X-axis: 150 steps
- Title: Starting index clearly labeled

### 2. Divergence Report
```
| Step | nH2_g Error | nH2S_g Error | SO4 Error | Status |
|------|-------------|--------------|-----------|--------|
| 50   | [value]     | [value]      | [value]   | Stable/Diverging |
| 100  | [value]     | [value]      | [value]   | Stable/Diverging |
| 150  | [value]     | [value]      | [value]   | Stable/Diverging |
```

---

## Success Criteria

- [ ] Error at step 150 is less than 10x error at step 50 (bounded growth)
- [ ] Predicted trajectory maintains correct trend direction
- [ ] No negative concentrations in predictions
- [ ] Model completes 150-step forecast without NaN/Inf

---

## Files Structure

```
2026-W02_Lstm_development/
+-- code/
|   +-- python/
|       +-- lstm_recursive_forecast.py
+-- results/
|   +-- figures/
|       +-- chain_test.png
|   +-- data/
|       +-- divergence_report.txt
+-- notes/
+-- README.md
+-- PROGRESS.md
+-- this_week_goal.md
```

---

## Questions

1. **Question:** Should we test multiple random starting points?
   - **Answer:** TBD (start with one, expand if stable)

2. **Question:** What if divergence is detected?
   - **Answer:** Try: longer SEQ_LEN, residual connections, or physics constraints

---

**Git Commit:** [pending]
