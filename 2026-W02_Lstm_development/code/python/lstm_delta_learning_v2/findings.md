# Findings: LSTM Delta Learning V2 - Interpolation Fix

## Context
This is V2 of the delta learning experiment, addressing the **staircase pattern** issue found in V1.

---

## V1 Baseline Results (Reference)

### RMSE Performance
| Variable | RMSE | RelErr |
|----------|------|--------|
| nH2_g | 0.0449 | 1.60% |
| nCO2_g | 0.0176 | 11.58% |
| nCH4_g | 0.0027 | 0.87% |
| nH2S_g | 0.0005 | 0.38% |
| SO4 | 0.0329 | 2.24% |
| X | 0.3120 | 0.70% |

### Problem Identified
- Output shows discrete "steps" every 30 timesteps
- Model predicts correctly at horizon boundaries
- No interpolation between predictions
- Sawtooth error pattern in time series

---

## V2 Modification: Linear Interpolation

### Concept
Instead of holding prediction constant for horizon steps:
```
y_interp(t + k) = y(t) + (k/H) * Δy_pred,  k = 1, 2, ..., H
```

Where:
- `y(t)` = current state
- `Δy_pred` = predicted delta
- `H` = horizon (30)
- `k` = step within horizon

### Expected Benefits
1. **Smooth output** - No more staircase
2. **Better visual match** - Curves follow ground truth
3. **Potentially better RMSE** - Intermediate values closer to truth

---

## V2 Results

### RMSE Comparison
| Variable | V1 RMSE | V2 RMSE | Change |
|----------|---------|---------|--------|
| nH2_g | 0.0449 | (pending) | - |
| nCO2_g | 0.0176 | (pending) | - |
| nCH4_g | 0.0027 | (pending) | - |
| nH2S_g | 0.0005 | (pending) | - |

### Visual Comparison
- (pending execution)

---

## Conclusions
- (pending execution)

---

## Related Memories
- `lstm_delta_learning_v1_results` - V1 baseline
- `lstm_delta_learning_experiment` - Original design
