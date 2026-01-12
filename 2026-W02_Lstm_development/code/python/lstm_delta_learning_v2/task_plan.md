# Task Plan: LSTM Delta Learning V2 - Interpolation Fix

## Problem Statement
V1 delta learning showed good RMSE (1.60% for H2) but produced **staircase output** due to:
- Predicting delta once per horizon (30 steps)
- Repeating same state for all intermediate steps
- No interpolation between prediction points

## Goal
Fix the staircase pattern while maintaining or improving prediction accuracy.

---

## Approach: Linear Interpolation

### Strategy
Instead of repeating the same predicted state for all horizon steps, **linearly interpolate** between current state and predicted next state.

### Before (V1):
```python
for _ in range(horizon):
    predictions.append(next_state_norm)  # Same value 30 times!
```

### After (V2):
```python
for step in range(horizon):
    alpha = (step + 1) / horizon  # 0.033, 0.067, ..., 1.0
    interp_state = current_state + alpha * pred_delta
    predictions.append(interp_state)
```

---

## Implementation Plan

### Phase 1: Setup ✅
- [x] Create V2 folder structure
- [x] Initialize planning files (task_plan.md, findings.md, progress.md)

### Phase 2: Code Modification
- [ ] Copy base code from V1
- [ ] Modify `recursive_forecast_delta()` function
- [ ] Add interpolation logic
- [ ] Update visualization to show improvement

### Phase 3: Validation
- [ ] Run on same test region (day 16.4 - 17.6)
- [ ] Compare RMSE with V1
- [ ] Generate comparison plots (V1 vs V2)

### Phase 4: Documentation
- [ ] Update findings.md with results
- [ ] Update progress.md
- [ ] Save to Serena memory

---

## Expected Outcome

| Metric | V1 | V2 (Expected) |
|--------|-----|---------------|
| H2 RMSE | 0.0449 | Similar or better |
| Output smoothness | Staircase | Smooth |
| Visual quality | Poor | Good |

---

## Files

| File | Purpose |
|------|---------|
| `lstm_delta_forecast_v2.py` | Main module with interpolation |
| `run_in_colab.ipynb` | Colab notebook |
| `findings.md` | Results and analysis |
| `progress.md` | Session log |
| `task_plan.md` | This file |

---

## Related Memory
- `lstm_delta_learning_v1_results` - V1 baseline results
- `lstm_delta_learning_experiment` - Original experiment design
