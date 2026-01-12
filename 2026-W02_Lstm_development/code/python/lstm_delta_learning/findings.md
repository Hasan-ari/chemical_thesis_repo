# Findings: LSTM Delta Learning

## Experiment Context
- **Goal:** Reduce drift and improve overlap in stiff reaction phases (exponential growth).
- **Hypothesis:** Training on $\Delta y$ (rate of change) will force the LSTM to learn the derivative/slope, preventing it from defaulting to "identity mapping" (lazy learning) during slow phases.
- **Dataset:** Basalt 25°C (ODE output).
- **Architecture:** Stacked LSTM (128->64), Horizon=30.

## Results Log

### 1. Feature Engineering Check
- [ ] **pH Feature:** Does adding pH improve the "Sulfide Speciation" prediction?
    - *Observation:* (Pending)

### 2. Delta vs Absolute Comparison
- [ ] **H2 Gas Drift:**
    - Absolute Model Error @ 150: 0.231 (Baseline)
    - Delta Model Error @ 150: (Pending)
- [ ] **Lag Phase Handling:**
    - Does the Delta model "wake up" faster when the exponential phase hits?
    - *Observation:* (Pending)

## Conclusions
- (Pending execution)
