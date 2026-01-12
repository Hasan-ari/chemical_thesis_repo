# Investigation: Why LSTM Fails to Overlap

## 1. Ground Truth Quality Check
**Files Compared:**
*   Experimental Data: `Muller_2024_H2_Basalt_at_25C.txt`
*   ODE Model Output: `Basalt_25C_inc_rates.dat`

**Finding:**
The "Ground Truth" for the LSTM (the ODE output) **does not match the Experimental Data perfectly**.
*   **Experimental Data:**
    *   Day 0.0: H2 = 9074, SO4 = 5.7
    *   Day 1.1: H2 = 8655, SO4 = 5.7
    *   Day 5.0: H2 = 8016, SO4 = 3.1
*   **ODE Model Output:**
    *   Day 0.0: H2 = 9.074 (Unit Mismatch: mmol vs µmol in experimental?)
    *   Day 5.0: The ODE curve is smooth, while experimental data has noise.

**Critical Note on Units:**
*   The `.txt` file says `9074` (likely µmol).
*   The `.dat` file says `9.074` (mmol).
*   **Result:** The physics are consistent (factor of 1000), so the ODE is a valid ground truth to train on. The mismatch is just unit scaling, which the LSTM's scaler handles.

## 2. Why the LSTM "Drifts" (The Real Issue)
The issue is **not** the quality of the ODE data. The ODE data is smooth and clean. The issue is **how the LSTM learns**.

### A. The "Stiffness" Problem
*   **Reaction Speed:**
    *   At the start (Day 0-5): Changes are slow (Lag phase).
    *   Middle (Day 5-9): changes are **extremely fast** (Exponential phase).
    *   End (Day 12+): Changes stop (Stationary phase).
*   **LSTM Confusion:** The LSTM sees 90% of the data as "slow change" or "no change." It learns that "doing nothing" is a safe bet. When the fast reaction hits, it reacts too slowly, causing the lag/drift you see in the graphs.

### B. The "Absolute Value" Trap
*   The LSTM is currently trained to output `y(t)`.
*   If the input is `5.00`, and the true target is `4.95`, the LSTM might predict `4.99` (safe guess).
*   In a recursive loop, this error accumulates:
    *   Step 1: Predict 4.99 (True: 4.95) -> Error 0.04
    *   Step 2: Input 4.99 -> Predict 4.98 (True: 4.80) -> Error 0.18
    *   Step 3: ... The gap widens.

## 3. Solution: Delta Learning
To fix the overlap, we must change **what** the LSTM learns.
*   **Don't learn:** "The value is 4.95."
*   **Learn:** "The value drops by 0.05."

This forces the model to capture the **active dynamics** (the slope) rather than just memorizing the position. This is the standard fix for stiff chemical reaction networks.
