# Analysis of LSTM Recursive Forecast Results

## Did the Professor's Approach Work?
**Yes, but with caveats.**

Your professor's key idea was to use **"Chain Correction"** (Recursive Forecasting) and **"Multi-Step Horizon"** (skipping steps) to stabilize the predictions.

### 1. The Good News (It Worked!)
*   **Horizon 30 is significantly better:**
    *   For $H_2$ Gas: Error dropped from **0.314** (Horizon 10) to **0.231** (Horizon 30).
    *   For Sulfate ($SO_4$): Error dropped from **0.086** to **0.076**.
    *   **Conclusion:** Skipping small steps (predicting $t+30$ instead of $t+10$) helped the model filter out noise and capture the "big picture" trend, exactly as your professor suspected ("Uzadikca kopma olacak mi" - will it break? It actually got *stronger* with longer jumps).

### 2. The Bad News (The "Drift")
*   **Errors Grow Over Time:**
    *   Look at $H_2$ Gas (Horizon 30):
        *   Step 50 Error: 0.05
        *   Step 100 Error: 0.14
        *   Step 150 Error: 0.23
    *   **Interpretation:** The error is accumulating. By Step 150, the prediction is drifting away from the ground truth. This is normal for recursive forecasting (errors compound), but it means the model isn't perfect yet. It is **stable** (not exploding to infinity), but it is **drifting**.

### 3. Visual Confirmation (From the Report)
The report shows:
*   **Horizon 10 (Short Jumps):** Higher error accumulation. The model gets "lost" faster.
*   **Horizon 30 (Long Jumps):** Lower error. The "Chain" is stronger because it has fewer links (5 jumps of 30 steps vs. 15 jumps of 10 steps to cover the same distance). Fewer links mean less compounded error.

## Summary for Professor
*"Your suggestion to use a multi-step horizon was correct. Moving from a 10-step to a 30-step prediction window reduced the error by **~26%** for Hydrogen gas. However, we still observe linear drift over 150 steps. The 'Chain Correction' is working—the model is stable—but we may need to explore 'Delta Learning' (learning the rate of change) to further reduce this drift."*
