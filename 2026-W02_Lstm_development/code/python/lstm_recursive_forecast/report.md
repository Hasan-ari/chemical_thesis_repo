1. "Chain Correction" & Recursive Forecasting
> Your Note: "haini correction yaparak gidelim... 3 seferde gibi" (Let's use chain correction... like in 3 times)
What it means: This refers to Recursive Forecasting. Instead of the model just predicting one single point and stopping, we take its prediction, feed it back into the input as if it were real data, and ask it to predict the next point. It creates a "chain" of predictions.
Why we do it: This is the only way to predict far into the future (150 steps) when you don't have the real data yet (e.g., predicting next month's weather day-by-day).
In Code: The function recursive_forecast_dense (Line 593) implements this loop.
2. Multi-Step Horizon
> Your Note: "Her adima da bakmayalim... 10 step gidelim 20 step" (Let's not look at every step... let's go 10 steps, 20 steps)
What it means: This is Multi-Step Output. Standard LSTMs predict $t+1$ (the immediate next moment). Your notes suggested jumping ahead: predicting $t+10$ or $t+20$ directly.
Why we do it: Predicting tiny steps ($t+1$) often leads to "Identity Mapping" (the model just copies the last value). Jumping 10 steps forces the model to learn the direction and physics of the reaction, not just the current position.
In Code:
    # Config class    PRED_HORIZON = 10      # Default    HORIZONS_TO_TEST = [10, 20, 30]  # Testing different jump sizes
3. Divergence Check
> Your Note: "Uzadikca kopma olacak mi" (Will there be a break/divergence as it gets longer?)
What it means: This is Stability Analysis. In recursive chains, a small error at step 1 becomes a bigger error at step 2, and eventually, the prediction can explode (diverge) away from reality.
Why we do it: To prove the model is chemically stable and doesn't hallucinate impossible values after a while.
In Code: The function analyze_divergence (Line 662) specifically compares the error at step 50 vs. step 150 to check for this "break."
4. Sequence Length
> Your Note: "100 100 yap... 100 adim sequance length" (Make it 100... 100 steps sequence length)
What it means: This is the Input Window or Memory. The model looks at the past 100 time steps to make a decision about the future.
Why we do it: Chemical reactions have "momentum." The model needs to see a long enough history (100 steps) to understand if a reaction is speeding up or slowing down.
In Code: SEQ_LEN = 100 (Line 48).
5. Fixed Step Size (Data Pre-processing)
> Your Note: "Step size fixlenmesi gerek... 1. Ve 2. Satir arasini sabitlicek sekilde inter" (Step size needs to be fixed... interpolate to fix interval between rows)
What it means: This is Uniform Sampling. Your original ODE solver used "adaptive steps" (sometimes 0.00001s, sometimes 0.1s). Neural networks require a steady, fixed "heartbeat" (e.g., exactly every 0.01 days).
Why we do it: If time gaps vary, the LSTM gets confused because it assumes every "step" takes the same amount of time.
In Code:
    # Line 276    t_eval = np.linspace(config.T_START, config.T_END, config.N_POINTS)    # This forces exactly equal spacing between every point
6. Random Validation
> Your Note: "Random yerden alabilirz." (We can take from a random place)
What it means: Randomized Testing. Instead of always testing the beginning of the reaction, we pick a random point in time to start the forecast.
In Code: start_idx = np.random.randint(min_start, max_start) (Line 809).