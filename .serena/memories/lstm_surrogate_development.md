# LSTM Surrogate Model Development (Updated March 2026)

## Goal
Replace PHREEQC simulations with trained LSTM for fast prediction.

## Data Pipeline
```
PHREEQC v23 → 1000 runs (13 cols, 101 steps) → Normalize → Sliding windows → LSTM
```

## Architecture
- Input: (batch, seq_len, 13) — sliding window of PHREEQC output variables
- Output: (batch, 13) — next timestep prediction
- Framework: PyTorch
- Training: Google Colab (GPU)
- Validation: Free-running (autoregressive) — NO teacher forcing

## Key Conventions
- Never shuffle time series — sequential train/test split
- Per-variable mean/std normalization (fit on train only)
- All concentrations must stay >= 0
- Modular code in src/ — each module independently testable

## Previous Experiments (archived)
- Seq lengths tested: 3, 5, 10, 20, 30, 50
- Data points tested: 500, 1000, 2500
- Best: seq_len=5 with 500pts for short-term
- Main challenge: error accumulation in autoregressive prediction

## PHREEQC Output Columns (for LSTM)
0:time_d 1:pH 2:Ptot_atm 3:pH2_atm 4:pCO2_atm 5:pCH4_atm
6:CH4_g_mol 7:H2_g_mol 8:CO2_g_mol 9:SO4 10:Formate 11:Acetate 12:Ca