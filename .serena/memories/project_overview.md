# Chemical Thesis Project - Overview (Updated March 2026)

## Project Goal
Build LSTM/RNN surrogate models that mimic PHREEQC geochemical simulations of microbial H2 consumption in underground hydrogen storage.

## Team
- Chemistry Professor (Advisor #1): MATLAB ODE codes, PHREEQC models, data analysis
- CS Professor (Advisor #2): Machine learning guidance
- Student (CS MSc, also works as mobile dev): PyTorch LSTM surrogate model development

## Active Data Source: PHREEQC v23
**Location:** `data/phreeqc_v23/`
- 1000 Monte Carlo runs with randomized initial conditions
- 13 output columns: time_d, pH, Ptot_atm, pH2_atm, pCO2_atm, pCH4_atm, CH4_g_mol, H2_g_mol, CO2_g_mol, SO4, Formate, Acetate, Ca
- 101 timesteps per run (0-96 days)
- 24 input parameters per run (species concentrations + mineral moles/areas)

## Archived: MATLAB ODE Model (v4)
**Location:** `data/matlab_reference/` — READ-ONLY
- 14 state variables, 28 parameters, 12 conditions (4 minerals × 3 temps)
- Was primary data source until March 2026

## Repo Structure
```
data/phreeqc_v23/     — PRIMARY data
data/matlab_reference/ — READ-ONLY reference
src/{data,models,training,evaluation}/ — Modular Python code
notebooks/            — Colab experiments
docs/                 — Study notes, papers
tests/                — Unit tests
archive/              — Old work (don't use)
```

## Workflow
- Issue-driven: every task = GitHub Issue
- Flow: Understand → Design → Code → Test → Commit
- Student focus: DATA + DL (not chemistry theory)
- Agent must discuss actively, ask questions, suggest readings