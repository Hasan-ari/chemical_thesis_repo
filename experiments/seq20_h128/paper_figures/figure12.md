# Figure 12

## What This Figure Shows
This figure shows generalization performance as a function of how unusual the initial condition is.

## How It Is Computed In This Repo
The repo computes initial-state novelty and compares it against rollout RMSE; this is an adapted generalization test, not a spatial-zone relocation test.
Rendered asset: `experiments/seq20_h128/paper_figures/figure12_generalization.png`
- Selected features: `['pH2_atm', 'pCH4_atm', 'CH4_g_mol', 'SO4', 'Formate', 'Acetate']`

## How To Read It
Higher novelty means the test initial state is farther from the training manifold. This is not a spatial-zone relocation test; it is an adapted OOD-style generalization view.

## Limitation Vs The Original Paper
Measures how far test initial conditions drift from the training manifold.
