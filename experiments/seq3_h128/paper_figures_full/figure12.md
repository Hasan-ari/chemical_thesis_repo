# Figure 12

## Original Paper Role
generalization study

## Repo Analog
Initial-state novelty vs rollout RMSE

## What This Figure Shows
This figure shows generalization performance as a function of how unusual the initial condition is.

## What It Measures In This Study
It measures whether rollout RMSE increases as test initial states move away from the training manifold.

## How It Is Computed In This Repo
The repo computes initial-state novelty and compares it against rollout RMSE; this is an adapted generalization test, not a spatial-zone relocation test.
Rendered asset: `experiments/seq3_h128/paper_figures_full/figure12_generalization.png`
- Selected features: `['pH2_atm', 'pCH4_atm', 'CH4_g_mol', 'SO4', 'Formate', 'Acetate']`

## How To Read It
Higher novelty means the test initial state is farther from the training manifold. This is not a spatial-zone relocation test; it is an adapted OOD-style generalization view.

## Limitation Vs The Original Paper
Measures how far test initial conditions drift from the training manifold.
