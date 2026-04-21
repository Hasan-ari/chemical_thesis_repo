# Figure 4

## What This Figure Shows
This figure shows representative rollout profiles chosen by RMSE percentile.

## How It Is Computed In This Repo
The repo selects P10, P50, and P90 trajectories from rollout RMSE and overlays ground truth vs prediction across selected variables.
Rendered asset: `experiments/seq10_h128/paper_figures/figure04_representative_profiles_a.png`
- Representative trajectories: `[84, 14, 90]`
- Selected features: `['pH2_atm', 'pCH4_atm', 'CH4_g_mol', 'SO4', 'Formate', 'Acetate']`

## How To Read It
Compare the shape, timing, and spread of the prediction against the ground truth.

## Limitation Vs The Original Paper
Time replaces depth because the current data have no spatial coordinate.
