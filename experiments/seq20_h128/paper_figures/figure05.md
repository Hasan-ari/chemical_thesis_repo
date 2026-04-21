# Figure 5

## What This Figure Shows
This figure shows a second representative profile family, now grouped by novelty level instead of RMSE percentile.

## How It Is Computed In This Repo
The repo bins test trajectories into low, mid, and high initial-state novelty groups and plots one representative trajectory from each group.
Rendered asset: `experiments/seq20_h128/paper_figures/figure05_representative_profiles_b.png`
- Representative trajectories: `[24, 8, 75]`
- Selected features: `['pH2_atm', 'pCH4_atm', 'CH4_g_mol', 'SO4', 'Formate', 'Acetate']`

## How To Read It
Compare the shape, timing, and spread of the prediction against the ground truth.

## Limitation Vs The Original Paper
Keeps the paper's comparison role while staying within available data.
