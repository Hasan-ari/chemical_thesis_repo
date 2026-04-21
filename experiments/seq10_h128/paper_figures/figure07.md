# Figure 7

## What This Figure Shows
This figure shows key output curves for one representative rollout.

## How It Is Computed In This Repo
The repo uses the same representative trajectory idea to compare predicted vs true chemistry variables over rollout time.
Rendered asset: `experiments/seq10_h128/paper_figures/figure07_breakthrough_summary.png`
- Reference trajectory index: `14`
- Selected features: `['pH2_atm', 'pCH4_atm', 'CH4_g_mol', 'SO4', 'Formate', 'Acetate']`

## How To Read It
Compare the shape, timing, and spread of the prediction against the ground truth.

## Limitation Vs The Original Paper
Uses a median-RMSE rollout instead of an outlet transport experiment.
