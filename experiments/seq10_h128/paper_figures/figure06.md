# Figure 6

## What This Figure Shows
This figure shows truth, prediction, and error as aligned heatmaps for one representative rollout.

## How It Is Computed In This Repo
The repo chooses one reference trajectory and plots feature-by-time heatmaps for truth, prediction, and normalized absolute error.
Rendered asset: `experiments/seq10_h128/paper_figures/figure06_truth_pred_error_heatmaps.png`
- Reference trajectory index: `14`
- Selected features: `['pH2_atm', 'pCH4_atm', 'CH4_g_mol', 'SO4', 'Formate', 'Acetate']`

## How To Read It
Compare the shape, timing, and spread of the prediction against the ground truth.

## Limitation Vs The Original Paper
A representative rollout stands in for the paper's space-time field.
