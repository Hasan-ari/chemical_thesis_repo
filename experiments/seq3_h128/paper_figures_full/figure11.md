# Figure 11

## Original Paper Role
sensitivity analysis

## Repo Analog
Normalized-vs-original-scale sensitivity heatmaps

## What This Figure Shows
This figure shows sensitivity to a modeling hyperparameter.

## What It Measures In This Study
It measures sensitivity of conclusions to normalized versus original-scale quality metrics.

## How It Is Computed In This Repo
The repo analog compares normalized training/evaluation RMSE against original-scale rollout RMSE because the LSTM has no physics-loss weighting factor.
Rendered asset: `experiments/seq3_h128/paper_figures_full/figure11_metric_sensitivity.png`
- No extra run-specific metadata recorded for this figure.

## How To Read It
Compare the shape, timing, and spread of the prediction against the ground truth.

## Limitation Vs The Original Paper
Reuses saved experiments to show metric sensitivity rather than physics-loss sensitivity.
