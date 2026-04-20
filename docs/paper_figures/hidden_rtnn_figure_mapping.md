# Hidden RTNN Figure Mapping

This note maps the figures from `Hidden_Reactive_Transport_Neural_Network_A_Physics.pdf`
onto the current PHREEQC -> LSTM repository as a paper-aligned evaluation workflow.

## Scope

- Repo-only adaptation
- No external Zenodo or Battistel experiment assets
- Output target: paper-aligned evaluation figures, not exact pyrite-transport reproduction

## Repo Reality

- Data shape: `1000 x 101 x 12`
- Source: `data/phreeqc_v23/output`
- Model: autoregressive `PhreeqcLSTM`
- Core evaluation path:
  - `src/evaluation/autoregressive.py`
  - `src/evaluation/comprehensive.py`
  - `src/evaluation/paper_figures.py`

## Figure Status Table

| Paper figure | Status | Repo equivalent | Reason |
| --- | --- | --- | --- |
| Figure 1 | unsupported | none | Transport workflow schematic is paper-specific. |
| Figure 2 | unsupported | none | HRTNet architecture does not exist in this repo. |
| Figure 3 | direct | training dynamics | `loss_history` already exists in `results.json`. |
| Figures 4-5 | adapted | representative time-series profiles | Time replaces depth; error percentiles replace spatial cases. |
| Figure 6 | adapted | truth/prediction/error heatmaps | Feature-time heatmaps stand in for space-time fields. |
| Figure 7 | adapted | breakthrough-style chemistry curves | One representative median-RMSE trajectory is plotted as the reference rollout. |
| Figure 8 | unsupported | none | No pyrite mass or reaction-rate state exists in the dataset. |
| Figure 9 | direct | parity plots | Existing predicted-vs-actual scatter already matches the role. |
| Figure 10 | adapted | ablation heatmap | Uses saved `seq_len x hidden_size` experiment matrix. |
| Figure 11 | adapted | sensitivity comparison | Compares normalized vs original-scale error surfaces. |
| Figure 12 | adapted | generalization vs novelty | Uses train-vs-test initial-state distance instead of zone shift. |

## Metric Contract

The repo currently mixes two different RMSE meanings:

- `results.json:rmse_total`
  - Canonical meaning: `nrmse_total`
  - Scale: normalized
  - Use: compare experiments across hyperparameters
- `results.json:rmse_per_var.*`
  - Scale: original
  - Use: interpret chemistry variables directly
- `comprehensive_stats.json:overall_rmse.mean`
  - Scale: original
  - Use: summarize trajectory-level rollout quality

The paper-aligned workflow treats `results.json:rmse_total` as a legacy alias and
normalizes it to `nrmse_total` before generating manifests.

## Generated Outputs

The figure bundle generator writes:

- `paper_figure_checklist.json`
- `metric_contract.json`
- `paper_figure_manifest.json`
- `figure3_training_dynamics.png`
- `figure4_5_representative_profiles.png`
- `figure6_truth_pred_error_heatmaps.png`
- `figure7_breakthrough_summary.png`
- `figure9_parity.png`
- `figure12_generalization_novelty.png`
- `figure10_11_ablation_sensitivity.png` when `experiments/summary.json` is available

## Canonical Command

```bash
env/bin/python scripts/generate_paper_figures.py --experiment-dir experiments/seq10_h128
```

This command is intentionally tied to the repo-local `env` interpreter so that figure
generation is repeatable and does not depend on system Python.
