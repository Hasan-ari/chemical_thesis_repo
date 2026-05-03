# Hidden RTNN Figure Mapping

This note maps the figures from `Hidden_Reactive_Transport_Neural_Network_A_Physics.pdf`
onto the current PHREEQC -> LSTM repository as a paper-aligned evaluation workflow.

## Scope

- Repo-only adaptation
- No external Zenodo or Battistel experiment assets
- Output target: paper-aligned evaluation figures, not exact pyrite-transport reproduction
- User-facing target set: Figures 3, 4, 5, 6, 7, 8, 9, and 12

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
| Figure 4 | adapted | representative time-series profiles | Time replaces depth; RMSE percentile slices replace spatial cases. |
| Figure 5 | adapted | second representative profile family | Time replaces depth; novelty-bin slices replace the paper's second profile family. |
| Figure 6 | adapted | truth/prediction/error heatmaps | Feature-time heatmaps stand in for space-time fields. |
| Figure 7 | adapted | breakthrough-style chemistry curves | One representative median-RMSE trajectory is plotted as the reference rollout. |
| Figure 8 | unsupported | markdown-only explanation | No pyrite mass or reaction-rate state exists in the dataset, so no PNG is produced. |
| Figure 9 | direct | parity plots | Existing predicted-vs-actual scatter already matches the role. |
| Figure 10 | adapted | supplementary only | Not part of the default user-facing figure bundle anymore. |
| Figure 11 | adapted | supplementary only | Not part of the default user-facing figure bundle anymore. |
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
- `figure03_training_dynamics.png`
- `figure03.md`
- `figure04_representative_profiles_a.png`
- `figure04.md`
- `figure05_representative_profiles_b.png`
- `figure05.md`
- `figure06_truth_pred_error_heatmaps.png`
- `figure06.md`
- `figure07_breakthrough_summary.png`
- `figure07.md`
- `figure08.md`
- `figure09_parity.png`
- `figure09.md`
- `figure12_generalization.png`
- `figure12.md`

## Canonical Command

```bash
env/bin/python scripts/generate_paper_figures.py --experiment-dir experiments/seq10_h128
```

This command is intentionally tied to the repo-local `env` interpreter so that figure
generation is repeatable and does not depend on system Python.
