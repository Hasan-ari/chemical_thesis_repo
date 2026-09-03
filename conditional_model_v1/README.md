# conditional_model_v1

Plain PyTorch pipeline for the thesis PHREEQC surrogate:

```text
numeric Input.txt conditions + time_d -> full Output.txt trajectory
```

## Quick Local Smoke Test

```bash
PYTHONPYCACHEPREFIX=/tmp/chemical_thesis_pycache \
MPLCONFIGDIR=/tmp/chemical_thesis_mpl \
env312/bin/python -m conditional_model_v1.cli.train \
  --config configs/conditional_model_v1/smoke_local.yaml
```

## Colab Environment

Set these environment variables in the notebook before running configs:

```bash
DATA_ROOT=/content/data
PROCESSED_ROOT=/content/processed
RUN_ROOT=/content/runs
```

Then run one config:

```bash
python -m conditional_model_v1.cli.train \
  --config configs/conditional_model_v1/full_colab_baseline.yaml
```

Or run a controlled sweep:

```bash
python -m conditional_model_v1.cli.sweep --configs \
  configs/conditional_model_v1/full_colab_lr1e-4.yaml \
  configs/conditional_model_v1/full_colab_lr3e-4.yaml \
  configs/conditional_model_v1/full_colab_baseline.yaml \
  configs/conditional_model_v1/full_colab_lr3e-3.yaml \
  configs/conditional_model_v1/full_colab_reduce_on_plateau.yaml \
  configs/conditional_model_v1/full_colab_cosine.yaml
```

Each run writes a self-contained run folder:

```text
config.yaml
resolved_config.json
history.csv
metrics.json
preprocessors.pkl
checkpoints/best.pt
checkpoints/final.pt
plots/loss_curve.png
plots/trajectory_examples/*.png
eval_predictions.npz
feature_metrics.csv
```

`eval_predictions.npz` stores every true/predicted trajectory in the evaluation
split. PNG plots are rendered for `plots.max_runs` runs and, by default, all
output features. Each plotted run gets one `*_all_outputs.png` overview grid
plus one PNG per output feature. Set `plots.max_runs: null` when you really
want PNGs for every run in the evaluation split.

`feature_metrics.csv` stores one row per output feature, including full-
trajectory RMSE/MAE and final-timestep RMSE/MAE on the original chemistry scale.

The global run registry is mirrored in:

```text
registry.sqlite
summary.csv
```

## Per-run RMSE and box plots

Every run writes one RMSE per evaluation run to `run_rmse.csv` (columns
`run_id, rock, rmse_normalized, rmse_original`) and to the `run_rmse` table of
`registry.sqlite`. Three box plots in `plots/boxplots/` show the distribution:
by rock, by output feature, and rock x feature (one panel per rock). A LORO
aggregate adds `plots/loro_unseen_rmse_boxplot.png` (held-out fold vs. the
reference model on the same rock) with the numbers in `loro_run_rmse.csv`.

## Leave-One-Rock-Out (LORO) Generalization Test

Question: what happens when the model meets a rock it has never seen?
`cli.loro` trains one model per held-out rock. That rock is removed from
train/val entirely and evaluated as `test_unseen`; the usual
train/val/test ratios apply to the remaining rocks, whose test share is
reported as `test_seen`. Compare both against the all-rocks reference run.

```bash
# local smoke (3 rocks, 4 runs each, 2 folds, 1 epoch)
env312/bin/python -m conditional_model_v1.cli.loro \
  --config configs/conditional_model_v1/smoke_loro_local.yaml

# Colab, all eight rocks (8 folds, same cache/model/epochs as the reference run)
python -m conditional_model_v1.cli.loro \
  --config configs/conditional_model_v1/loro_colab_eight_rocks.yaml \
  --reference-run-dir "$RUN_ROOT/<timestamp>_eight_rocks_condition_lstm_v1" \
  [--held-out-rocks Calcite Trona]   # optional subset for short sessions
```

Outputs:

- one ordinary run folder per fold, `<ts>_<name>_loro_<Rock>/`
  (`metrics.json` carries `held_out_rock`, `test_unseen`, `test_seen`),
- an aggregate folder `<ts>_<name>_loro/` with `loro_summary.csv` (rewritten
  after every fold, so an interrupted session keeps finished folds),
  `loro_rock_feature_metrics.csv`, `loro_metrics.json`, `loro_config.json`, and
  `plots/` (unseen-vs-seen-vs-reference bars in normalized and original units,
  a rock x feature heatmap, and the held-out rock's best/worst/mean overviews),
- a `loro_folds` table in `registry.sqlite` next to the existing `runs` and
  `rock_feature_metrics` tables (key: `loro_name`, `held_out_rock`).

`cli.train` refuses a config with `split.strategy: leave_one_rock_out`; the
default `rock_aware_run_level` path is unchanged. Normalized RMSE values are
computed with each fold's own output scaler, so use the original-unit columns
for strict cross-fold comparison.
