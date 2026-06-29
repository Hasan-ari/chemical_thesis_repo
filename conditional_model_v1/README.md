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
