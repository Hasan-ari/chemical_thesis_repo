# Four-rock distribution visuals

Standalone EDA for the exact Calcite, Dolomite, Halite, and Trona datasets used
by the shared-model experiment. It reads local raw `Input.txt` files and only
the first two data rows from each matching `Output.txt`; it does not load or
modify the model, training pipeline, cache, notebook, or run artifacts.

From the repository root:

```bash
MPLCONFIGDIR=/tmp/chemical_thesis_mpl \
env312/bin/python -m scripts.four_rock_visuals.generate \
  --data-root data \
  --output-dir results/four_rock_visuals
```

The script uses time **indices** 0 and 1 and reads their real `time_d` values
from every file. In the current four datasets these are `0.0` and `0.6` days.
It rejects inconsistent timestamps instead of silently mixing them.

Outputs:

```text
calcite_t0_t1_output_transition.png
dolomite_t0_t1_output_transition.png
halite_t0_t1_output_transition.png
trona_t0_t1_output_transition.png
condition_distributions_all_rocks.png
```

Each rock-specific output PNG contains one readable row per output. The stacked
bar shows what fraction of matched runs decreased, stayed unchanged, or
increased from index 0 to index 1. The text on the right shows each time index's
median and 5th–95th-percentile interval plus the exact decreased/unchanged/
increased run counts. A truly constant distribution is labelled `fixed`. This
is a compact transition summary rather than a full density plot.

The condition PNG contains all 19 input features. Each rock is summarized by a
thin 5th–95th-percentile interval, a thick middle-50% interval, and a median dot.
Rock is used only to group the EDA marks; this script does not add rock to model
inputs.

These figures show the size and distribution of the first transition. They do
not establish the chemical mechanism causing it; that interpretation remains a
chemistry/advisor question.
