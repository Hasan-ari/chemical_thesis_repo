# Input optimization v1: experiment design

Date: 2026-11-09. Status: design for the first experiment.

This document gives the design of the first input optimization experiment.
The experiment uses the frozen v2 LSTM surrogate. It finds the input values
that give the smallest predicted H2 loss for each rock. The result is a table
for the chemistry team.

## 1. Question

For each rock: which input values give the smallest predicted relative H2
loss between day 0.6 and day 180?

The answer is a prediction of the surrogate model. It is not a PHREEQC
result. The chemistry team must check the best candidates with PHREEQC.

## 2. Terms

| Term | Meaning |
|---|---|
| Surrogate | The frozen v2 LSTM. It predicts the 26 output trajectories from the 57 inputs. |
| Input | One of the 57 condition values of a run. Solution values are in mol/kgw. |
| Free input | An input that the optimizer can change. |
| Tied input | An input that the generator script computes from other inputs. It moves with the free inputs. |
| Fixed input | An input that has one value in all runs of the rock. It does not change. |
| z-space | The free inputs after the rock z-score: z = (x − rock mean) / rock std. |
| Start | One initial point of the search. Each start is one real run of the rock. |
| Candidate | The result of one start after the search. |

## 3. Objective

The relative H2 loss of one run is:

```
relative_loss = (H2(t_start) − H2(t_end)) / max(H2(t_start), eps)
```

- H2 is the output `H2_g_mol`, in mol.
- t_start = 0.6 days (time index 1). t_end = 180 days (time index 300).
- eps = `target.eps_mol` = 0.1 mol. It prevents a division by zero.
- The optimizer minimizes relative_loss.

The eps value is equal to `target.min_start_mol` (0.1 mol). The reason:

1. An eligible real run has H2(t_start) ≥ min_start_mol (section 4.1).
   Thus eps does not change the objective of an eligible real run.
2. The surrogate can predict H2(t_start) ≤ 0 for a candidate. With a small
   eps, the division then gives a very large negative loss. With
   eps = 1e-5, one Calcite start went to a predicted loss of −19319.
3. The config check stops the run if eps_mol > min_start_mol.

The absolute loss `H2(t_start) − H2(t_end)` is a second objective. The config
selects the objective. The report gives both values for each candidate.

NOTE: Measured on the real runs, the relative loss decreases when H2(t_start)
increases. The Spearman correlation is −0.50 to −0.84 in the eight rocks.
Thus the relative objective can prefer a large gas volume. The absolute loss
column shows this effect.

NOTE: About 0.5% to 3% of the runs have H2(t_start) below 1e-5 mol. The
real runs with H2(t_start) < 0.1 mol have a median relative loss of 0.17 to
0.33. Thus this corner is not a low-loss region. Section 4.1 removes it from
the starts, and section 6.1 keeps the search out of it.

## 4. Search space

The search is done for each rock separately.

1. Select the runs of the rock.
2. Find the inputs with one value in all runs. These inputs are fixed.
3. Find the tied inputs with a greedy linear test, in feature order:
   1. Fit the input on [1, current free inputs] with least squares.
   2. If R² ≥ 0.99, the input is tied. Keep the fit coefficients.
   3. If R² < 0.99, add the input to the free inputs.
4. Decode a z-space point to the 57 inputs with one affine map:
   `x = base + z @ W`.

The 0.99 threshold is in a data gap. On the eight rocks, the highest R² of a
free input is 0.9412 (CO2 in the schists). The lowest R² of a tied input is
0.9964.

| Rock | Varying inputs | Free | Tied |
|---|---|---|---|
| Calcite | 18 | 12 | 6 |
| Dolomite | 18 | 12 | 6 |
| Halite | 18 | 11 | 7 |
| Trona | 18 | 12 | 6 |
| Sandstone | 33 | 14 | 19 |
| Quartzite_schist | 33 | 21 | 12 |
| Mica_schist | 34 | 21 | 13 |
| Mica_carbonate_schist | 34 | 21 | 13 |

NOTE: In Halite, Mica_schist, and Mica_carbonate_schist, GAS_VOLUME is tied
to POROSITY. Thus the gas volume is not free in these rocks.

The rock label selects the runs. The rock label never enters the model input.

### 4.1 Eligible runs

A real run is eligible if H2(t_start) ≥ `target.min_start_mol` (0.1 mol).
Only an eligible run can be a start. The baseline and the surrogate check
also use only the eligible runs.

### 4.2 Context inputs

`input_space.context_inputs` is a list of free input names. A context input
stays at the value of its start run during the search. Use it for a site
property that the operator cannot change. The v1 config has an empty list.
Thus all free inputs can change. A tied input cannot be a context input.
The run stops with an error in this case.

### 4.3 Linear constraints

The box on the free inputs is not sufficient. A corner of the box can break
a rule of the generator scripts. Each constraint has the form
`sum(coefficient × input) ≤ upper`.

| Name | Left side | Rocks |
|---|---|---|
| `gas_and_water_fit_in_pore_space` | GAS + WATER − 1000 × POROSITY | All |
| `gas_at_most_half_of_free_pore_space` | GAS + 0.5 × WATER − 500 × POROSITY | Calcite, Dolomite, Trona, Sandstone |
| `<input> >= 0` | −input | All, one for each of the 57 inputs |

The coefficients come from the generator rules:

- Legacy rocks and Sandstone: GAS = (1000 × POROSITY − WATER) × U(0, 0.5)
  (`docs/input_generator.md`).
- Rocks with a full pore: GAS = 1000 × POROSITY − WATER.

The upper limit is not in the config. It is the max of the left side over
the real runs of the rock (data-calibrated). Thus each real run obeys each
constraint, also when the input files hold rounded values.

Measured on the real runs (max of the left side, rounded):

| Rock | Full-pore side | Half-pore side |
|---|---|---|
| Calcite, Dolomite, Trona | −27 to −29 | 2.3 to 2.5 |
| Sandstone | 0.5 | 2.1 |
| Quartzite_schist | 0.7 | 2.3 |
| Halite, Mica_schist, Mica_carbonate_schist | 5.0 (min −5.0) | 138 to 149 |

NOTE: The data show the half-pore rule in Quartzite_schist too, but the v1
config does not apply it to this rock. In Halite, Mica_schist, and
Mica_carbonate_schist, GAS is tied to POROSITY and WATER (section 4).

The violation of one constraint is `relu(left side − upper)`. It is divided
by `norm(coefficients × condition_scale)`. Thus the scaled violation is a
distance in the normalized input space of the model. All constraints have
the same unit.

## 5. Two modes

| Mode | Rule after each step |
|---|---|
| bounded | Clamp each free input to the min and max of the rock runs. |
| unbounded | No clamp. |

The unbounded mode shows where the surrogate goes without a limit. Its
results are extrapolations. Do not send them to the chemistry team as
recommendations without a PHREEQC check.

In the bounded mode, the search also adds one constraint pair
`min ≤ input ≤ max` for each varying input. The box keeps the free inputs
in range. These constraints keep the tied inputs in range.

CAUTION: A tied input can leave its observed range at a corner of the box.
For example, N2 is tied to CH4 and CO2 (not to H2S). In all eight rocks, the
N2 fit has large terms only on CH4 and CO2. H2S comes after N2 in feature
order, so it is not a source of the N2 tie. The report flags each candidate
that has an input outside the observed range of the rock.

## 6. Search algorithm

1. Select the starts from the eligible runs of the rock (section 4.1).
   Use the 8 runs with the smallest real (PHREEQC) objective. Add 56
   random eligible runs (seeded). Total: 64 starts.
2. Encode each start to z-space.
3. Run Adam for 300 steps, learning rate 0.05. All starts run in one batch.
   The starts do not interact, because the loss is a sum of the start losses.
4. After each step, apply the mode rule (clamp or no clamp).
5. Keep the point of each start with the smallest search loss
   (section 6.1) over all steps.
6. Mark a candidate as usable if it is feasible, not `low_start`, and not
   `negative` (section 7.5).
7. Rank the usable candidates by the predicted objective. Remove a
   candidate that is closer than 0.25 × the median nearest-neighbor
   distance of the train runs to a better candidate. The distance is in
   the normalized input space of the model (section 7.2).
8. Keep the 5 best candidates for each rock and mode.

### 6.1 Search loss

The search minimizes this loss for each start:

```
loss = objective + penalty_weight × (sum(violation²) + start_shortfall²)
start_shortfall = relu(min_start_mol − H2(t_start)) / min_start_mol
```

- `violation` is the scaled violation of each constraint (section 4.3).
- `penalty_weight` = 100 (`constraints.penalty_weight`).
- `start_shortfall` is a hinge on the predicted H2(t_start). It is 0 if
  H2(t_start) ≥ min_start_mol. It is 1 at H2(t_start) = 0 mol.

The hinge closes an exploit. Without it, the search could push the
predicted H2(t_start) to 0 or below. Then the relative loss becomes very
small or negative, but the chemistry cannot give this result.

A candidate is feasible if its max scaled violation is ≤ 0.01
(`constraints.feasibility_tolerance`).

## 7. Checks

### 7.1 Surrogate check on test runs

The v2 training used a rock-aware run-level split: 0.8 / 0.1 / 0.1,
seed 42. The experiment rebuilds this split from the run table.

The rebuilt split is confirmed:

- The train set has 44,096 runs. The condition scaler of v2 saw 44,096
  samples (`n_samples_seen_`).
- The test set is equal to the 5,513 runs in the v2 `run_rmse.csv` file.

For each rock, the check compares the predicted and the real objective on
the eligible test runs. It reports MAE, MAE / std, R², Spearman, and the
95th percentile of the absolute error. The same metrics on the eligible
train runs show the gap between seen and unseen runs.

The check is done for the relative loss and for the absolute loss. The
column `objective` in `surrogate_check.csv` names the objective. The search
uses only `search.objective`.

A rock is resolvable for one objective if, on the test runs:

- Spearman ≥ 0.7 (`acceptance.min_spearman`), and
- MAE / std of the real values ≤ 0.5 (`acceptance.max_mae_over_std`).

### 7.2 Distance to the data

For each candidate, the report gives the Euclidean distance to the nearest
train run of the rock. The distance is in the normalized input space of the
model: (x − condition_mean) / condition_scale, from the v2 scaler.

The report also gives a distance percentile. The reference is the
nearest-neighbor distance of up to 2000 sampled train runs of the rock
(`report.nn_sample_size`). A percentile of 100 means that the candidate is
farther from the train runs than all sampled train runs are from their
nearest neighbor. The surrogate is less reliable there.

### 7.3 Baseline

For each rock, `baseline.csv` gives two eligible real runs:

1. The run with the smallest real objective, and its prediction.
2. The run with the smallest predicted objective, and its real value.

The predicted baseline is the smallest predicted objective of the eligible
runs. A candidate is `promising` if its predicted objective is smaller than
the predicted baseline by more than the 95th percentile of the absolute
test error of the rock.

### 7.4 Implementation check

A test builds a small random LSTM. It compares the differentiable torch path
of this work stream with the numpy path of v2
(`build_condition_time_tensor` and `OutputScaler.inverse_transform`). The
two paths must give the same values.

The adapter normalizes the inputs in float64 and then casts them to the
model dtype. v2 does the same. One v2 input has mean 1.18 and std 1.4e-4.
In float32, this input loses 1e-3 of its std. Thus `predict` requires a
float64 input. A test with a narrow input checks this.

A check against the v2 file `run_rmse.csv` (56 test runs, 7 for each rock)
gives a max relative difference of 3.3e-5 for `rmse_normalized` and
2.4e-4 for `rmse_original`. The v2 evaluation path, run locally, gives the
same differences.

### 7.5 Candidate flags

| Flag | Meaning |
|---|---|
| `feasible` | The max scaled violation is ≤ the feasibility tolerance. |
| `low_start` | The predicted H2(t_start) is below min_start_mol. |
| `negative` | The predicted relative loss is below 0 (an H2 gain). |
| `out_of_range_inputs` | The inputs outside the observed range of the rock. |
| `promising` | Better than the predicted baseline by more than the error p95. |

`optimized_inputs.csv` contains only usable candidates: feasible, not
`low_start`, and not `negative`. `all_candidates.csv` contains all starts.

## 8. Outputs

Each run writes one folder:
`conditional_input_optimization/results/<timestamp>_<name>/`.

| File | Content |
|---|---|
| `config.yaml` | A copy of the experiment config. |
| `summary.md` | The result of each rock in short form. |
| `README.md` | The meaning of each file, column, and flag. |
| `manifest.yaml` | The git commit, the model file hashes, the library versions, the seeds, and the run time. |
| `optimized_inputs.csv` | The top usable candidates. All 57 inputs in mol/kgw. |
| `optimized_inputs_file_units.csv` | The same rows. Solution inputs in the units of the rock input files (mg/L for the schists). |
| `all_candidates.csv` | All starts, with the start run, the objective, and the flags. |
| `h2_trajectories.csv` | The predicted H2 of each top candidate at each time point. |
| `input_space.csv` | The role of each input for each rock: free, tied, or fixed. |
| `surrogate_check.csv` | The metrics of section 7.1, for each objective and split. |
| `baseline.csv` | The baseline of section 7.3. |
| `search_history.csv` | The min and the median search loss at each step. |

Each candidate row has: rock, mode, rank, predicted relative loss, predicted
absolute loss, predicted H2(t_start), start run id, nearest-train-run
distance, distance percentile, the flags of section 7.5, and the inputs
outside the observed range.

## 9. Model files

The model files are not in git. Put them in
`conditional_input_optimization/models/v2/`:

1. Open the Drive folder `20260903_135204_eight_rocks_condition_lstm_v1`.
2. Download `checkpoints/best.pt`.
3. Download `preprocessors.pkl` from the folder root.
4. Copy the two files to `conditional_input_optimization/models/v2/`.

NOTE: `preprocessors.pkl` was saved with scikit-learn 1.9.0. The installed
version is 1.8.0. scikit-learn shows an `InconsistentVersionWarning`. The
experiment reads only the arrays `mean_` and `scale_`. It does not call the
scikit-learn transform code.

## 10. Run

```
PYTHONPATH=. env312/bin/python -m \
    conditional_input_optimization.optimization.experiment \
    --config conditional_input_optimization/configs/optimization_v1.yaml
```

## 11. Limits

1. The surrogate is an approximation of PHREEQC. A low predicted loss is a
   hypothesis. PHREEQC must confirm it.
2. The tie fit is linear. A non-linear tie is treated as free.
3. The bounded box is per input. The real runs do not fill all corners of the
   box. A corner of the box can be far from all real runs. Section 7.2 shows
   this.
4. Gradient search finds local minima. The multi-start reduces this risk. It
   does not remove it.
