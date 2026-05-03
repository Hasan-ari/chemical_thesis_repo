# Hidden RTNN Math Foundations vs This Repo

This note explains what transfers from the paper's mathematics into the current
PHREEQC -> LSTM surrogate workflow, and what does not.

## Paper Side

The paper uses a physics- and chemistry-informed surrogate:

- Inputs include space, time, and scenario information.
- One network tracks hidden solid-state behavior.
- Another network predicts chemistry fields while satisfying transport and reaction
  structure through residual terms in the loss.
- The loss is not just data mismatch. It also penalizes violations of governing
  reactive-transport equations.

Conceptually, the paper learns:

```text
field(t, z) = model_theta(t, z, hidden_state, scenario)
```

with an objective of the form:

```text
L_total = L_data + omega * L_physics
```

This means the model is judged both on fit quality and on whether its predictions
respect transport/kinetic constraints.

## Repo Side

This repository uses a standard autoregressive LSTM on trajectory windows:

```text
X_(t+1) = f_theta(X_(t-seq+1:t))
```

where:

- `X_t` is a 12-variable chemistry state
- the model sees only past states, not an explicit spatial coordinate
- training loss is plain next-step MSE on normalized data
- long-horizon behavior is evaluated by feeding predictions back into the input window

The most important evaluation distinction is:

- training optimizes one-step prediction
- evaluation measures multi-step rollout stability

## What Transfers Cleanly

These scientific questions transfer well from the paper into this repo:

1. How does prediction quality evolve over training?
2. How close are predictions to reference data across all variables?
3. Which variables remain stable under long autoregressive rollout?
4. Which hyperparameter settings generalize better?
5. Do more unusual initial states lead to larger rollout error?

These questions justify the adapted figures for:

- training dynamics
- parity
- ablation
- sensitivity
- generalization

## What Does Not Transfer Cleanly

The following paper claims cannot be reproduced faithfully with repo-only data:

1. Spatial profile accuracy over depth
2. Space-time transport field agreement
3. PDE residual satisfaction
4. Hidden solid-state evolution
5. Reaction-rate diagnostics tied to pyrite mass
6. Experiment overlays from external pyrite-column measurements

The reason is structural: this repo does not expose `z`, transport coefficients, or
physics residual terms in either the model inputs or the loss.

## Interpreting the Paper-Aligned Figures

The paper-aligned figures should be read as:

- paper-aligned evaluation views
- not evidence of reactive-transport physics consistency
- not proof of mechanistic interpretability

In particular:

- `Figure 6` adaptation shows feature-time error structure, not a transport field
- `Figure 12` adaptation shows novelty relative to the train initial-state manifold,
  not physical relocation of a reactive inclusion
- `Figure 11` adaptation studies metric sensitivity across saved experiments,
  not sensitivity to a physics-loss weight

## Safe Thesis-Level Claims

The current repo can support these claims:

- The LSTM can approximate PHREEQC trajectory rollouts with measurable variable-wise
  error.
- Error accumulation is variable-dependent and increases across the autoregressive
  horizon.
- Sequence length and hidden size materially affect rollout quality.
- Test trajectories that start farther from the training initial-state manifold can
  exhibit worse generalization.

The current repo cannot support these claims without new data or new model structure:

- The surrogate respects reactive-transport PDEs.
- Hidden states correspond to physical pyrite evolution.
- The model reproduces spatially resolved transport chemistry.
- The surrogate can explain mechanistic reaction pathways.

## Practical Reading Rule

When discussing results, use the following split:

- `nrmse_total` for cross-experiment comparison
- original-scale RMSE for chemistry interpretation

Do not compare those two quantities as if they were the same statistic.
