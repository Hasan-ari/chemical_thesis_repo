# Thesis report — design

Date: 2026-05-03
Status: approved

## Goal

Produce an English-language thesis report (`report.tex`) covering the LSTM
surrogate work end-to-end, written so that a reader without chemistry background
can follow the data, model, training procedure, evaluation procedure, and the
saved figures.

## Report structure

1. Introduction — short. Problem statement only.
2. Data — PHREEQC v23 outputs, 1000 runs × 101 timesteps × 13 features,
   normalisation.
3. Model — LSTM, accessible explanation of recurrence and gating, then the
   formal definition.
4. Training procedure — train / val / test split, sliding windows, MSE loss,
   optimiser, stopping criterion.
5. Evaluation procedure — autoregressive (free-running) prediction, metric
   definitions (RMSE, MAE, R²) with formulas.
6. Results — figure by figure. The largest chapter; one careful pass per saved
   figure explaining what it plots, how each number is computed, and how to
   read it.

No motivation chapter. No discussion / limitations / future-work chapter
(out of scope for this iteration).

## Build

`xelatex` via `latexmk`. A single `.latexmkrc` pins the engine; the user types
`latexmk` and gets `report.pdf`. Continuous preview with `latexmk -pvc`.

## Workflow per chapter

- **Issue** — "Chapter N: Title". Scope, deliverables, acceptance criteria.
- **PR** — chapter draft into `report/sections/0N-*.tex`, plus figure references
  and bibliography entries. Some chapters may need a follow-up polish PR.

## Issue plan

1. Scaffolding *(this PR)*
2. Chapter 1+2: Introduction & Data
3. Chapter 3: Model — LSTM
4. Chapter 4: Training procedure
5. Chapter 5: Evaluation procedure
6. Chapter 6: Results — one PR per figure (~12 PRs)
