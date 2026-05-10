# Agent Instructions

## Operating Protocol

Use this file as the repo-level router and enforcer. Keep detailed workflows in
skills; do not duplicate skill bodies here.

Before substantial code, data, thesis-writing, learning, or Obsidian-memory
work:

1. Read this file.
2. Stop and load `thesis-socratic-code-ownership`.
3. If that skill is unavailable, say so and manually follow its gates:
   context, required skills, Socratic question, scientific understanding, code
   ownership, debate, learning check, verification, and Obsidian memory.

For all work:

- Inspect real files before editing or summarizing.
- Prefer GitHub connector tools for live PR/issue context; use local `gh` only
  when needed and authenticated.
- Keep PR scope explicit. Stage only files that belong to the requested change.
- Keep Obsidian vault notes out of git.
- Treat private mail and local data as sensitive unless the user explicitly
  includes them in scope.
- Verify before claiming completion.

## Project Context

This repository is an MSc thesis project on LSTM/RNN surrogate models for
PHREEQC geochemical simulations of microbial hydrogen consumption in underground
hydrogen storage.

Primary goal: replace expensive PHREEQC simulations with PyTorch sequence models
that predict chemistry time-series rollouts.

Scientific domain: anaerobic microbial metabolism in subsurface hydrogen
storage, including hydrogen consumption, methane production, sulfate reduction,
formate/acetate dynamics, and mineral interactions.

Team context:

- Chemistry advisor: PHREEQC models, MATLAB ODE references, experimental and
  synthetic geochemical data.
- CS advisor: machine-learning guidance.
- Student: data pipeline, PyTorch LSTM/RNN surrogate modelling, evaluation, and
  thesis writing.

## Known Recent Repository State

This section is a snapshot, not a substitute for live inspection. Always run
`git status --short --branch` and inspect the relevant files before acting.

Known recent branch context: `report/ch2-data`.

The current report-writing effort is Chapter 2 / Data. The recent local history
shows:

- `ace7ac5 report(ch2): draft Data section (source, sampling, derived, output)`
- `65d7a29 data(tooling): add txt-to-csv preview script for PHREEQC runs`
- `83f8f42 data(eda): add per-column summary and trajectory plots`
- `bd0b299 chore(gitignore): exclude phreeqc preview/, eda/, and biber *.blg`

Root `README.md`, `CONTEXT.md`, and `docs/adr/` do not exist yet. Until those
are created, use this file plus the files below as the repo memory:

- `docs/plans/2026-05-03-thesis-report-design.md`
- `docs/study_plan_lstm.md`
- `docs/paper_figures/*.md`
- `src/data/constants.py`
- `src/data/*`
- `src/evaluation/*`
- `report/report.tex`
- `report/sections/*.tex`

Do not assume an absent context or ADR file exists.

## Issue And PR Context

The intended workflow is issue-driven: understand, design, code or write, test,
then commit. The thesis report plan expects one issue/PR per chapter or figure
group.

GitHub is the intended tracker for this repo. Prefer GitHub connector tools for
live issue/PR context. Use local `gh` only when a connector workflow is
insufficient and `gh` is authenticated.

If GitHub lookup fails, state that clearly and fall back to local git history
and repo documents instead of inventing issue state.

## Sensitive And Local-Only Data

`chem_prof_mails/` is sensitive and ignored by git. Never commit it. Do not
quote or summarize private mail content unless the user explicitly asks for that
context in the current task.

`.agents/`, `.claude/`, `.cursor/`, local vault files, virtual environments, and
generated LaTeX/build artifacts are local working context unless the user asks
to version them.

The worktree may already contain user changes. Do not revert unrelated changes.
Known local changes have included:

- deleted `.cursor/...` files
- untracked `data/Calcite_wat_sat_data/`
- untracked `data/Trona_par_sat_data/`
- untracked `data/Trona_par_sat_no_solution_data/`

## Active Data Contract: PHREEQC v23

Primary active training data is `data/phreeqc_v23/`.

Shape and file contract:

- `input/`: 1000 `*_Input.txt` files
- `output/`: 1000 `*_Output.txt` files
- each output has one header row plus 101 time rows
- raw output columns:
  `time_d`, `pH`, `Ptot_atm`, `pH2_atm`, `pCO2_atm`, `pCH4_atm`,
  `CH4_g_mol`, `H2_g_mol`, `CO2_g_mol`, `SO4`, `Formate`, `Acetate`, `Ca`

Model features are the 12 columns after dropping `time_d`.

Canonical constants live in `src/data/constants.py`:

- `N_TRAJECTORIES = 1000`
- `N_TIMESTEPS = 101`
- `N_FEATURES = 12`
- `FEATURE_NAMES = RAW_COLUMN_NAMES[1:]`
- log-transform feature indices: `(2, 4, 5, 6, 7, 9, 10)`

Important discrepancy to check before changing data logic:
`report/sections/02-data.tex` says row 1 at `t = 0` should be dropped because
gas equilibration occurs after the first integration step, but the current
loader reads all 101 rows and only drops `time_d`.

## New Professor Data: Trona And Calcite

The current workspace includes three newer untracked data sets from May 2026:

- `data/Trona_par_sat_data/`
- `data/Trona_par_sat_no_solution_data/`
- `data/Calcite_wat_sat_data/`

These are not yet integrated into the active LSTM loader.

Observed structure:

- `Trona_par_sat_data`: 993 outputs, 7 failed inputs, 301 time steps, 33 columns.
- `Trona_par_sat_no_solution_data`: 989 outputs, 11 failed inputs, 301 time
  steps, 27 columns.
- `Calcite_wat_sat_data`: 991 outputs, 9 failed inputs, 301 time steps, 33
  columns.

These data sets represent REV-style sampling: representative element volume
around 1 cubic meter, porosity sampled roughly between 0.10 and 0.30, water/gas
volumes varied inside pore volume, rock mass derived from density, and hydrogen
molar fraction fixed around 2 percent. The generator scripts use Windows PHREEQC
paths and are not directly portable without changes.

Do not point existing training code at these folders without adding a
dataset-specific schema/config layer. They differ from `phreeqc_v23` in number
of runs, failed simulations, time steps, feature count, and output names.

## Model And Training Contract

Main model: `src/models/lstm.py::PhreeqcLSTM`.

Architecture:

- input shape: `(batch, seq_len, n_features)`
- LSTM with `batch_first=True`
- linear head maps the final hidden state to the next chemistry state
- output shape: `(batch, n_features)`

Training data uses sliding windows from normalized trajectories:

```text
X[t:t+seq_len] -> X[t+seq_len]
```

Windows must not cross trajectory boundaries.

Normalization contract:

- fit preprocessing on train data only
- apply `log1p` to selected concentration/pressure-like columns
- apply per-variable `StandardScaler`
- inverse transform uses `expm1` and clamps negatives to zero

Evaluation contract:

- always evaluate by autoregressive rollout, not teacher-forced next-step loss
- seed rollout with the first `seq_len` true steps
- feed predictions back into the input window
- compute rollout metrics only on the predicted portion after the seed window

Metric naming:

- `results.json:rmse_total` is a legacy alias for normalized total RMSE
- prefer `nrmse_total` for cross-experiment comparison
- `rmse_per_var` is original-scale per-variable RMSE for chemistry
  interpretation
- do not compare normalized total RMSE and original-scale RMSE as the same
  statistic

## Experiments And Paper Figures

Saved experiments exist under `experiments/`, especially:

- `experiments/seq3_h128`
- `experiments/seq5_h128`
- `experiments/seq10_h128`
- `experiments/seq20_h128`

The latest full paper-aligned bundle is under
`experiments/seq3_h128/paper_figures_full/`. Figures 1-12 are represented, but
Figure 8 and Figure 10 are unsupported in the current repo contract.

Canonical commands:

```bash
env/bin/python scripts/generate_paper_figures.py --experiment-dir experiments/seq10_h128
env/bin/python scripts/generate_full_paper_figures.py --experiment-dir experiments/seq3_h128
```

Do not claim the repo supports physics-informed reactive transport unless the
model/data structure changes. Current repo supports standard autoregressive LSTM
rollouts only.

Safe claims:

- LSTM approximates PHREEQC trajectory rollouts with measurable variable-wise
  error.
- Error accumulation is variable-dependent over autoregressive horizons.
- Sequence length and hidden size affect rollout quality.
- Initial-state novelty may correlate with worse rollout generalization.

Unsupported claims:

- model respects reactive-transport PDEs
- hidden states correspond to pyrite or solid-state physical evolution
- model reproduces spatial transport fields
- model explains mechanistic reaction pathways

## Thesis Report Workflow

Report target: English thesis report in `report/report.tex`.

Approved structure:

1. Introduction
2. Data
3. Model
4. Training procedure
5. Evaluation procedure
6. Results

No motivation chapter. No discussion, limitations, or future-work chapter for
the current iteration.

Current report state:

- `report/sections/02-data.tex` has the active Chapter 2 draft.
- Chapters 1, 3, 4, 5, and 6 are stubs.
- `report/figures/` has not yet been populated with generated paper figures.
- `report/references.bib` is effectively empty.
- `report/.latexmkrc` pins the LaTeX engine; check it before claiming the
  build engine.

Build from `report/` with:

```bash
latexmk
```

Use `latexmk -pvc` for continuous preview when requested.

## Verification Commands

Preferred local Python is `env/bin/python` because figure runtime checks expect
that interpreter.

Focused verification:

```bash
env/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Quality checks, if dependencies are installed:

```bash
env/bin/python -m ruff check .
env/bin/python -m ruff format --check .
env/bin/python -m pre_commit run --all-files
```

Training smoke test, if needed:

```bash
env/bin/python run_experiment.py --seq_len 3 --hidden_size 64 --epochs 1 --output_dir /tmp/chemical_thesis_smoke
```

Do not run full matrix training unless explicitly asked; it is expensive and
writes experiment outputs.

## Obsidian Vault Workflow

The user's active Obsidian vault is `/Users/macbook/Obsidian/Çalışma`.
`/Users/macbook/Obsidian/Çalışma/Fully-chemical-thesis` is a project folder
inside that vault, not a standalone vault.

Use the repo-local Obsidian skills when editing vault content:

- `obsidian-markdown` for notes, wikilinks, callouts, frontmatter, embeds.
- `obsidian-bases` for `.base` database views.
- `json-canvas` for `.canvas` maps.
- `obsidian-cli` only when Obsidian is open and the CLI is available.

Vault conventions observed:

- Existing notes mostly use simple headings and wikilinks, not heavy
  frontmatter.
- Main thesis index: `[[Tezle alakalı detaylı bilgiler]]`.
- Mathematics/learning index: `[[Akademik Matematik]]`.
- Existing raw/wiki split: source-like material goes under `raw/tez/`; digested
  notes go under `wiki/tez/`.

For this project, keep a durable second-brain layer in
`Fully-chemical-thesis/` and link it back to the existing vault indexes.

## Agent Working Rules

Read relevant source and docs before editing. State assumptions if the domain,
metric, or data schema is ambiguous.

Keep changes surgical. Match existing project style. Do not refactor unrelated
code. Do not invent abstractions unless they remove repeated complexity or match
an existing local pattern.

Prefer tests for behavioral changes. For report-only or vault-only changes,
verify by checking file existence, Markdown/YAML syntax, and relevant links.

Student focus is data and deep learning. Chemical reaction theory belongs mostly
to the chemistry advisor. Do not over-interpret chemistry or make mechanistic
claims beyond the data/model.

Use local skills when they apply:

- `thesis-socratic-code-ownership` is mandatory for substantial code, data,
  thesis-writing, learning, or Obsidian-memory work in this repo.
- `karpathy-guidelines` for surgical, verifiable code work.
- `grill-with-docs` for plan/domain terminology stress tests.
- `improve-codebase-architecture` for architecture review.
- `obsidian-*` and `json-canvas` for vault work.
- Use GitHub connector first for PR/issue context; use local `gh` only when
  needed and authenticated.

## Project Working Rules

These rules apply to every task in this project unless explicitly overridden.
Bias toward caution over speed on non-trivial work. Use judgment on trivial
tasks.

### Rule 1 — Think Before Coding

State assumptions explicitly. If uncertain, ask rather than guess. Present
multiple interpretations when ambiguity exists. Push back when a simpler
approach exists. Stop when confused and name what is unclear.

### Rule 2 — Simplicity First

Use the minimum code that solves the problem. Add nothing speculative. Do not
add features beyond what was asked. Do not add abstractions for single-use code.
If a senior engineer would call it overcomplicated, simplify.

### Rule 3 — Surgical Changes

Touch only what is needed. Clean up only your own mess. Do not improve adjacent
code, comments, or formatting. Do not refactor unrelated code. Match existing
style.

### Rule 4 — Goal-Driven Execution

Define success criteria before doing substantial work. Loop until verified.
Strong success criteria should make it clear what command, artifact, or behavior
proves the task is handled.

### Rule 5 — Use Deterministic Tools For Deterministic Work

Use code and shell tools for routing, counting, parsing, retries, and mechanical
transforms. Use the model for judgment, explanation, design, synthesis, and
review. If code can answer a factual/mechanical question more reliably, run
code.

### Rule 6 — Read Before Writing

Before adding or changing code, read the relevant exports, immediate callers,
tests, and shared utilities. If the current structure is unclear, ask or inspect
more before editing.

### Rule 7 — Surface Conflicts, Do Not Average Them

If two patterns contradict, choose one based on recency, test coverage, and
local convention. Explain the choice. Flag the other for cleanup if needed. Do
not blend conflicting patterns silently.

### Rule 8 — Tests Verify Intent

Tests should encode why behavior matters, not just what happened once. A test
that cannot fail when the intended logic breaks is weak.

### Rule 9 — Checkpoint After Significant Steps

After meaningful progress, summarize what changed, what is verified, and what
remains. Do not continue from a state you cannot describe clearly.

### Rule 10 — Match Codebase Conventions

Conformance beats personal taste inside this codebase. If a convention appears
harmful, surface it and explain the tradeoff instead of forking silently.

### Rule 11 — Fail Loud

Do not hide skipped work, skipped tests, uncertainty, or partial completion.
`Completed` is wrong if anything required was skipped silently. `Tests pass` is
wrong if only a subset was run without saying so.
