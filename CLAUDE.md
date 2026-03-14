# Chemical Thesis Repository - CLAUDE.md

## Project Overview

MSc thesis project: Building LSTM/RNN surrogate models that mimic PHREEQC geochemical simulations of microbial H2 consumption in underground hydrogen storage environments.

**Team:**
- Chemistry Professor (Advisor #1): MATLAB ODE codes, PHREEQC models, experimental data analysis
- CS Professor (Advisor #2): Machine learning guidance
- Student (CS MSc): PyTorch LSTM/RNN surrogate model development

**Scientific domain:** Anaerobic microbial metabolism in subsurface H2 storage — H2 consumption producing CH4, H2S, acetate via methanogenesis, sulfate reduction, and homoacetogenesis.

## Repository Structure

```
chemical_thesis_repo/
├── data/
│   ├── phreeqc_v23/               # PRIMARY — PHREEQC geochemical model + 1000 runs
│   │   ├── model_v23.phr          # PHREEQC reaction model (Mura 2024 based)
│   │   ├── main.py                # Data generator (multiprocessing, 1000 runs)
│   │   ├── phreeqc.dat            # Thermodynamic database
│   │   ├── input/                 # 1000 input files (randomized initial conditions)
│   │   ├── output/                # 1000 output files (time-series, 101 steps each)
│   │   ├── default_output.txt     # Reference baseline output
│   │   └── requirements.txt       # numpy, matplotlib, phreeqpy
│   └── matlab_reference/          # READ-ONLY — Professor's MATLAB ODE codes
│       ├── muller_2024/           # Low-pressure (4 minerals × 3 temps = 12 conditions)
│       └── mura_2025/             # High-pressure (36°C, ~60 bar)
│
├── src/                           # Modular Python source code
│   ├── data/                      # Data loading, normalization
│   ├── models/                    # LSTM, RNN model definitions
│   ├── training/                  # Training loops, schedulers
│   └── evaluation/                # Metrics, plotting, autoregressive eval
│
├── notebooks/                     # Jupyter/Colab experiment notebooks
├── experiments/                   # Experiment configs and results
├── docs/                          # Study notes, theory, research papers
│   ├── study_plan_lstm.md         # Self-study plan for DL theory
│   └── lstm_fluid_reserach.pdf    # Background research paper
├── tests/                         # Unit tests
├── archive/                       # OLD — Do NOT use for active development
│
├── CLAUDE.md                      # This file
├── .gitignore
└── requirements.txt               # (to be created: project-wide dependencies)
```

## Primary Data Source: PHREEQC v23

**Location:** `data/phreeqc_v23/`

Full equilibrium + kinetics geochemical model simulating the Mura 2024 high-pressure experiment.

### Output Format (13 columns, 101 timesteps per run)
```
0: time_d     — Simulation time (days, 0-96)
1: pH         — Solution pH
2: Ptot_atm   — Total pressure (atm)
3: pH2_atm    — H2 partial pressure (atm)
4: pCO2_atm   — CO2 partial pressure (atm)
5: pCH4_atm   — CH4 partial pressure (atm)
6: CH4_g_mol  — CH4 in gas phase (mol)
7: H2_g_mol   — H2 in gas phase (mol)
8: CO2_g_mol  — CO2 in gas phase (mol)
9: SO4        — Sulfate concentration (mol/kgw)
10: Formate   — Formate concentration (mol/kgw)
11: Acetate   — Acetate concentration (mol/kgw)
12: Ca        — Calcium concentration (mol/kgw)
```

### Input Format (24 parameters per run)
Randomized species concentrations (±50% uniform) + mineral moles/surface areas.

### PHREEQC Kinetic Reactions
1. **SR_AC** — Sulfate reduction on acetate
2. **SR_FOR** — Sulfate reduction on formate
3. **ACETO_METH** — Acetoclastic methanogenesis
4. **FORMATE_H2_EQ** — Formate-H2 reversible shuttle
5. **SR_H2** — Sulfate reduction on H2
6. **ACETO_H2** — Homoacetogenesis (H2 + CO2 → acetate)
7. **FORMATE_H2** — Formate production from H2
8. **METH_H2** — Hydrogenotrophic methanogenesis (H2 + CO2 → CH4)

All reactions use Monod kinetics with HS⁻ inhibition. Minerals (Quartz, Calcite, Pyrite, Barite, Illite) undergo surface-area-controlled dissolution.

## Reference: MATLAB ODE Model (v4 Two-Phase)

**Location:** `data/matlab_reference/` — READ-ONLY, not used for active development.

Custom Monod-kinetics ODE for low-pressure Muller 2024 experiments:
- 14 state variables, 28 fitted parameters, 12 conditions
- Solver: `ode15s` (stiff), Python equiv: `solve_ivp(method='BDF')`

## LSTM Surrogate Model

### Goal
Replace computationally expensive PHREEQC simulations with trained LSTM for fast prediction.

### Data Pipeline
```
PHREEQC v23 → 1000 randomized runs → Output time-series → Normalize → Sliding windows → LSTM training
```

### Architecture
- **Input:** Sliding window of states — shape `(batch, seq_len, n_features)`
- **Output:** Next state prediction — shape `(batch, n_features)`
- **Framework:** PyTorch
- **Validation:** Free-running (autoregressive) prediction without teacher forcing

### Previous Experiments (archived)
- Sequence lengths tested: 3, 5, 10, 20, 30, 50
- Data points tested: 500, 1000, 2500
- Main challenge: Error accumulation in long-horizon autoregressive prediction

## Development Workflow

### Issue-Driven Development
- Every task is tracked as a GitHub Issue
- Flow per issue: Understand → Design → Code → Test → Commit
- Agent must actively discuss, ask research questions, suggest readings

### Code Conventions
- **Modular code** in `src/` — each module independently testable
- **Time series:** Never shuffle — use sequential train/test split
- **Normalization:** Per-variable mean/std (fit on train only)
- **Validation:** Always use free-running prediction (not teacher forcing)
- **Non-negativity:** All concentrations must stay ≥ 0

### Stack
- PyTorch (LSTM, RNN)
- numpy, scipy, matplotlib
- Google Colab (GPU training)
- GitHub Issues (task tracking)

### Important Notes
- `chem_prof_mails/` is in `.gitignore` — sensitive, never commit
- `archive/` is read-only — all active work in `src/`, `data/`, `experiments/`
- PHREEQC model uses CVODE solver with tolerance 1e-6
- Student's focus is DATA + DL — chemical reaction details are professor's domain, don't deep-dive into chemistry theory

## Available Plugins & Skills

Agent should use these proactively at the right moments.

### Core Workflow
| Skill | When to Use |
|-------|-------------|
| `/commit` | Her issue bitiminde temiz commit oluştur |
| `/chemistry` | PHREEQC verisi veya model hakkında hızlı soru-cevap |
| `superpowers:brainstorming` | Yeni modül veya feature tasarımından ÖNCE — intent ve requirements keşfet |
| `superpowers:test-driven-development` | Modül yazarken önce test yaz, sonra implementasyon |
| `superpowers:verification-before-completion` | Issue kapatmadan veya commit atmadan önce son doğrulama |
| `superpowers:writing-plans` | Çok adımlı bir görev başlamadan önce plan yaz |
| `superpowers:systematic-debugging` | Bug veya test failure durumunda sistematik debug |

### Code Quality
| Skill | When to Use |
|-------|-------------|
| `coderabbit:review` | Kod yazdıktan sonra otomatik AI code review |
| `superpowers:requesting-code-review` | Major feature tamamlandığında review iste |
| `superpowers:finishing-a-development-branch` | Branch tamamlandığında merge/PR kararı |
| `/simplify` | Yazılan kodu sadeleştir ve kalitesini artır |

### Documentation & Output
| Skill | When to Use |
|-------|-------------|
| `document-skills:xlsx` | Deney sonuçlarını Excel tablosuna çıkar |
| `document-skills:docx` | Tez bölümü taslağı veya hocaya rapor yaz |
| `document-skills:pdf` | PDF okuma/oluşturma (araştırma makaleleri) |

### Memory & Continuity
| Skill | When to Use |
|-------|-------------|
| `episodic-memory:search-conversations` | "Geçen sefer ne yapmıştık?" — önceki konuşmaları ara |
| `episodic-memory:remembering-conversations` | Benzer bir sorun daha önce çözüldü mü? |
| `claude-md-management:revise-claude-md` | CLAUDE.md'yi güncel tut |

### Not Relevant for This Repo
figma, atlassian, pptx, slack-gif, brand-guidelines, frontend-design, webapp-testing — bu repo bilimsel Python, UI yok.
