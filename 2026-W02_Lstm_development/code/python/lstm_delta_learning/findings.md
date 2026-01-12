# Findings: LSTM Delta Learning

## Experiment Context
- **Goal:** Reduce drift and improve overlap in stiff reaction phases (exponential growth).
- **Hypothesis:** Training on $\Delta y$ (rate of change) will force the LSTM to learn the derivative/slope, preventing it from defaulting to "identity mapping" (lazy learning) during slow phases.
- **Dataset:** Basalt 25°C (ODE output from MATLAB v4 two-phase model).
- **Architecture:** Stacked LSTM (128->64), Horizon=30.

---

## MATLAB Model Analysis (v4)

### State Variables (14 total)
| Index | Variable | Description | Units |
|-------|----------|-------------|-------|
| 0 | nH2_g | H2 gas in headspace | mmol |
| 1 | nCO2_g | CO2 gas in headspace | mmol |
| 2 | nCH4_g | CH4 gas in headspace | mmol |
| 3 | nH2S_g | H2S gas in headspace | mmol |
| 4 | H2_aq | Dissolved H2 | mmol/L |
| 5 | CO2_aq | Dissolved CO2 | mmol/L |
| 6 | SO4 | Sulfate concentration | mmol/L |
| 7 | FeS | Precipitated iron sulfide | mmol/L |
| 8 | X | Biomass concentration | mmol/L |
| 9 | Acetate | Acetate concentration | mmol/L |
| 10 | HCO3 | Bicarbonate (constant) | mmol/L |
| 11 | S_tot | Total dissolved sulfide | mmol/L |
| 12 | Lag | Lag phase activation (0-1) | - |
| 13 | Fe_pool | Dissolved Fe(II) pool | mmol/L |

### Key Physics (Two-Phase Model)
1. **Gas-Liquid Transfer (Henry's Law):**
   - $C_{eq} = H_{cp} \times p$ (equilibrium concentration)
   - $J = k_{la} \times (C_{eq} - C_{aq})$ (mass transfer flux)
   - Henry constants @25°C: H2=0.78, CO2=34.0, H2S=90.0 mmol/L/atm

2. **Sulfide Speciation (pH-dependent):**
   - $f_{HS} = \frac{1}{1 + 10^{pK_a - pH}}$ where $pK_a = 7.05$
   - $HS^- = S_{tot} \times f_{HS}$
   - $H_2S_{aq} = S_{tot} \times (1 - f_{HS})$

3. **Three Microbial Pathways:**
   - Methanogenesis: $4H_2 + CO_2 \rightarrow CH_4 + 2H_2O$
   - Sulfate Reduction: $4H_2 + SO_4^{2-} \rightarrow H_2S + ...$
   - Homoacetogenesis: $4H_2 + 2CO_2 \rightarrow Acetate + ...$

4. **FeS Precipitation (Fe-limited):**
   - $r_{prec} = \min(k_{prec} \times \max(0, HS_{aq} - HS_{sat}), Fe_{pool})$

### Parameter Vector (28 parameters)
| Range | Parameters | Description |
|-------|------------|-------------|
| 1-3 | k_m, k_s, k_a | Reaction rate constants |
| 4-6 | Y_m, Y_s, Y_a | Biomass yields |
| 7-9 | KI_m, KI_s, KI_a | Sulfide inhibition |
| 10-13 | k_prec, HS_sat, H2_th, DG_th | Precipitation & thermo |
| 14-16 | K_H2, K_SO4, K_CO2 | Monod half-saturation |
| 17-19 | kla_H2, kla_CO2, kla_H2S | Mass transfer coefs |
| 20-22 | b, t_lag, w_lag | Decay & lag phase |
| 23-24 | k_diss_gyp, beta_SO4_m | Gypsum & competition |
| 25-28 | phi_H2, phi_CO2, phi_H2S, alpha_H2S | Henry scale factors |

### Experimental Data (Basalt 25°C)
- **Source:** Muller 2024 dataset
- **Duration:** 0-19 days (11 time points)
- **Observables:** nH2_g, nCO2_g, nCH4_g, nH2S_g, pH, SO4
- **Initial H2:** 9.074 mmol (9074 µmol)

---

## Output File Structure (Basalt_25C_inc_rates.dat)

21 columns total:
```
Time(days) nH2_g nCO2_g nCH4_g nH2S_g H2_aq CO2_aq SO4 FeS X
Acetate HCO3 S_tot H2S_aq HS Lag Fe_pool r_meth r_sulf r_precip r_aceto
```

- Rows: Dense timesteps (adaptive ODE output)
- Time range: 0 to 19 days
- Contains both states AND reaction rates (useful for PINN!)

---

## Data Source Decision: Uniform Grid vs .dat File

### The Timestep Problem

**MATLAB's ODE15s uses adaptive timesteps:**
- Smaller dt during stiff/fast regions (microseconds)
- Larger dt during slow regions (hours/days)
- Example from .dat: `0.000002, 0.000005, ... 0.5, 1.0, 2.0` (non-uniform!)

**LSTM Requirements:**
- Expects **fixed-interval sequences** (e.g., every 0.01 days)
- Hidden state $h_t$ implicitly assumes uniform time delta between steps
- Non-uniform timesteps → model learns "wrong" temporal relationships

### Decision: Re-generate with Uniform Grid (CORRECT APPROACH)

```python
t_eval = np.linspace(T_START, T_END, N_POINTS)  # Fixed grid!
sol = solve_ivp(..., t_eval=t_eval, ...)  # Forces uniform output
```

By passing `t_eval` to `solve_ivp`, we force the solver to interpolate onto a **uniform grid** (2500 points over 20 days = dt ≈ 0.008 days).

### Options Compared

| Approach | Pros | Cons |
|----------|------|------|
| **Re-generate with uniform `t_eval`** | Fixed dt, LSTM-compatible | Must re-implement ODE in Python |
| **Load .dat + resample/interpolate** | Uses exact MATLAB output | Interpolation error in stiff regions |
| **Time-LSTM / Neural ODE** | Handles non-uniform dt | Complex, overkill for thesis |

### When to Use .dat File
- **Validation targets** - Compare final predictions against MATLAB output
- **Reaction rates** - Could use as auxiliary loss in PINN approach
- **NOT for direct LSTM training** - Non-uniform timesteps break sequence learning

---

## Results Log

### 1. Feature Engineering Check
- [ ] **pH Feature:** Does adding pH improve the "Sulfide Speciation" prediction?
    - *Observation:* (Pending)

### 2. Delta vs Absolute Comparison
- [ ] **H2 Gas Drift:**
    - Absolute Model Error @ 150: 0.231 (Baseline)
    - Delta Model Error @ 150: (Pending)
- [ ] **Lag Phase Handling:**
    - Does the Delta model "wake up" faster when the exponential phase hits?
    - *Observation:* (Pending)

---

## Conclusions
- (Pending execution)
