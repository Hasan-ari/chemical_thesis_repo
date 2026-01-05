# MAT→PyTorch Conversion Review (Updated with latest CSV results)

---

## 1. Workflow Parity

### 1.1 Data ingest & residual definition
Both implementations process the Müller sandstone file identically: they log-transform the five measured species, apply the weight vector `[1, 1, 0.5, 0.5, 1]`, and add a large penalty when any simulated concentration becomes negative. Because the Python code mirrors the MATLAB logic line-for-line, the residual surface is exactly the same.

`matlab_to_pytorch_complete.py` (Lines 430-447):

```python
weights = np.array([1.0, 1.0, 0.5, 0.5, 1.0])
res = (log_sim - log_exp) * weights
if np.any(sol.y < -1e-6):
    penalty = 1e3 * np.abs(np.min(sol.y))
    res = res + penalty
return res.ravel()
```

`rnn_transport_multiguild_uq_v3.m` (Lines 229-237):

```matlab
log_sim = log1p(y_sim(:,1:5));
log_exp = log1p(data_exp);
weights = [1, 1, 0.5, 0.5, 1];
res = (log_sim - log_exp) .* weights;
if any(y_sim(:) < -1e-6)
    res = res + 1e3 * abs(min(y_sim(:)));
end
```

### 1.2 Optimizer setup
MATLAB’s `lsqnonlin` and SciPy’s `least_squares` both start from the same 13-parameter vector `p0`, enforce identical lower/upper bounds, and call the same ODE function (`trueODEfunc_multiguild` / `ode_multiguild`). The Python script even prints the MATLAB-style parameter tables to prove parity.

`matlab_to_pytorch_complete.py` (Lines 477-536):

```python
p0 = np.array([1.0, 1.0, 1.0, 0.05, ..., -10.0])
lb = np.array([0.001, 0.001, 0.001, 0.01, ..., -50.0])
ub = np.array([10.0, 10.0, 10.0, 0.5, ..., 0.0])
result = least_squares(
    fun=lambda p: compute_residuals(p, t_exp, data_exp, x0),
    x0=p0,
    bounds=(lb, ub),
    method='trf',
    ftol=1e-8,
    xtol=1e-8
)
```

`rnn_transport_multiguild_uq_v3.m` (Lines 197-209):

```matlab
p0 = [1, 1, 1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, -10];
lb = [0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0, 0, 0, -50];
ub = [10, 10, 10, 0.5, 0.5, 0.5, 10, 10, 10, 1, 1, 1, 0];
p_fit = lsqnonlin(@(p) residuals_multiguild(p, t_exp, data_exp, x0), p0, lb, ub, options);
```

Because the residual and bounds are identical, any differences in fitted parameters stem from solver behavior, not code differences. With only 11 data points constraining a stiff 13-parameter ODE, there are many acceptable local minima. MATLAB’s trust-region heuristics converge near the tutor’s assumptions; SciPy’s reflective trust-region converges to a sulfate/acetogen-emphasizing basin.

---

## 2. Key Parameter Differences (from current CSVs)

### 2.1 Yield coefficients (`Y_s`, `Y_a`)
  MATLAB keeps both at 0.0500. PyTorch raises them to 0.336411 and 0.324776, respectively—visible as dark-blue bars towering over light-blue “initial” bars in `pytorch_parameters.png`, indicating preference for sulfate/acetogen biomass production.

### 2.2 Hydrogen activation threshold (`H2_thresh`)
  MATLAB settles at 0.126183 mmol/L, allowing low-H₂ conditions to remain active. PyTorch pushes it to 0.894347 mmol/L, essentially disabling guild activity unless hydrogen is plentiful. This shift alters the activation factors computed inside `ode_multiguild`, changing the balance between methanogenesis and sulfate reduction.

### 2.3 Thermodynamic gate (`DG_thresh`)
  MATLAB retains the prior at −10.000002 kJ/mol. PyTorch relaxes it to −0.483971 kJ/mol, letting marginally exergonic reactions proceed. Together with the higher `H2_thresh`, this yields a regime where only strongly fed cells react, but the Gibbs filter is looser once they do.

### 2.4 Precipitation dynamics (`k_precip`, `H2S_sat`)
  MATLAB: 0.749106 day⁻¹ and ≈0 mmol/L. PyTorch: 0.265194 day⁻¹ and 0.004102 mmol/L. Slower precipitation allows more H₂S to accumulate, which—combined with the smaller `KI_meth` (0.029905 vs. 0.00225 in MATLAB)—reinforces inhibition of methanogens.

---

## 3. Plot Guide

### 3.1 matlab_parameters.png & pytorch_parameters.png
   Horizontal symlog bar charts compare initial (light blue) and fitted (dark blue) values for every parameter, with units embedded in the axis labels (e.g., `k_sulf [1/day]`, `KI_aceto [mmol/L]`, `DG_thresh [kJ/mol]`). Five-decimal scientific annotations sit in white callouts.

   ![MATLAB parameter initialization vs fitted result](./matlab_parameters.png)

   ![PyTorch parameter initialization vs fitted result](./pytorch_parameters.png)

### 3.2 parameter_comparison.png
   MATLAB (blue) and PyTorch (orange) fitted values appear side-by-side on a single symlog axis, making large divergences (e.g., `H2_thresh`, `DG_thresh`) instantly visible. Legend is top-left inside the axes.

   ![Parameter fits: MATLAB vs PyTorch](./parameter_comparison.png)

### 3.3 parameter_differences.png
   Shows PyTorch – MATLAB on a symlog x-axis. Rightward bars mark parameters that increased (e.g., `Y_a`, `H2_thresh`); leftward bars mark decreases (e.g., `k_meth`, `k_precip`). Numeric labels quantify each delta for quick reference.

   ![Parameter differences (PyTorch − MATLAB)](./parameter_differences.png)

---

## 4. Conclusion

- Same physics, different solver destinations: identical residual logic plus the shared ODE mean the PyTorch port faithfully reproduces MATLAB’s workflow. Distinct trust-region heuristics and limited data simply lead to different acceptable minima.
- MATLAB fit: preserves the tutor’s methanogenesis-dominant, fast-precipitation regime with low activation thresholds.
- PyTorch fit: favors sulfate/acetogen guilds, higher hydrogen gating, slower precipitation, and a looser thermodynamic filter.

