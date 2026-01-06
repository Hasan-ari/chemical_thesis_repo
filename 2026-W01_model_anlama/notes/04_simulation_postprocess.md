# Simulation & Post-Processing - Satır Satır Kod Açıklaması

**Bölüm**: Satır 118-256 (Final simulation + Post-processing)
**Amaç**: Dense grid simülasyon, diagnostics, plotting

---

## SIMULATION (Satır 118-124)

### Satır 118: ODE Function Handle
```matlab
odes = @(t,y) model_mixed(t,y,p_fit,env);
```
- Anonymous function
- `p_fit` kullanılıyor (fitted parameters)
- Artık optimize değil, fixed!

### Satır 119: ODE Options
```matlab
opts = odeset('NonNegative',1:14,'RelTol',1e-8,'AbsTol',1e-10,'MaxStep',0.5);
```
- Aynı options (fitting'teki gibi)

### Satır 120: Final Solver Call
```matlab
[t_sim, y_sim] = ode15s(odes, [0 t_exp(end)], y0, opts);
```

**Fark (Fitting vs Simulation):**
- **Fitting**: `ode15s(odes, t_exp, y0, opts)` → Sadece t_exp noktalarında (20 nokta)
- **Simulation**: `ode15s(odes, [0 t_end], y0, opts)` → Dense grid (500+ nokta)

**Output:**
- `t_sim`: Dense time vector (örnek: 500 nokta)
- `y_sim`: 500×14 matrix

---

## POST-PROCESSING

### Satır 126-127: Sulfide Speciation
```matlab
[H2S_aq, HS_aq] = speciate_sulfide(y_sim(:,12), env.pH_fun(t_sim), env.pKa_H2S);
```

**Ne yapıyor?**
- `y_sim(:,12)` → S_tot (tüm timesteps)
- `env.pH_fun(t_sim)` → Her timestep için pH
- Helper function çağır → H2S_aq, HS_aq vektörleri döndür

---

### Satır 129-164: Sulfur Mass Balance Diagnostic

**Satır 133-136: Mass Calculation**
```matlab
S_gas_mmol = y_sim(:,4);                 % nH2S_g (mmol)
S_aq_mmol  = y_sim(:,12) * env.Vl;       % S_tot * Vl (mmol/L → mmol)
S_FeS_mmol = y_sim(:,8)  * env.Vl;       % FeS * Vl
S_total_model = S_gas_mmol + S_aq_mmol + S_FeS_mmol;
```

**Satır 138-145: Cumulative Production**
```matlab
rates_over = zeros(length(t_sim),4);
for k = 1:length(t_sim)
    rates_over(k,:) = rate_out_mixed(t_sim(k), y_sim(k,:), p_fit, env);
end
r_sulf_vec   = rates_over(:,2);  % Sulfate reduction rate
S_prod_cum   = cumtrapz(t_sim, r_sulf_vec) * env.Vl;
```

**Ne yapıyor?**
- Loop: Her timestep için rate hesapla
- `cumtrapz`: Kümülatif integral (trapezoidal rule)
  - `∫ r_sulf dt` → Toplam üretilen sülfür

**Satır 148-150: Expected Total**
```matlab
S_total0 = y_sim(1,12)*env.Vl + y_sim(1,4) + y_sim(1,8)*env.Vl;
S_total_expected = S_total0 + S_prod_cum;
```

**Satır 152-163: Plot**
```matlab
figure;
subplot(2,1,1);
plot(t_sim, S_total_model, 'b-', t_sim, S_total_expected, 'r--');
legend('Model', 'Expected');

subplot(2,1,2);
plot(t_sim, S_total_model - S_total_expected, 'k-');
title('Mass-balance error');
```

**Neden önemli?**
- Eğer mavi ve kırmızı çakışırsa → Kütle korunuyor ✅
- Error plot'ta ~0 olmalı

---

### Satır 166-181: H2S Headspace Diagnostic

```matlab
phi_H2S_fit = p_fit(27);
Hcp_H2S_eff = phi_H2S_fit * env.Hcp_H2S_eff;
nH2S_g_eq = (H2S_aq .* env.Vg) ./ (Hcp_H2S_eff * env.Rgas * env.T) * 1000;

figure;
plot(t_sim, y_sim(:,4), 'b-', t_sim, nH2S_g_eq, 'r--');
legend('nH2S_g (model)', 'nH2S_g^{eq} from H2S(aq)');
```

**Ne yapıyor?**
- Henry yasası ile dengede olması gereken nH2S_g hesapla
- Model ile karşılaştır
- Eğer ayrılırsa → kla_H2S veya phi_H2S yanlış

---

### Satır 183-189: Reaction Rates for Plotting

```matlab
rates = zeros(length(t_sim), 4);
for i = 1:length(t_sim)
    rates(i,:) = rate_out_mixed(t_sim(i), y_sim(i,:), p_fit, env);
end
```

**Ne yapıyor?**
- ODE içinde rates hesaplanıyor ama return edilmiyor
- Plot için tekrar hesaplıyoruz

---

### Satır 191-205: Write .dat File

```matlab
fileID = fopen('Sandstone_25C_inc_rates.dat','w');
fprintf(fileID, 'Time(days) nH2_g nCO2_g ... r_meth r_sulf r_prec r_aceto\n');

for i = 1:length(t_sim)
    fprintf(fileID, '%10.6f %12.6g %12.6g ... %12.6g\n', ...
        t_sim(i), y_sim(i,1), y_sim(i,2), ..., rates(i,1), rates(i,2), ...);
end
fclose(fileID);
```

**Ne yapıyor?**
- ASCII text file yaz
- Space-separated columns
- Her satır: 1 timestep (500 satır)

---

### Satır 207-224: Figure 1 - States Plot

```matlab
species = {'nH2_g','nCO2_g','nCH4_g','nH2S_g','H2(aq)', ...};
figure('Name','Gases & Aqueous - Sandstone (25 °C)');
for i = 1:length(species)
    subplot(7,2,i)
    if i <= 4
        plot(t_exp, data_exp(:,i), 'ko'); hold on;  % Experimental
        plot(t_sim, y_sim(:,i), 'b-');              % Model
    elseif i == 7
        plot(t_exp, data_exp(:,5), 'ko'); hold on;
        plot(t_sim, y_sim(:,7), 'b-');
    else
        plot(t_sim, y_sim(:,i), 'b-');
    end
    title(species{i}); legend;
end
```

**Ne yapıyor?**
- 14 subplot (7×2 grid)
- İlk 4 (gazlar) + SO4 → Experimental data ile karşılaştır
- Diğerleri → Sadece model

---

### Satır 226-235: Figure 2 - Sulfide & pH

```matlab
figure;
subplot(3,1,1); plot(t_sim, H2S_aq, 'b-'); title('H2S(aq)');
subplot(3,1,2); plot(t_sim, HS_aq,  'b-'); title('HS^-');
subplot(3,1,3); plot(t_sim, env.pH_fun(t_sim), 'k-'); title('pH');
```

---

### Satır 237-245: Figure 3 - Rates

```matlab
figure;
plot(t_sim, rates(:,1), 'r-', 'DisplayName','r_{meth}'); hold on;
plot(t_sim, rates(:,2), 'b-', 'DisplayName','r_{sulf}');
plot(t_sim, rates(:,3), 'g-', 'DisplayName','r_{precip}');
plot(t_sim, rates(:,4), 'm-', 'DisplayName','r_{aceto}');
legend;
```

---

### Satır 247-251: RMSE Calculation

```matlab
yH2_on_exp  = interp1(t_sim, y_sim(:,1), t_exp, 'linear', 'extrap');
yCO2_on_exp = interp1(t_sim, y_sim(:,2), t_exp, 'linear', 'extrap');
fprintf('RMSE: H2=%.4f mmol CO2=%.4f mmol\n', ...
        rmse_equal(yH2_on_exp, data_exp(:,1)), rmse_equal(yCO2_on_exp, data_exp(:,2)));
```

**Ne yapıyor?**
- Model output (dense grid) → Experimental grid'e interpolate et
- RMSE hesapla

**Helper function (satır 420-422):**
```matlab
function r = rmse_equal(a,b)
r = sqrt(mean((a(:)-b(:)).^2,'omitnan'));
end
```

---

## Helper Functions

### 1. rate_out_mixed (Satır 394-415)

```matlab
function dr = rate_out_mixed(t, y, p, env)
% ODE içindeki rate hesaplamalarını tekrar yap
% Input: t, y, p, env
% Output: [r_meth, r_sulf, r_prec, r_aceto]
```

**Neden ayrı?**
- ODE içinde rates hesaplanıyor ama dışarı verilmiyor
- Post-processing için lazım

### 2. speciate_sulfide (Satır 417-422)

```matlab
function [H2S_aq, HS_aq] = speciate_sulfide(S_tot, pH, pKa)
frac_HS  = 1 ./ (1 + 10.^(pKa - pH));
HS_aq    = S_tot .* frac_HS;
H2S_aq   = S_tot - HS_aq;
end
```

**Element-wise operations:**
- `./ , .^, .*` → Vector'e uygula

---

## Özet: Simulation & Post-Processing

```
1. SIMULATION:
   - ode15s ile dense grid çöz (p_fit kullan)
   - t_sim, y_sim → 500×14

2. DIAGNOSTICS:
   - Sulfur mass balance check
   - H2S headspace diagnostic
   - Speciation (H2S_aq, HS_aq)

3. RATES:
   - Her timestep için reaction rates hesapla

4. OUTPUT:
   - .dat file yaz (ASCII)

5. PLOTS:
   - Figure 1: 14 state vs time
   - Figure 2: Sulfide & pH
   - Figure 3: Reaction rates
   - Mass balance figures

6. RMSE:
   - Model → Experimental grid interpolate
   - RMSE hesapla ve yazdır
```

---

**Hazırlayan**: Hasan Arı
**Tarih**: 6 Ocak 2026
**Amaç**: Simulation ve post-processing kodunu anlamak
