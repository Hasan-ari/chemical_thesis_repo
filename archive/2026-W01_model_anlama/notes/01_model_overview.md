# v4 İki Fazlı Model - Genel Bakış

**Dosya**: `anaerobic_model_two_phase_mixedSR_25C_v4.m`  
**Kayaç**: Sandstone  
**Sıcaklık**: 25°C (298.15 K)  
**Tarih**: 6 Ocak 2026

---

## Model Hakkında

Bu MATLAB kodu, Muller 2024 deneysel verileri için **iki fazlı (gas + liquid) biyokimyasal reaktif transport** modelini çözer.

### Ana Özellikler

✅ **14 state variables** (v3'te 10'du, v4'te 14)  
✅ **28 parameters** (v3'te 13'tü, v4'te 28)  
✅ **İki fazlı sistem**: Gaz (headspace) + Sıvı (aqueous)  
✅ **Henry yasası**: Gaz-sıvı dengesi  
✅ **pH-dependent sulfur speciation**: H2S ↔ HS⁻  
✅ **Fe pool tracking**: FeS çökelmesi için sınırlı Fe(II)  
✅ **Gypsum dissolution**: Kayaç özel kimya  

---

## Fiziksel Sistem

### Deneysel Düzenek

```
┌─────────────────────────┐
│   HEADSPACE (Vg)        │  ← 140 mL gaz fazı
│   H₂, CO₂, CH₄, H₂S     │
│   (basınç: ~2 bar)      │
├─────────────────────────┤
│   LIQUID (Vl)           │  ← 15 mL sıvı fazı
│   H₂(aq), CO₂(aq), etc. │
│   Bacteria, Minerals    │
└─────────────────────────┘
        ↕ Henry's Law
```

**Hacimler**:
- Headspace: `Vg = 0.14 L` (140 mL)
- Liquid: `Vl = 0.015 L` (15 mL)
- Toplam: ~155 mL

**Sıcaklık**:
- `T = 298.15 K` (25°C)
- Tüm Henry sabitleri 25°C için

---

## Model v3 → v4 Evrilişi

### v3 (ESKİ - YANLIŞ FİZİK)

❌ Tek fazlı sistem (sadece sıvı)  
❌ Tüm gazların çözündüğü varsayımı  
❌ Aşırı yüksek başlangıç konsantrasyonları  
❌ Çok hızlı reaksiyon hızları  
❌ İlk 1-2 günde ani düşüşler  

### v4 (YENİ - DOĞRU FİZİK)

✅ İki fazlı sistem (gaz + sıvı)  
✅ Henry yasası ile gaz-sıvı dengesi  
✅ Realistik konsantrasyonlar  
✅ Fiziksel kütle transferi (`kLa`)  
✅ Doğru reaksiyon hızları  

---

## Kritik Yenilikler (v4)

### 1. Fe Pool Tracking

```matlab
% State 14: Fe_pool (mmol/L)
Fe_pool0 = 0.10;  % Initial dissolved Fe(II)

% FeS precipitation rate limited by available Fe
r_prec_raw = k_prec * max(0, HS_aq - HS_sat);
r_prec     = min(r_prec_raw, Fe_pool);

% Fe pool depletes as FeS forms
dFe_pool = - r_prec;
```

**Fiziksel Anlam**:
- Çözünmüş Fe(II) sınırlı (0.05-0.5 mM)
- FeS çökmesi için Fe gerekli (1:1 stoichiometry)
- Fe tükenince çökelme durur

### 2. H2S Degassing Flux (Outgassing-Positive)

```matlab
% H2S: outgassing-positive flux
Jout_H2S = kla_H2S * (H2S_aq - Ceq_H2S);

% Gas balance for H2S (mmol/day)
dnH2S_g = + Jout_H2S * Vl;  % Degassing from liquid
```

**Neden Önemli?**:
- H2S gazlaşması kritik (toksik gaz!)
- `kla_H2S` yüksek (25 day⁻¹) → Hızlı transfer
- `alpha_H2S` faktörü film direncini temsil eder

### 3. Gypsum Buffering (Kayaç Özel)

```matlab
% Gypsum özel: CaSO₄·2H₂O dissolution
env.SO4_sat_gyp = 15.0;  % mM (Sandstone için)
k_diss_gyp      = 0.12;  % 1/day

% Controlled SO4 release
r_diss_gyp = k_diss_gyp * max(0, SO4_sat - SO4);
```

**Kayaç Farklılıkları**:
| Kayaç | SO4_sat (mM) | Neden? |
|-------|--------------|--------|
| Sandstone | 15.0 | Orta çözünme |
| Basalt | 15.0 | Az çözünme |
| Calcite | 15.0 | Orta çözünme |
| Gypsum | 36.0 | Çok yüksek! (CaSO₄ dissolution) |

---

## Kod Yapısı (Akış Şeması)

```
1. EXPERIMENT SETTINGS
   ├─ Volumes (Vg, Vl)
   ├─ Temperature (T)
   └─ Henry constants (Hcp_H2, Hcp_CO2, Hcp_H2S)

2. LOAD EXPERIMENTAL DATA
   ├─ Read Muller_2024_H2_Sandstone_at_25C.txt
   ├─ Extract: t_exp, nH2_g, nCO2_g, nCH4_g, nH2S_g, pH, SO4
   └─ Create pH interpolant: pH_fun(t)

3. INITIAL CONDITIONS
   ├─ Partial pressures from gas moles
   ├─ Henry equilibrium: H2_aq0, CO2_aq0
   └─ 14 state vector: y0 = [nH2_g, ..., Fe_pool]

4. PARAMETER SETUP
   ├─ p0 = [28 parameters] initial guess
   ├─ lb, ub = bounds
   └─ env struct (Vg, Vl, T, Henry, pH_fun, pKa)

5. FITTING (lsqnonlin)
   ├─ Objective: residuals_full()
   ├─ Solve ODE at t_exp points
   ├─ Compare to data: nH2_g, nCO2_g, nCH4_g, nH2S_g, SO4
   └─ Output: p_fit (best parameters)

6. FINAL SIMULATION (dense grid)
   ├─ ode15s with fitted p_fit
   ├─ NonNegative constraints (1:14 states)
   └─ Output: t_sim, y_sim

7. POST-PROCESSING
   ├─ Sulfide speciation (H2S_aq, HS_aq)
   ├─ Reaction rates (r_meth, r_sulf, r_prec, r_aceto)
   ├─ Mass balance diagnostics
   └─ Write .dat file + plots

8. DIAGNOSTIC PLOTS
   ├─ Sulfur mass balance
   ├─ H2S headspace diagnostic
   ├─ Gases & aqueous species
   ├─ Sulfide speciation & pH
   └─ Kinetic rates
```

---

## Önemli Fonksiyonlar

### Ana Fonksiyon
```matlab
function anaerobic_model_two_phase_mixedSR_25C_v4
    % Entry point
    % Loads data → Fits parameters → Simulates → Plots
end
```

### ODE Model
```matlab
function dydt = model_mixed(t, y, p, env)
    % Computes derivatives for 14 states
    % Called by ode15s at each timestep
    % Returns: dydt = [dnH2_g, dnCO2_g, ..., dFe_pool]
end
```

### Residuals (Fitting)
```matlab
function res = residuals_full(p, t_exp, data_exp, y0, env)
    % Solves ODE at t_exp
    % Computes log1p residuals
    % Weighted: [1, 1, 0.9, 1.0, 2.0] for [H2, CO2, CH4, H2S, SO4]
end
```

### Rate Output (Diagnostics)
```matlab
function dr = rate_out_mixed(t, y, p, env)
    % Computes reaction rates for plotting
    % Returns: [r_meth, r_sulf, r_prec, r_aceto]
end
```

### Speciation Helper
```matlab
function [H2S_aq, HS_aq] = speciate_sulfide(S_tot, pH, pKa)
    % pH-dependent sulfur speciation
    % H2S ⇌ HS⁻ + H⁺
    % pKa = 7.05 at 25°C
end
```

---

## Önemli Kavramlar

### 1. Stiff System (Katı Sistem)
- v4 modeli **stiff ODE** içeriyor
- Hızlı + yavaş dinamikler bir arada
- `ode15s` kullanılmalı (implicit solver)
- `ode45` **BAŞARISIZ OLUR**

### 2. NonNegative Constraints
```matlab
opts = odeset('NonNegative',1:14, ...);
```
- Fiziksel olarak: Konsantrasyonlar < 0 olamaz
- Numerik olarak: Stability için kritik

### 3. Mass Balance Validation
```matlab
S_total_model = S_gas_mmol + S_aq_mmol + S_FeS_mmol;
S_total_expected = S_total0 + S_prod_cum;

mb_err = S_total_model(end) - S_total_expected(end);
% mb_err < 1e-5 → Model doğru
% mb_err > 1e-3 → Model yanlış
```

---

## Sonraki Adımlar

Bu overview'ı okuduktan sonra:

1. ✅ [02_state_variables.md](02_state_variables.md) → 14 state tanımları
2. ✅ [03_parameters.md](03_parameters.md) → 28 parametre detayları
3. ✅ [04_henry_law.md](04_henry_law.md) → Gaz-sıvı dengesi
4. ✅ [05_pH_speciation.md](05_pH_speciation.md) → Sülfür türlenmesi
5. ✅ [06_ode_system.md](06_ode_system.md) → ODE differansiyel denklemleri

---

**Hazırlayan**: Hasan Arı  
**Tarih**: 6 Ocak 2026  
**Kaynak**: `CURRENT/code/v4_two_phase/sandstone_25C/`
