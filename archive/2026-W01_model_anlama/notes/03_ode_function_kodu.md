# ODE Function (model_mixed) - Satır Satır Kod Açıklaması

**Bölüm**: Satır 278-392 (model_mixed fonksiyonu)
**Amaç**: 14 state için derivatives hesaplamak

---

## Fonksiyon Signature (Satır 278)

```matlab
function dydt = model_mixed(t, y, p, env)
```

**Input:**
- `t`: Scalar time (örnek: 3.5 gün)
- `y`: 14×1 state vector
- `p`: 28×1 parameter vector
- `env`: Struct (sabitler)

**Output:**
- `dydt`: 14×1 derivatives vector

**Çağrılan yer:**
```matlab
odes = @(t,y) model_mixed(t,y,p_fit,env);
[t, y] = ode15s(odes, tspan, y0, opts);
```

---

## Bölüm 1: Unpack (Satır 279-301)

### Satır 279-283: env Unpack
```matlab
Vg=env.Vg; Vl=env.Vl; T=env.T; Rgas=env.Rgas;
Hcp_H2=env.Hcp_H2_eff; Hcp_CO2=env.Hcp_CO2_eff; Hcp_H2S=env.Hcp_H2S_eff;
pH=env.pH_fun(t); pKa=env.pKa_H2S;
```

**Ne yapıyor?**
- Struct field'ları → Local variable'lara kopyala
- `pH=env.pH_fun(t)` → `t` anındaki pH değerini hesapla

### Satır 285-289: y (State) Unpack
```matlab
nH2_g=y(1); nCO2_g=y(2); nCH4_g=y(3); nH2S_g=y(4);
H2_aq=y(5); CO2_aq=y(6); SO4=y(7); FeS=y(8); X=y(9);
Ac=y(10); HCO3=y(11); S_tot=y(12); Lag=y(13);
Fe_pool = y(14);
```

**Ne yapıyor?**
- 14 elemanlı `y` vector → Isimlendirilmiş değişkenler
- `y(1)` yerine `nH2_g` kullanmak daha okunabilir

### Satır 291-301: p (Parameters) Unpack
```matlab
k_m  = p(1);  k_s  = p(2);  k_a  = p(3);
Y_m  = p(4);  Y_s  = p(5);  Y_a  = p(6);
KI_m = p(7);  KI_s = p(8);  KI_a = p(9);
k_prec = p(10); HS_sat = p(11); H2_th = p(12); DG_th = p(13);
K_H2 = p(14); K_SO4 = p(15); K_CO2 = p(16);
kla_H2 = p(17); kla_CO2 = p(18); kla_H2S = p(19);
b = p(20); t_lag = p(21); w_lag = p(22);
k_diss_gyp = p(23); beta_SO4_m = p(24);
```

**Ne yapıyor?**
- 28 elemanlı `p` vector → Isimlendirilmiş parametreler
- `p(1)` yerine `k_m` kullanmak okunabilir

---

## Bölüm 2: Guards (Satır 318-323)

```matlab
eps=1e-12;
nH2_g=max(nH2_g,eps); nCO2_g=max(nCO2_g,eps); nCH4_g=max(nCH4_g,eps); nH2S_g=max(nH2S_g,eps);
H2_aq=max(H2_aq,eps); CO2_aq=max(CO2_aq,eps); SO4=max(SO4,eps); S_tot=max(S_tot,eps);
Ac=max(Ac,eps); HCO3=max(HCO3,eps); X=max(X,eps); Fe_pool=max(Fe_pool,0);
```

**Ne yapıyor?**
- `eps = 1e-12` → Çok küçük pozitif sayı
- `nH2_g = max(nH2_g, eps)` → Eğer `nH2_g < eps` → `eps` yap

**Neden?**
- Division by zero önlemek
- Log hesaplamalarında sorun olmasın
- Fiziksel: Konsantrasyon < 0 olamaz

---

## Bölüm 3: Partial Pressures (Satır 325-329)

```matlab
pH2  = (nH2_g /1000)  * Rgas * T / Vg;
pCO2 = (nCO2_g/1000)  * Rgas * T / Vg;
pH2S = (nH2S_g/1000)  * Rgas * T / Vg;
```

**Satır satır:**
- **325**: `(nH2_g/1000)` → mmol → mol
  - `* Rgas * T / Vg` → `p = n*R*T/V` (ideal gaz)
  - Sonuç: `pH2` (atm)
- **326-327**: Aynı şekilde CO2, H2S

**Örnek:**
```matlab
nH2_g = 150 mmol → 0.15 mol
Rgas = 0.082 L·atm/(mol·K)
T = 298.15 K
Vg = 0.14 L
pH2 = 0.15 * 0.082 * 298.15 / 0.14 = 26.2 atm ❌ (çok yüksek!)
```

*Not: Gerçek değerler daha düşük olacak*

---

## Bölüm 4: Henry Equilibrium (Satır 331-335)

```matlab
Ceq_H2  = Hcp_H2  * pH2;
Ceq_CO2 = Hcp_CO2 * pCO2;
Ceq_H2S = Hcp_H2S * pH2S;
```

**Ne yapıyor?**
- Henry yasası: `C_eq = Hcp × p`
- `Hcp_H2 = 0.78 mmol/L/atm`
- `pH2 = 1.5 atm` (örnek)
- `Ceq_H2 = 0.78 × 1.5 = 1.17 mmol/L`

**Fiziksel anlam:**
- Gaz fazındaki `pH2` basıncında
- Sıvıda dengede olması gereken konsantrasyon

---

## Bölüm 5: Mass Transfer (Satır 337-343)

```matlab
J_H2  = kla_H2  * (Ceq_H2  - H2_aq);
J_CO2 = kla_CO2 * (Ceq_CO2 - CO2_aq);

% ... sulfide speciation (satır 345-351) ...

Jout_H2S = kla_H2S * (H2S_aq - Ceq_H2S);
```

**Satır 337-338: H2, CO2 Dissolution**
```matlab
J_H2 = kla_H2 * (Ceq_H2 - H2_aq);
```
- `kla_H2` → Mass transfer coefficient (1/day)
- `Ceq_H2 - H2_aq` → Driving force
- Eğer `H2_aq < Ceq_H2` → `J_H2 > 0` → Gaz çözünüyor
- Eğer `H2_aq > Ceq_H2` → `J_H2 < 0` → Gaz uçuyor

**Satır 343: H2S Degassing**
```matlab
Jout_H2S = kla_H2S * (H2S_aq - Ceq_H2S);
```
- **Dikkat**: `H2S_aq - Ceq_H2S` (ters yön!)
- H2S moleküler formda gazlaşıyor
- Pozitif flux → Degassing

---

## Bölüm 6: Sulfide Speciation (Satır 345-351)

```matlab
frac_HS  = 1/(1+10^(pKa - pH));
frac_H2S = 1 - frac_HS;
HS_aq  = S_tot*frac_HS;
H2S_aq = S_tot*frac_H2S;
```

**Satır satır:**

### Henderson-Hasselbalch Equation
```
H2S ⇌ HS⁻ + H⁺
pKa = 7.05 @ 25°C
```

**Satır 345-346: Fraction Calculation**
```matlab
frac_HS = 1 / (1 + 10^(pKa - pH));
```
- `pKa - pH` → 7.05 - 7.2 = -0.15 (örnek)
- `10^(-0.15)` → 0.708
- `frac_HS = 1 / (1 + 0.708) = 0.586` → %58.6 HS⁻

**Satır 347-348: Apply Fractions**
```matlab
HS_aq = S_tot * frac_HS;
H2S_aq = S_tot * frac_H2S;
```
- `S_tot = 5 mmol/L` (örnek)
- `HS_aq = 5 × 0.586 = 2.93 mmol/L`
- `H2S_aq = 5 × 0.414 = 2.07 mmol/L`

---

## Bölüm 7: Inhibitions & Gates (Satır 353-367)

### Satır 353-355: Sulfide Inhibition
```matlab
f_inh_m = KI_m/(KI_m+HS_aq);
f_inh_s = KI_s/(KI_s+HS_aq);
f_inh_a = KI_a/(KI_a+HS_aq);
```

**Monod-like Inhibition:**
- `KI_m = 0.2 mmol/L` (örnek)
- `HS_aq = 0.1 mmol/L`
- `f_inh_m = 0.2 / (0.2 + 0.1) = 0.667` → %66.7 aktif

### Satır 356-358: Activation Gates
```matlab
f_H2    = H2_aq/(H2_aq+H2_th);
f_lag   = 1/(1+exp((t_lag - t)/max(w_lag,1e-3)));
f_act   = f_H2 * f_lag;
```

**Satır 356: H2 Threshold**
```matlab
f_H2 = H2_aq / (H2_aq + H2_th);
```
- `H2_aq = 1.0 mmol/L`, `H2_th = 0.02 mmol/L`
- `f_H2 = 1.0 / (1.0 + 0.02) = 0.980` → Yeterli H2 var

**Satır 357: Lag Sigmoid**
```matlab
f_lag = 1 / (1 + exp((t_lag - t) / w_lag));
```
- `t = 5 gün`, `t_lag = 3 gün`, `w_lag = 0.7 gün`
- `exp((3-5)/0.7) = exp(-2.86) = 0.057`
- `f_lag = 1 / (1 + 0.057) = 0.946` → Aktive olmuş

---

## Bölüm 8: Monod Kinetics (Satır 360-364)

```matlab
mH2  = H2_aq /(K_H2  + H2_aq);
mSO4 = SO4   /(K_SO4 + SO4);
mCO2 = CO2_aq/(K_CO2 + CO2_aq);
```

**Monod Function:**
```
m = S / (K + S)
```

**Satır 360: H2 Monod**
```matlab
mH2 = H2_aq / (K_H2 + H2_aq);
```
- `H2_aq = 1.0 mmol/L`, `K_H2 = 0.5 mmol/L`
- `mH2 = 1.0 / (0.5 + 1.0) = 0.667` → %66.7 saturated

---

## Bölüm 9: Reaction Rates (Satır 379-387)

### Satır 383: Methanogenesis
```matlab
r_meth  = k_m * X * mH2 * mCO2       * f_inh_m * f_act * fT_m * f_comp_m;
```

**Çarpanlar:**
- `k_m` → Max rate (0.06 day⁻¹)
- `X` → Biomass (0.05 mmol/L)
- `mH2` → H2 saturation (0.667)
- `mCO2` → CO2 saturation (0.800)
- `f_inh_m` → Sulfide inhibition (0.667)
- `f_act` → Activation gate (0.946)
- `fT_m` → Thermodynamic gate (~1.0)
- `f_comp_m` → SO4 competition (0.500)

**Hesaplama:**
```
r_meth = 0.06 × 0.05 × 0.667 × 0.8 × 0.667 × 0.946 × 1.0 × 0.5
       = 0.00063 mmol/L/day
```

### Satır 384-385: Sulfate Reduction & Homoacetogenesis
```matlab
r_sulf  = k_s * X * mH2 * mSO4       * f_inh_s * f_act * fT_s;
r_aceto = k_a * X * mH2 * (mCO2.^2)  * f_inh_a * f_act * fT_a;
```

**Dikkat:**
- `r_sulf` → CO2 gate YOK! (SR için CO2 gerekmez)
- `r_aceto` → `(mCO2.^2)` → CO2 bağımlılığı daha yüksek

---

## Bölüm 10: Precipitation (Satır 387-391)

```matlab
r_prec_raw = k_prec * max(0, HS_aq - HS_sat);
r_prec     = min(r_prec_raw, Fe_pool);
```

**Satır 387: Raw Precipitation Rate**
```matlab
r_prec_raw = k_prec * max(0, HS_aq - HS_sat);
```
- `k_prec = 0.02 day⁻¹`
- `HS_aq = 2.0 mmol/L`, `HS_sat = 0.1 mmol/L`
- `HS_aq - HS_sat = 1.9 mmol/L` (oversaturation)
- `r_prec_raw = 0.02 × 1.9 = 0.038 mmol/L/day`

**Satır 388: Fe Limitation**
```matlab
r_prec = min(r_prec_raw, Fe_pool);
```
- `Fe_pool = 0.05 mmol/L`
- `r_prec = min(0.038, 0.05) = 0.038 mmol/L/day`
- Eğer `Fe_pool < r_prec_raw` → Fe sınırlar

---

## Bölüm 11: Gypsum Dissolution (Satır 393-395)

```matlab
SO4_sat    = env.SO4_sat_gyp;
r_diss_gyp = k_diss_gyp * max(0, SO4_sat - SO4);
```

**Satır 394: Dissolution Rate**
```matlab
r_diss_gyp = k_diss_gyp * max(0, SO4_sat - SO4);
```
- `k_diss_gyp = 0.12 day⁻¹`
- `SO4_sat = 15.0 mmol/L` (Sandstone)
- `SO4 = 10.0 mmol/L` (current)
- `r_diss_gyp = 0.12 × (15 - 10) = 0.6 mmol/L/day`

**Fiziksel anlam:**
- SO4 < SO4_sat → Gypsum çözünüyor
- SO4 ≥ SO4_sat → Çözünme durdu

---

## Bölüm 12: Derivatives (Satır 397-415)

### Gas Balances (mmol/day)
```matlab
dnH2_g  = - J_H2  * Vl;
dnCO2_g = - J_CO2 * Vl;
dnCH4_g = + r_meth * Vl;
dnH2S_g = + Jout_H2S * Vl;
```

**Satır 398: H2 Gas**
```matlab
dnH2_g = - J_H2 * Vl;
```
- `J_H2` → mmol/L/day (sıvı bazında)
- `* Vl` → mmol/day (toplam)
- `-` → Gaz çözünüyor (azalıyor)

### Liquid Balances (mmol/L/day)
```matlab
dH2_aq  = + J_H2  - 4*r_meth - 4*r_sulf - 4*r_aceto;
dCO2_aq = + J_CO2 - 1*r_meth - 2*r_aceto;
dSO4    = - 1*r_sulf + r_diss_gyp;
dFeS    = + r_prec;
dX      = + Y_m*r_meth + Y_s*r_sulf + Y_a*r_aceto - b*X;
dAc     = + r_aceto;
dHCO3   = 0;
dS_tot  = + 1.00*r_sulf - r_prec - Jout_H2S;
dLag    = (f_lag - Lag)/max(w_lag,1e-3);
dFe_pool = - r_prec;
```

**Satır 406: H2(aq) Balance**
```matlab
dH2_aq = + J_H2 - 4*r_meth - 4*r_sulf - 4*r_aceto;
```
- `+J_H2` → Gazdan geliyor
- `-4*r_meth` → Methanogenesis tüketiyor (4 H2)
- `-4*r_sulf` → SR tüketiyor (4 H2)
- `-4*r_aceto` → Homoacetogenesis tüketiyor (4 H2)

### Final Assembly (Satır 418)
```matlab
dydt = [dnH2_g; dnCO2_g; dnCH4_g; dnH2S_g; dH2_aq; dCO2_aq; dSO4; dFeS; dX; dAc; dHCO3; dS_tot; dLag; dFe_pool];
```

**14×1 column vector:**
```
dydt = [
  dnH2_g,      % 1
  dnCO2_g,     % 2
  dnCH4_g,     % 3
  dnH2S_g,     % 4
  dH2_aq,      % 5
  dCO2_aq,     % 6
  dSO4,        % 7
  dFeS,        % 8
  dX,          % 9
  dAc,         % 10
  dHCO3,       % 11
  dS_tot,      % 12
  dLag,        % 13
  dFe_pool     % 14
]
```

---

## Özet: ODE Function Akışı

```
1. Unpack: env, y, p → Named variables
2. Guards: Ensure all states ≥ eps (no division by zero)
3. Partial pressures: p = n*R*T/V (ideal gas)
4. Henry equilibrium: C_eq = Hcp * p
5. Mass transfer: J = kla * (C_eq - C_aq)
6. Sulfide speciation: S_tot → H2S_aq + HS_aq (pH dependent)
7. Inhibitions: f_inh = KI / (KI + HS)
8. Gates: f_H2, f_lag, f_act
9. Monod: m = S / (K + S)
10. Thermodynamics: fT = sigmoid(ΔG)
11. Rates: r = k * X * m * f_inh * f_act * ...
12. Precipitation: r_prec limited by Fe_pool
13. Gypsum dissolution: r_diss_gyp
14. Derivatives: Sources - Sinks
15. Return dydt (14×1 vector)
```

**ODE Solver kullanımı:**
- `ode15s` her timestep'te `model_mixed(t, y, p, env)` çağırır
- `dydt` alır, `y` günceller: `y_new = y + dydt*dt`

---

**Hazırlayan**: Hasan Arı
**Tarih**: 6 Ocak 2026
**Amaç**: ODE function kodunu satır satır anlamak
