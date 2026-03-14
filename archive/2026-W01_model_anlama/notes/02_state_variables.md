# 14 State Variables - Detaylı Açıklama

**Dosya**: `anaerobic_model_two_phase_mixedSR_25C_v4.m`  
**Tarih**: 6 Ocak 2026

---

## State Vector Tanımı

```matlab
% y = [nH2_g, nCO2_g, nCH4_g, nH2S_g, 
%      H2_aq, CO2_aq, SO4, FeS, X, Acetate, 
%      HCO3, S_tot, Lag, Fe_pool]
```

**Toplam**: 14 state variables  
**v3'ten fark**: Fe_pool, HCO3, S_tot, Lag eklendi

---

## Gaz Fazı (Headspace) - 4 State

### 1. nH2_g - Hidrojen Gazı (mmol)

```matlab
% State 1: nH2_g (mmol in headspace)
% Initial: nH2_g_exp(1) ~ 100-200 mmol
```

**Fiziksel Anlam**:
- Headspace'deki H₂ mol sayısı
- Birim: **mmol** (gaz fazında mol, sıvıda mmol/L)
- Bakteriler tarafından tüketilir

**Reaksiyonlarda Rolü**:
- **Methanogenesis**: 4H₂ + CO₂ → CH₄ + 2H₂O (tüketilir)
- **Sulfate reduction**: 4H₂ + SO₄²⁻ → HS⁻ + 3H₂O + OH⁻ (tüketilir)
- **Homoacetogenesis**: 4H₂ + 2CO₂ → CH₃COO⁻ + H⁺ + 2H₂O (tüketilir)

**Differential Equation**:
```matlab
% Gas balance (mmol/day)
dnH2_g = - J_H2 * Vl;

% J_H2 = kla_H2 * (Ceq_H2 - H2_aq)  [mmol/L/day]
% If H2_aq < Ceq_H2 → J_H2 > 0 → gas dissolves → dnH2_g < 0
```

**Tipik Değer Aralığı**:
- Başlangıç: 100-200 mmol
- Son: 0-20 mmol (çoğu tüketilmiş)

---

### 2. nCO2_g - Karbondioksit Gazı (mmol)

```matlab
% State 2: nCO2_g (mmol in headspace)
% Initial: nCO2_g_exp(1) ~ 50-100 mmol
```

**Fiziksel Anlam**:
- Headspace'deki CO₂ mol sayısı
- Bakteriler tarafından tüketilir **VE** methanogenesis'de üretilir

**Reaksiyonlarda Rolü**:
- **Methanogenesis**: 4H₂ + CO₂ → CH₄ + 2H₂O (tüketilir)
- **Homoacetogenesis**: 4H₂ + 2CO₂ → CH₃COO⁻ + H⁺ + 2H₂O (tüketilir)

**Differential Equation**:
```matlab
% Gas balance (mmol/day)
dnCO2_g = - J_CO2 * Vl;

% J_CO2 = kla_CO2 * (Ceq_CO2 - CO2_aq)  [mmol/L/day]
```

**Tipik Değer Aralığı**:
- Başlangıç: 50-100 mmol
- Son: 10-50 mmol

---

### 3. nCH4_g - Metan Gazı (mmol)

```matlab
% State 3: nCH4_g (mmol in headspace)
% Initial: nCH4_g_exp(1) ~ 0-5 mmol (seed)
```

**Fiziksel Anlam**:
- Headspace'deki CH₄ mol sayısı
- **Sadece methanogenesis tarafından üretilir**

**Reaksiyonlarda Rolü**:
- **Methanogenesis**: 4H₂ + CO₂ → CH₄ + 2H₂O (üretilir)

**Differential Equation**:
```matlab
% Gas balance (mmol/day)
dnCH4_g = + r_meth * Vl;

% CH4 directly enters gas phase
% No liquid-phase CH4 tracked
```

**Tipik Değer Aralığı**:
- Başlangıç: 0-5 mmol
- Son: 50-150 mmol (kümülatif üretim)

**Neden Önemli?**:
- CH₄ üretimi → Methanogen aktivitesinin direkt göstergesi
- Deneysel veri ile fit kalitesini belirler

---

### 4. nH2S_g - Hidrojen Sülfür Gazı (mmol)

```matlab
% State 4: nH2S_g (mmol in headspace)
% Initial: nH2S_g_exp(1) ~ 0-2 mmol (seed)
```

**Fiziksel Anlam**:
- Headspace'deki H₂S mol sayısı
- **Toksik gaz** - yumurta kokusu
- Sülfat indirgeme ürünü

**Reaksiyonlarda Rolü**:
- **Sulfate reduction**: 4H₂ + SO₄²⁻ → HS⁻ → H₂S(aq) → H₂S(g) (üretilir)

**Differential Equation**:
```matlab
% Gas balance (mmol/day)
dnH2S_g = + Jout_H2S * Vl;

% Jout_H2S = kla_H2S * (H2S_aq - Ceq_H2S)  [mmol/L/day]
% Outgassing-positive flux (H2S escapes from liquid)
```

**Tipik Değer Aralığı**:
- Başlangıç: 0-2 mmol
- Son: 5-50 mmol (kayaç ve sıcaklığa göre değişir)

**Kritik Not**:
- `kla_H2S` = 25 day⁻¹ (yüksek!) → Hızlı degassing
- `alpha_H2S` = 0.7-3.0 → Film direnci faktörü
- H2S gazlaşması fit kalitesini çok etkiler!

---

## Sıvı Fazı (Aqueous) - 10 State

### 5. H2_aq - Çözünmüş Hidrojen (mmol/L)

```matlab
% State 5: H2_aq (mmol/L dissolved in liquid)
% Initial: H2_aq0 = Hcp_H2_eff * pH2 ~ 0.5-2 mmol/L
```

**Fiziksel Anlam**:
- Sıvıdaki çözünmüş H₂ konsantrasyonu
- Henry yasası ile gaz fazı ile dengede
- Bakterilerin kullandığı asıl form

**Henry Dengesi**:
```matlab
% Partial pressure (atm)
pH2 = (nH2_g/1000) * R_gas * T / Vg;

% Equilibrium concentration (mmol/L)
Ceq_H2 = Hcp_H2_eff * pH2;

% Transfer flux (mmol/L/day)
J_H2 = kla_H2 * (Ceq_H2 - H2_aq);
```

**Differential Equation**:
```matlab
% Liquid balance (mmol/L/day)
dH2_aq = + J_H2 - 4*r_meth - 4*r_sulf - 4*r_aceto;

% Source: Gas dissolution (J_H2)
% Sinks: All 3 pathways consume 4 H2 each
```

**Tipik Değer Aralığı**:
- Başlangıç: 0.5-2 mmol/L
- Son: 0.01-0.1 mmol/L (tükenme)

**Monod Kinetics**:
```matlab
mH2 = H2_aq / (K_H2 + H2_aq);
% K_H2 ~ 0.5 mmol/L → Half-saturation constant
% H2_aq >> K_H2 → mH2 ≈ 1 (saturated)
% H2_aq << K_H2 → mH2 ≈ H2_aq/K_H2 (first-order)
```

---

### 6. CO2_aq - Çözünmüş Karbondioksit (mmol/L)

```matlab
% State 6: CO2_aq (mmol/L dissolved in liquid)
% Initial: CO2_aq0 = Hcp_CO2_eff * pCO2 ~ 5-15 mmol/L
```

**Fiziksel Anlam**:
- Sıvıdaki çözünmüş CO₂ konsantrasyonu
- CO₂ ↔ H₂CO₃ ↔ HCO₃⁻ ↔ CO₃²⁻ dengede
- pH'ya bağlı türlenme

**Differential Equation**:
```matlab
% Liquid balance (mmol/L/day)
dCO2_aq = + J_CO2 - 1*r_meth - 2*r_aceto;

% Source: Gas dissolution (J_CO2)
% Sinks:
%   - Methanogenesis: 1 CO2
%   - Homoacetogenesis: 2 CO2
% NOTE: Sulfate reduction does NOT consume CO2!
```

**Tipik Değer Aralığı**:
- Başlangıç: 5-15 mmol/L
- Son: 2-10 mmol/L

**Monod Kinetics**:
```matlab
mCO2 = CO2_aq / (K_CO2 + CO2_aq);
% K_CO2 ~ 0.8 mmol/L
```

---

### 7. SO4 - Sülfat (mmol/L)

```matlab
% State 7: SO4 (mmol/L sulfate in liquid)
% Initial: SO4_exp(1) ~ 5-15 mmol/L (kayaça göre değişir)
```

**Fiziksel Anlam**:
- Sülfat iyonu (SO₄²⁻)
- Sülfat indirgeyici bakterilerin elektron akseptörü
- Gypsum çözünmesi ile beslenir

**Differential Equation**:
```matlab
% Liquid balance (mmol/L/day)
dSO4 = - 1*r_sulf + r_diss_gyp;

% Sink: Sulfate reduction (-r_sulf)
% Source: Gypsum dissolution (+r_diss_gyp)
```

**Gypsum Dissolution** (Kayaç Özel):
```matlab
% CaSO₄·2H₂O → Ca²⁺ + SO₄²⁻ + 2H₂O
r_diss_gyp = k_diss_gyp * max(0, SO4_sat - SO4);

% Sandstone: SO4_sat = 15.0 mM, k_diss_gyp = 0.12 day⁻¹
% Gypsum:    SO4_sat = 36.0 mM, k_diss_gyp = 0.12 day⁻¹
```

**Tipik Değer Aralığı**:
- Başlangıç: 5-15 mmol/L
- Plateau: 10-15 mM (Sandstone), 30-36 mM (Gypsum)

**Monod Kinetics**:
```matlab
mSO4 = SO4 / (K_SO4 + SO4);
% K_SO4 ~ 0.5 mmol/L
```

**Competition with Methanogens**:
```matlab
% Sulfate inhibits methanogenesis
f_comp_m = 1 / (1 + beta_SO4_m * SO4);
% beta_SO4_m ~ 0.1 mM⁻¹
% High SO4 → f_comp_m ≈ 0 → Methanogens suppressed
```

---

### 8. FeS - Demir Sülfür Çökeltisi (mmol/L)

```matlab
% State 8: FeS (mmol/L precipitated iron sulfide)
% Initial: FeS0 = 0.01 mmol/L (seed)
```

**Fiziksel Anlam**:
- Çökelmiş FeS (pyrrhotite/mackinawite)
- Sülfürü uzaklaştırma mekanizması
- Toksik HS⁻'yi nötralize eder

**Precipitation Reaction**:
```
Fe²⁺ + HS⁻ → FeS(s) + H⁺
```

**Differential Equation**:
```matlab
% Liquid balance (mmol/L/day)
dFeS = + r_prec;

% r_prec_raw = k_prec * max(0, HS_aq - HS_sat);
% r_prec = min(r_prec_raw, Fe_pool);  % Limited by Fe availability!
```

**Tipik Değer Aralığı**:
- Başlangıç: 0.01 mmol/L
- Son: 0.1-5 mmol/L (Fe_pool'a bağlı)

**Kritik Not**:
- **FeS çökelmesi Fe pool tarafından sınırlandırılıyor!**
- Fe tükenince çökelme durur

---

### 9. X - Biyokütle (mmol/L)

```matlab
% State 9: X (mmol/L equivalent biomass units)
% Initial: X0 = 0.01 mmol/L (seed)
```

**Fiziksel Anlam**:
- Toplam bakteri konsantrasyonu
- "Equivalent biomass units" (tam tanım yok, karbon bazlı?)
- Methanogens + SRB + Acetogens toplamı

**Differential Equation**:
```matlab
% Liquid balance (mmol/L/day)
dX = + Y_m*r_meth + Y_s*r_sulf + Y_a*r_aceto - b*X;

% Sources: Biomass yield from each pathway
%   Y_m ~ 0.06 (methanogens)
%   Y_s ~ 0.05 (sulfate reducers)
%   Y_a ~ 0.05 (acetogens)
% Sink: Biomass decay
%   b ~ 0.01 day⁻¹
```

**Tipik Değer Aralığı**:
- Başlangıç: 0.01 mmol/L
- Pik: 0.05-0.2 mmol/L (exponential growth)
- Son: 0.02-0.1 mmol/L (decay)

**Monod Kinetics** (Her reaksiyon için):
```matlab
r_meth  = k_m * X * mH2 * mCO2 * f_inh_m * f_act * fT_m * f_comp_m;
r_sulf  = k_s * X * mH2 * mSO4 * f_inh_s * f_act * fT_s;
r_aceto = k_a * X * mH2 * (mCO2^2) * f_inh_a * f_act * fT_a;
```

---

### 10. Acetate - Asetat (mmol/L)

```matlab
% State 10: Acetate (mmol/L acetic acid)
% Initial: Ac0 = 0 mmol/L
```

**Fiziksel Anlam**:
- Asetik asit (CH₃COO⁻)
- Homoacetogenesis ürünü
- Methanogen substratı olabilir (bu modelde yok)

**Differential Equation**:
```matlab
% Liquid balance (mmol/L/day)
dAc = + r_aceto;

% Source: Homoacetogenesis
% 4H₂ + 2CO₂ → CH₃COO⁻ + H⁺ + 2H₂O
```

**Tipik Değer Aralığı**:
- Başlangıç: 0 mmol/L
- Son: 0-10 mmol/L

**Not**:
- Bu modelde acetoclastic methanogenesis YOK
- Acetate sadece accumulate oluyor

---

### 11. HCO3 - Bikarbonat (mmol/L)

```matlab
% State 11: HCO3 (mmol/L bicarbonate)
% Initial: HCO3_0 = 0 mmol/L
```

**Fiziksel Anlam**:
- Bikarbonat iyonu (HCO₃⁻)
- pH buffer sistemi
- CO₂ ↔ H₂CO₃ ↔ HCO₃⁻ ↔ CO₃²⁻

**Differential Equation**:
```matlab
% Liquid balance (mmol/L/day)
dHCO3 = 0;

% KEPT CONSTANT (pH buffering assumption)
% In reality: SR produces HCO3, but pH interpolant used instead
```

**Tipik Değer Aralığı**:
- Sabit: 0 mmol/L (bu modelde kullanılmıyor)

**Not**:
- pH experimentally measured → pH_fun(t) interpolant
- HCO₃⁻ dynamicsi tracked değil

---

### 12. S_tot - Toplam Çözünmüş Sülfür (mmol/L)

```matlab
% State 12: S_tot (mmol/L total dissolved sulfide)
% S_tot = H2S(aq) + HS⁻
% Initial: S_tot0 = 1.0 mmol/L (seed for early H2S appearance)
```

**Fiziksel Anlam**:
- Toplam çözünmüş sülfür türleri
- pH'ya bağlı olarak H₂S ↔ HS⁻ dengede

**Differential Equation**:
```matlab
% Liquid balance (mmol/L/day)
dS_tot = + 1.00*r_sulf - r_prec - Jout_H2S;

% Source: Sulfate reduction (+r_sulf)
% Sinks:
%   - FeS precipitation (-r_prec)
%   - H2S degassing (-Jout_H2S)
```

**pH Speciation**:
```matlab
% H2S ⇌ HS⁻ + H⁺  (pKa = 7.05 @ 25°C)
frac_HS  = 1 / (1 + 10^(pKa - pH));
frac_H2S = 1 - frac_HS;

HS_aq  = S_tot * frac_HS;   % Ionic form (toxic!)
H2S_aq = S_tot * frac_H2S;  % Molecular form (volatile)
```

**Tipik Değer Aralığı**:
- Başlangıç: 1.0 mmol/L
- Pik: 5-20 mmol/L (SR aktif)
- Son: 2-10 mmol/L

---

### 13. Lag - Lag Aktivasyon (0-1)

```matlab
% State 13: Lag (dimensionless, 0-1)
% Smooth activation gate for reactions
% Initial: Lag0 = 0
```

**Fiziksel Anlam**:
- Bakterilerin adaptasyon zamanı
- 0 → Inactive, 1 → Fully active
- Sigmoid geçiş (smooth)

**Differential Equation**:
```matlab
% Lag tracker (1/day)
dLag = (f_lag - Lag) / max(w_lag, 1e-3);

% f_lag = 1 / (1 + exp((t_lag - t) / w_lag))
% t_lag ~ 3.0 day (center)
% w_lag ~ 0.7 day (width)
```

**Fiziksel Yorum**:
```
t < t_lag - w_lag  →  f_lag ≈ 0  →  Inactive
t ≈ t_lag          →  f_lag ≈ 0.5 →  Activating
t > t_lag + w_lag  →  f_lag ≈ 1  →  Fully active
```

**Tipik Değer Aralığı**:
- 0-2 gün: Lag ≈ 0 (inactive)
- 3-4 gün: Lag ≈ 0.5 (activating)
- 5+ gün: Lag ≈ 1 (active)

**Neden Gerekli?**:
- Deneysel olarak: İlk 2-3 gün reaksiyon yok
- Bakteriler adaptasyon süresi gerektiriyor

---

### 14. Fe_pool - Çözünmüş Demir Havuzu (mmol/L)

```matlab
% State 14: Fe_pool (mmol/L dissolved Fe(II))
% Initial: Fe_pool0 = 0.10 mmol/L (choose 0.05-0.5 mM)
```

**Fiziksel Anlam**:
- Çözünmüş Fe²⁺ konsantrasyonu
- FeS çökelmesi için gerekli
- **Sınırlı kaynak** (tükendikçe çökelme durur)

**Differential Equation**:
```matlab
% Liquid balance (mmol/L/day)
dFe_pool = - r_prec;

% Fe²⁺ + HS⁻ → FeS(s) + H⁺
% 1:1 stoichiometry
```

**Precipitation Limit**:
```matlab
% Raw precipitation rate
r_prec_raw = k_prec * max(0, HS_aq - HS_sat);

% Limited by Fe availability
r_prec = min(r_prec_raw, Fe_pool);

% When Fe_pool → 0, r_prec → 0 (no more precipitation)
```

**Tipik Değer Aralığı**:
- Başlangıç: 0.05-0.5 mmol/L
- Son: 0-0.1 mmol/L (depleted)

**Kritik Kavram**:
```
High HS⁻ + High Fe²⁺  →  FeS precipitation
High HS⁻ + Low Fe²⁺   →  HS⁻ accumulates (toxic!)
```

---

## State Değişkenleri Arasındaki İlişkiler

### Gaz-Sıvı Dengesi (Henry)

```
nH2_g  ↔  H2_aq   (Henry: Hcp_H2  = 0.78 mmol/L/atm)
nCO2_g ↔  CO2_aq  (Henry: Hcp_CO2 = 34.0 mmol/L/atm)
nH2S_g ↔  H2S_aq  (Henry: Hcp_H2S = 90.0 mmol/L/atm)
```

### pH Bağımlı Türlenme

```
S_tot  →  H2S_aq + HS_aq  (pH dependent, pKa=7.05)
```

### Reaksiyon Zinciri

```
H2_aq + CO2_aq + SO4  →  (bacteria: X)  →  CH4, H2S, Acetate
                         ↓
                      Growth: dX > 0
```

### Sülfür Döngüsü

```
SO4  →  (SR)  →  S_tot  →  HS_aq  →  (prec)  →  FeS
                            ↓
                       (degassing)
                            ↓
                         H2S_g
```

---

## Özet Tablo

| # | State | Birim | Fazı | Başlangıç | Son | Rol |
|---|-------|-------|------|-----------|-----|-----|
| 1 | nH2_g | mmol | Gaz | 100-200 | 0-20 | Substrat (tüketilir) |
| 2 | nCO2_g | mmol | Gaz | 50-100 | 10-50 | Substrat (tüketilir) |
| 3 | nCH4_g | mmol | Gaz | 0-5 | 50-150 | Ürün (üretilir) |
| 4 | nH2S_g | mmol | Gaz | 0-2 | 5-50 | Ürün (üretilir) |
| 5 | H2_aq | mmol/L | Sıvı | 0.5-2 | 0.01-0.1 | Substrat (Henry dengesi) |
| 6 | CO2_aq | mmol/L | Sıvı | 5-15 | 2-10 | Substrat (Henry dengesi) |
| 7 | SO4 | mmol/L | Sıvı | 5-15 | 10-15 | Substrat (buffer) |
| 8 | FeS | mmol/L | Katı | 0.01 | 0.1-5 | Sink (çökelme) |
| 9 | X | mmol/L | Sıvı | 0.01 | 0.05-0.2 | Katalizör (bakteriler) |
| 10 | Acetate | mmol/L | Sıvı | 0 | 0-10 | Ürün (accumulation) |
| 11 | HCO3 | mmol/L | Sıvı | 0 | 0 | Buffer (kullanılmıyor) |
| 12 | S_tot | mmol/L | Sıvı | 1.0 | 2-10 | Ürün (SR) |
| 13 | Lag | - | - | 0 | 1 | Aktivasyon (gate) |
| 14 | Fe_pool | mmol/L | Sıvı | 0.05-0.5 | 0-0.1 | Kaynak (Fe²⁺) |

---

**Hazırlayan**: Hasan Arı  
**Tarih**: 6 Ocak 2026  
**Kaynak**: Kod satır 13-137 (state tanımları)
