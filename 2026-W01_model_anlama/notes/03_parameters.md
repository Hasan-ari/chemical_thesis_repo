# 28 Parameters - Detaylı Açıklama

**Dosya**: `anaerobic_model_two_phase_mixedSR_25C_v4.m`  
**Tarih**: 6 Ocak 2026

---

## Parameter Vektörü (p)

```matlab
% p = [k_m, k_s, k_a,                              % 1-3:   Kinetic rate constants
%      Y_m, Y_s, Y_a,                              % 4-6:   Biomass yields
%      KI_m, KI_s, KI_a,                           % 7-9:   Sulfide inhibition constants
%      k_prec, HS_sat,                             % 10-11: FeS precipitation
%      H2_th, DG_th,                               % 12-13: Activation thresholds
%      K_H2, K_SO4, K_CO2,                         % 14-16: Monod half-saturations
%      kla_H2, kla_CO2, kla_H2S,                   % 17-19: Mass transfer coefficients
%      b,                                          % 20:    Biomass decay
%      t_lag, w_lag,                               % 21-22: Lag parameters
%      k_diss_gyp,                                 % 23:    Gypsum dissolution
%      beta_SO4_m,                                 % 24:    Sulfate-methanogen competition
%      phi_H2, phi_CO2, phi_H2S,                   % 25-27: Henry scale factors
%      alpha_H2S]                                  % 28:    H2S degassing scale
```

**Toplam**: 28 parameters  
**v3'ten fark**: +15 parametre (Henry factors, lag, Fe pool, vb.)

---

## Kategoriler

| Kategori | Parametreler | Sayı |
|----------|-------------|------|
| Kinetik Sabitler | k_m, k_s, k_a | 3 |
| Biyokütle Verimi | Y_m, Y_s, Y_a | 3 |
| İnhibisyon | KI_m, KI_s, KI_a | 3 |
| FeS Çökelmesi | k_prec, HS_sat | 2 |
| Aktivasyon | H2_th, DG_th | 2 |
| Monod Sabitleri | K_H2, K_SO4, K_CO2 | 3 |
| Kütle Transferi | kla_H2, kla_CO2, kla_H2S | 3 |
| Biyokütle Dinamiği | b | 1 |
| Lag Fazı | t_lag, w_lag | 2 |
| Kayaç Özel | k_diss_gyp | 1 |
| Rekabet | beta_SO4_m | 1 |
| Henry Faktörleri | phi_H2, phi_CO2, phi_H2S | 3 |
| Degassing | alpha_H2S | 1 |

---

## 1-3: Kinetik Hız Sabitleri

### p(1): k_m - Methanogenesis Rate (1/day)

```matlab
% Parameter 1: k_m (maximum rate constant for methanogenesis)
% Typical: 0.06 day⁻¹
% Bounds: [1e-4, 5.0]
```

**Fiziksel Anlam**:
- Maksimum methanogenesis hızı
- 4H₂ + CO₂ → CH₄ + 2H₂O

**Reaksiyon Hızı**:
```matlab
r_meth = k_m * X * mH2 * mCO2 * f_inh_m * f_act * fT_m * f_comp_m;
```

**Tipik Değer**:
- Sandstone 25°C: ~0.06 day⁻¹
- Daha yüksek sıcaklık: Daha yüksek k_m (Arrhenius)

**Fit Hassasiyeti**: ⭐⭐⭐⭐⭐ (çok kritik - CH₄ üretimi)

---

### p(2): k_s - Sulfate Reduction Rate (1/day)

```matlab
% Parameter 2: k_s (maximum rate constant for sulfate reduction)
% Typical: 0.08 day⁻¹
% Bounds: [1e-4, 5.0]
```

**Fiziksel Anlam**:
- Maksimum sulfate reduction hızı
- 4H₂ + SO₄²⁻ → HS⁻ + 3H₂O + OH⁻

**Reaksiyon Hızı**:
```matlab
r_sulf = k_s * X * mH2 * mSO4 * f_inh_s * f_act * fT_s;
```

**Tipik Değer**:
- Sandstone 25°C: ~0.08 day⁻¹
- Gypsum: Daha yüksek (SO₄ bolluğu)

**Fit Hassasiyeti**: ⭐⭐⭐⭐⭐ (çok kritik - H₂S üretimi)

---

### p(3): k_a - Homoacetogenesis Rate (1/day)

```matlab
% Parameter 3: k_a (maximum rate constant for homoacetogenesis)
% Typical: 0.03 day⁻¹
% Bounds: [1e-4, 5.0]
```

**Fiziksel Anlam**:
- Maksimum homoacetogenesis hızı
- 4H₂ + 2CO₂ → CH₃COO⁻ + H⁺ + 2H₂O

**Reaksiyon Hızı**:
```matlab
r_aceto = k_a * X * mH2 * (mCO2^2) * f_inh_a * f_act * fT_a;
```

**Tipik Değer**:
- Sandstone 25°C: ~0.03 day⁻¹
- Genellikle k_a < k_m, k_s (daha az baskın)

**Fit Hassasiyeti**: ⭐⭐⭐ (orta - acetate az ölçülüyor)

---

## 4-6: Biyokütle Verimi

### p(4): Y_m - Methanogen Yield (mmolX/mmol)

```matlab
% Parameter 4: Y_m (biomass yield from methanogenesis)
% Typical: 0.06 mmolX/mmol substrate
% Bounds: [0.01, 0.5]
```

**Fiziksel Anlam**:
- Methanogenesis'den elde edilen biyokütle
- 1 mmol CH₄ üretimi → Y_m mmol biyokütle

**Biyokütle Dinamiği**:
```matlab
dX = + Y_m*r_meth + Y_s*r_sulf + Y_a*r_aceto - b*X;
```

**Tipik Değer**:
- Y_m ~ 0.06 (düşük yield - enerji verimliliği düşük)

**Fit Hassasiyeti**: ⭐⭐ (düşük - X doğrudan ölçülmüyor)

---

### p(5): Y_s - SRB Yield (mmolX/mmol)

```matlab
% Parameter 5: Y_s (biomass yield from sulfate reduction)
% Typical: 0.05 mmolX/mmol substrate
% Bounds: [0.01, 0.5]
```

**Fiziksel Anlam**:
- Sulfate reduction'dan elde edilen biyokütle

**Tipik Değer**:
- Y_s ~ 0.05 (düşük yield)

**Fit Hassasiyeti**: ⭐⭐

---

### p(6): Y_a - Acetogen Yield (mmolX/mmol)

```matlab
% Parameter 6: Y_a (biomass yield from homoacetogenesis)
% Typical: 0.05 mmolX/mmol substrate
% Bounds: [0.01, 0.5]
```

**Fiziksel Anlam**:
- Homoacetogenesis'den elde edilen biyokütle

**Tipik Değer**:
- Y_a ~ 0.05

**Fit Hassasiyeti**: ⭐⭐

---

## 7-9: Sülfür İnhibisyonu

### p(7): KI_m - Methanogen Sulfide Inhibition (mmol/L)

```matlab
% Parameter 7: KI_m (sulfide inhibition constant for methanogens)
% Typical: 0.20 mmol/L as HS⁻ equivalent
% Bounds: [1e-3, 5.0]
```

**Fiziksel Anlam**:
- HS⁻ methanogens'i inhibe eder
- KI_m: HS⁻ seviyesi bu değerde → %50 inhibisyon

**İnhibisyon Fonksiyonu**:
```matlab
f_inh_m = KI_m / (KI_m + HS_aq);

% HS_aq << KI_m  →  f_inh_m ≈ 1 (no inhibition)
% HS_aq >> KI_m  →  f_inh_m ≈ 0 (full inhibition)
```

**Tipik Değer**:
- KI_m ~ 0.20 mmol/L
- Methanogens HS⁻'ye hassas!

**Fit Hassasiyeti**: ⭐⭐⭐⭐ (önemli - HS⁻ toksisitesi)

---

### p(8): KI_s - SRB Sulfide Inhibition (mmol/L)

```matlab
% Parameter 8: KI_s (sulfide inhibition constant for SRB)
% Typical: 0.20 mmol/L
% Bounds: [1e-3, 5.0]
```

**Fiziksel Anlam**:
- SRB de HS⁻'den etkilenir ama daha az hassas

**İnhibisyon Fonksiyonu**:
```matlab
f_inh_s = KI_s / (KI_s + HS_aq);
```

**Tipik Değer**:
- KI_s ~ 0.20 mmol/L
- SRB genellikle KI_s > KI_m (daha tolerant)

**Fit Hassasiyeti**: ⭐⭐⭐⭐

---

### p(9): KI_a - Acetogen Sulfide Inhibition (mmol/L)

```matlab
% Parameter 9: KI_a (sulfide inhibition constant for acetogens)
% Typical: 0.20 mmol/L
% Bounds: [1e-3, 5.0]
```

**Fiziksel Anlam**:
- Acetogens de HS⁻'den etkilenir

**İnhibisyon Fonksiyonu**:
```matlab
f_inh_a = KI_a / (KI_a + HS_aq);
```

**Tipik Değer**:
- KI_a ~ 0.20 mmol/L

**Fit Hassasiyeti**: ⭐⭐⭐

---

## 10-11: FeS Çökelmesi

### p(10): k_prec - FeS Precipitation Rate (1/day)

```matlab
% Parameter 10: k_prec (precipitation kinetic factor for FeS)
% Typical: 0.02 day⁻¹
% Bounds: [0.0, 1.0]
```

**Fiziksel Anlam**:
- FeS çökelme hızı (Fe²⁺ + HS⁻ → FeS)

**Çökelme Hızı**:
```matlab
r_prec_raw = k_prec * max(0, HS_aq - HS_sat);
r_prec     = min(r_prec_raw, Fe_pool);  % Fe limited
```

**Tipik Değer**:
- k_prec ~ 0.02 day⁻¹ (hızlı çökelme)
- k_prec = 0 → Çökelme yok

**Fit Hassasiyeti**: ⭐⭐⭐⭐ (HS⁻ konsantrasyonu belirler)

---

### p(11): HS_sat - HS⁻ Solubility Threshold (mmol/L)

```matlab
% Parameter 11: HS_sat (HS⁻ solubility threshold for precipitation)
% Typical: 0.10 mmol/L
% Bounds: [0.0, 5.0]
```

**Fiziksel Anlam**:
- HS⁻ çözünürlük eşiği
- HS_aq > HS_sat → Çökelme başlar

**Çökelme Koşulu**:
```matlab
% Oversaturation drives precipitation
if HS_aq > HS_sat:
    r_prec_raw = k_prec * (HS_aq - HS_sat)
else:
    r_prec_raw = 0
```

**Tipik Değer**:
- HS_sat ~ 0.10 mmol/L

**Fit Hassasiyeti**: ⭐⭐⭐⭐

---

## 12-13: Aktivasyon Eşikleri

### p(12): H2_th - H2 Activation Threshold (mmol/L)

```matlab
% Parameter 12: H2_th (dissolved H2 threshold for activation gate)
% Typical: 0.02 mmol/L
% Bounds: [0.0, 1.0]
```

**Fiziksel Anlam**:
- Minimum H₂ konsantrasyonu aktivasyon için

**Aktivasyon Fonksiyonu**:
```matlab
f_H2 = H2_aq / (H2_aq + H2_th);

% H2_aq >> H2_th  →  f_H2 ≈ 1 (saturated)
% H2_aq ≈ H2_th   →  f_H2 ≈ 0.5 (half)
% H2_aq << H2_th  →  f_H2 ≈ 0 (inactive)
```

**Tipik Değer**:
- H2_th ~ 0.02 mmol/L (düşük eşik - kolayca aktive olur)

**Fit Hassasiyeti**: ⭐⭐⭐

---

### p(13): DG_th - Thermodynamic Threshold (kJ/mol)

```matlab
% Parameter 13: DG_th (free-energy threshold for thermodynamic gates)
% Typical: -12 kJ/mol
% Bounds: [-50, 0]
```

**Fiziksel Anlam**:
- Termodinamik eşik (ΔG)
- ΔG < DG_th → Reaksiyon termodinamikçe favourable

**Termodinamik Gate**:
```matlab
% Gibbs free energy
DG_m = DG0_m;  % -130 kJ/mol (always favorable)
DG_s = DG0_s + RT*log(Q_s);  % -152 kJ/mol
DG_a = DG0_a + RT*log(Q_a);

% Thermodynamic gate
fT_m = 1 / (1 + exp((DG_m - DG_th) / RT));
fT_s = 1 / (1 + exp((DG_s - DG_th) / RT));
fT_a = 1 / (1 + exp((DG_a - DG_th) / RT));
```

**Tipik Değer**:
- DG_th ~ -12 kJ/mol
- ΔG çok negatif → fT ≈ 1 (aktif)

**Fit Hassasiyeti**: ⭐⭐ (düşük - hep negatif ΔG)

---

## 14-16: Monod Half-Saturation Constants

### p(14): K_H2 - H2 Monod Constant (mmol/L)

```matlab
% Parameter 14: K_H2 (Monod half-saturation for dissolved H2)
% Typical: 0.50 mmol/L
% Bounds: [1e-3, 20]
```

**Fiziksel Anlam**:
- Monod sabiti (Michaelis-Menten kinetikleri)
- H2_aq = K_H2 → %50 doygunluk

**Monod Fonksiyonu**:
```matlab
mH2 = H2_aq / (K_H2 + H2_aq);

% H2_aq >> K_H2  →  mH2 ≈ 1 (zero-order)
% H2_aq ≈ K_H2   →  mH2 ≈ 0.5 (half-saturated)
% H2_aq << K_H2  →  mH2 ≈ H2_aq/K_H2 (first-order)
```

**Tipik Değer**:
- K_H2 ~ 0.50 mmol/L

**Fit Hassasiyeti**: ⭐⭐⭐⭐ (önemli - H₂ tüketimi)

---

### p(15): K_SO4 - SO4 Monod Constant (mmol/L)

```matlab
% Parameter 15: K_SO4 (Monod half-saturation for SO4)
% Typical: 0.50 mmol/L
% Bounds: [1e-3, 20]
```

**Fiziksel Anlam**:
- SO₄ Monod sabiti

**Monod Fonksiyonu**:
```matlab
mSO4 = SO4 / (K_SO4 + SO4);
```

**Tipik Değer**:
- K_SO4 ~ 0.50 mmol/L

**Fit Hassasiyeti**: ⭐⭐⭐⭐

---

### p(16): K_CO2 - CO2 Monod Constant (mmol/L)

```matlab
% Parameter 16: K_CO2 (Monod half-saturation for dissolved CO2)
% Typical: 0.80 mmol/L
% Bounds: [1e-3, 20]
```

**Fiziksel Anlam**:
- CO₂ Monod sabiti

**Monod Fonksiyonu**:
```matlab
mCO2 = CO2_aq / (K_CO2 + CO2_aq);
```

**Tipik Değer**:
- K_CO2 ~ 0.80 mmol/L

**Fit Hassasiyeti**: ⭐⭐⭐⭐

---

## 17-19: Kütle Transfer Katsayıları (kLa)

### p(17): kla_H2 - H2 Mass Transfer (1/day)

```matlab
% Parameter 17: kla_H2 (gas-liquid mass transfer coefficient for H2)
% Typical: 10.0 day⁻¹
% Bounds: [0.1, 200]
```

**Fiziksel Anlam**:
- Gaz-sıvı kütle transferi hızı (H₂)
- Yüksek kLa → Hızlı denge

**Transfer Akısı**:
```matlab
J_H2 = kla_H2 * (Ceq_H2 - H2_aq);

% kla_H2 large → Fast equilibration
% kla_H2 small → Slow equilibration
```

**Tipik Değer**:
- kla_H2 ~ 10 day⁻¹ (orta hız)

**Fit Hassasiyeti**: ⭐⭐⭐⭐⭐ (çok kritik - gaz tüketimi)

---

### p(18): kla_CO2 - CO2 Mass Transfer (1/day)

```matlab
% Parameter 18: kla_CO2 (gas-liquid mass transfer coefficient for CO2)
% Typical: 10.0 day⁻¹
% Bounds: [0.1, 200]
```

**Fiziksel Anlam**:
- CO₂ kütle transferi

**Transfer Akısı**:
```matlab
J_CO2 = kla_CO2 * (Ceq_CO2 - CO2_aq);
```

**Tipik Değer**:
- kla_CO2 ~ 10 day⁻¹

**Fit Hassasiyeti**: ⭐⭐⭐⭐⭐

---

### p(19): kla_H2S - H2S Mass Transfer (1/day)

```matlab
% Parameter 19: kla_H2S (gas-liquid mass transfer coefficient for H2S)
% Typical: 25.0 day⁻¹ (higher than H2/CO2!)
% Bounds: [0.1, 200]
```

**Fiziksel Anlam**:
- H₂S kütle transferi (degassing)
- **Yüksek kla_H2S** → H₂S hızlı gazlaşır

**Degassing Akısı**:
```matlab
Jout_H2S = kla_H2S * (H2S_aq - Ceq_H2S);

% H2S volatilizes faster than H2/CO2
% kla_H2S typically > kla_H2, kla_CO2
```

**Tipik Değer**:
- kla_H2S ~ 25 day⁻¹ (yüksek!)

**Fit Hassasiyeti**: ⭐⭐⭐⭐⭐ (çok kritik - H₂S(g) miktarı)

---

## 20: Biyokütle Bozunması

### p(20): b - Biomass Decay Rate (1/day)

```matlab
% Parameter 20: b (biomass decay coefficient)
% Typical: 0.01 day⁻¹
% Bounds: [0, 0.2]
```

**Fiziksel Anlam**:
- Biyokütle bozunma/ölüm hızı

**Biyokütle Dinamiği**:
```matlab
dX = + (Y_m*r_meth + Y_s*r_sulf + Y_a*r_aceto) - b*X;
     ↑ Growth                                     ↑ Decay
```

**Tipik Değer**:
- b ~ 0.01 day⁻¹ (yavaş bozunma)
- Half-life: ln(2)/b ~ 69 gün

**Fit Hassasiyeti**: ⭐⭐ (düşük - X ölçülmüyor)

---

## 21-22: Lag Fazı

### p(21): t_lag - Lag Center Time (days)

```matlab
% Parameter 21: t_lag (lag center time)
% Typical: 3.0 days
% Bounds: [0, 10]
```

**Fiziksel Anlam**:
- Bakterilerin aktive olma zamanı (merkez)

**Lag Fonksiyonu**:
```matlab
f_lag = 1 / (1 + exp((t_lag - t) / w_lag));

% t < t_lag  →  f_lag ≈ 0 (inactive)
% t = t_lag  →  f_lag = 0.5 (half-active)
% t > t_lag  →  f_lag ≈ 1 (active)
```

**Tipik Değer**:
- t_lag ~ 3.0 gün (3 gün sonra aktive)

**Fit Hassasiyeti**: ⭐⭐⭐⭐ (önemli - ilk günler)

---

### p(22): w_lag - Lag Width (days)

```matlab
% Parameter 22: w_lag (lag width)
% Typical: 0.7 days
% Bounds: [0.1, 2.0]
```

**Fiziksel Anlam**:
- Aktivasyon geçiş genişliği

**Geçiş Hızı**:
```
w_lag small → Sharp transition (sigmoid steep)
w_lag large → Smooth transition (sigmoid gradual)
```

**Tipik Değer**:
- w_lag ~ 0.7 gün

**Fit Hassasiyeti**: ⭐⭐⭐

---

## 23: Kayaç Özel Parametre

### p(23): k_diss_gyp - Gypsum Dissolution Rate (1/day)

```matlab
% Parameter 23: k_diss_gyp (dissolution rate feeding SO4 buffer)
% Typical: 0.12 day⁻¹
% Bounds: [0.01, 2.00]
```

**Fiziksel Anlam**:
- Gypsum (CaSO₄·2H₂O) çözünme hızı
- **Kayaç özel** (Gypsum için kritik!)

**Çözünme Hızı**:
```matlab
r_diss_gyp = k_diss_gyp * max(0, SO4_sat - SO4);

% SO4 < SO4_sat → Dissolution occurs
% SO4 ≥ SO4_sat → No dissolution (saturated)
```

**Kayaçlara Göre**:
| Kayaç | SO4_sat (mM) | k_diss_gyp (day⁻¹) |
|-------|--------------|---------------------|
| Sandstone | 15.0 | 0.12 |
| Basalt | 15.0 | 0.12 |
| Calcite | 15.0 | 0.12 |
| Gypsum | 36.0 | 0.12 (çok önemli!) |

**Fit Hassasiyeti**: ⭐⭐⭐⭐⭐ (Gypsum için kritik!)

---

## 24: Rekabet Parametresi

### p(24): beta_SO4_m - Sulfate-Methanogen Competition (mM⁻¹)

```matlab
% Parameter 24: beta_SO4_m (sulfate-methanogen competition strength)
% Typical: 0.10 mM⁻¹
% Bounds: [0.00, 1.00]
```

**Fiziksel Anlam**:
- SO₄ varlığı methanogens'i inhibe eder
- Yüksek SO₄ → SRB dominant, methanogens suppress

**Competition Gate**:
```matlab
f_comp_m = 1 / (1 + beta_SO4_m * SO4);

% SO4 = 0     → f_comp_m = 1 (no competition)
% SO4 = 10 mM → f_comp_m ≈ 0.5 (if beta=0.1)
% SO4 high    → f_comp_m ≈ 0 (methanogens suppressed)
```

**Tipik Değer**:
- beta_SO4_m ~ 0.10 mM⁻¹

**Fit Hassasiyeti**: ⭐⭐⭐⭐ (önemli - SR/methanogen balance)

---

## 25-27: Henry Scale Factors

### p(25): phi_H2 - H2 Henry Scale Factor (-)

```matlab
% Parameter 25: phi_H2 (Henry scale factor for H2)
% Typical: 1.00 (dimensionless)
% Bounds: [0.85, 1.15]
```

**Fiziksel Anlam**:
- Henry sabitine çarpan (fine-tuning)
- Hcp_H2_eff = phi_H2 * Hcp_H2_base

**Neden Gerekli?**:
- Literature Henry sabitleri ±10-15% hata içerebilir
- Ortam koşullarına göre ayar (salinity, etc.)

**Tipik Değer**:
- phi_H2 ~ 1.00 (±15%)

**Fit Hassasiyeti**: ⭐⭐⭐⭐

---

### p(26): phi_CO2 - CO2 Henry Scale Factor (-)

```matlab
% Parameter 26: phi_CO2 (Henry scale factor for CO2)
% Typical: 1.00
% Bounds: [0.85, 1.15]
```

**Fiziksel Anlam**:
- CO₂ Henry sabitine çarpan

**Tipik Değer**:
- phi_CO2 ~ 1.00

**Fit Hassasiyeti**: ⭐⭐⭐⭐

---

### p(27): phi_H2S - H2S Henry Scale Factor (-)

```matlab
% Parameter 27: phi_H2S (Henry scale factor for H2S)
% Typical: 1.00
% Bounds: [0.90, 1.10]
```

**Fiziksel Anlam**:
- H₂S Henry sabitine çarpan
- **En hassas parametre** (H₂S(g) fit kalitesi)

**Tipik Değer**:
- phi_H2S ~ 1.00

**Fit Hassasiyeti**: ⭐⭐⭐⭐⭐ (çok kritik!)

---

## 28: Degassing Scale Factor

### p(28): alpha_H2S - H2S Degassing Scale (-)

```matlab
% Parameter 28: alpha_H2S (H2S degassing scale multiplying kla_H2S)
% Typical: 1.00
% Bounds: [0.70, 3.00]
```

**Fiziksel Anlam**:
- H₂S degassing'i için ekstra faktör
- Interfacial effects, film resistance

**Effective kLa**:
```matlab
% Actual H2S transfer uses:
kla_H2S_eff = alpha_H2S * kla_H2S;
```

**Neden Gerekli?**:
- H₂S surface-active (yüzey etkileşimleri)
- Film resistance ekstra direnç yaratır
- alpha_H2S < 1 → Slower degassing
- alpha_H2S > 1 → Faster degassing

**Tipik Değer**:
- alpha_H2S ~ 0.7-1.5

**Fit Hassasiyeti**: ⭐⭐⭐⭐⭐ (çok kritik - H₂S(g))

---

## Parametre Hassasiyet Sıralaması

### En Kritik (⭐⭐⭐⭐⭐)
1. **kla_H2**, **kla_CO2**, **kla_H2S** → Gaz-sıvı dengesi
2. **k_m**, **k_s** → Reaksiyon hızları
3. **phi_H2S**, **alpha_H2S** → H₂S fit kalitesi
4. **k_diss_gyp** → Gypsum için kritik

### Önemli (⭐⭐⭐⭐)
5. **KI_m**, **KI_s** → Sülfür inhibisyonu
6. **K_H2**, **K_SO4**, **K_CO2** → Monod kinetikleri
7. **t_lag** → Aktivasyon zamanı
8. **beta_SO4_m** → SR-methanogen rekabet
9. **k_prec**, **HS_sat** → FeS çökelmesi

### Orta (⭐⭐⭐)
10. **k_a**, **KI_a** → Homoacetogenesis
11. **H2_th**, **w_lag** → Aktivasyon

### Düşük (⭐⭐)
12. **Y_m**, **Y_s**, **Y_a**, **b** → Biyokütle (ölçülmüyor)
13. **DG_th** → Termodinamik (hep favorable)

---

## Kayaçlara Göre Farklılıklar

### Sandstone 25°C
```matlab
k_m  = 0.06, k_s  = 0.08, k_a  = 0.03
kla_H2 = 10.0, kla_CO2 = 10.0, kla_H2S = 25.0
phi_H2 = 1.00, phi_CO2 = 1.00, phi_H2S = 1.00
alpha_H2S = 1.00
SO4_sat = 15.0 mM, k_diss_gyp = 0.12 day⁻¹
```

### Gypsum (Farklı!)
```matlab
SO4_sat = 36.0 mM   ← Çok yüksek!
k_diss_gyp = 0.12 day⁻¹  (aynı)
k_s daha yüksek olabilir (SO₄ bolluğu)
```

### Sıcaklık Etkisi
```matlab
25°C → 34°C → 40°C:
- k_m, k_s, k_a artar (Arrhenius)
- kla değerleri artar (diffusion)
- Henry sabitleri azalır (çözünürlük azalır)
```

---

## Fitting Stratejisi

### Initial Guess (p0)
```matlab
p0 = [0.06, 0.08, 0.03, ...  % k_m, k_s, k_a
      0.06, 0.05, 0.05, ...  % Y_m, Y_s, Y_a
      0.20, 0.20, 0.20, ...  % KI_m, KI_s, KI_a
      0.02, 0.10, 0.02, -12, ...  % k_prec, HS_sat, H2_th, DG_th
      0.50, 0.50, 0.80, ...  % K_H2, K_SO4, K_CO2
      10.0, 10.0, 25.0, ...  % kla_H2, kla_CO2, kla_H2S
      0.01, 3.0, 0.7, ...    % b, t_lag, w_lag
      0.12, 0.10, ...        % k_diss_gyp, beta_SO4_m
      1.00, 1.00, 1.00, ...  % phi_H2, phi_CO2, phi_H2S
      1.00];                 % alpha_H2S
```

### Bounds
- Kinetic rates: `[1e-4, 5]`
- Yields: `[0.01, 0.5]`
- Inhibition: `[1e-3, 5]`
- Monod: `[1e-3, 20]`
- kLa: `[0.1, 200]`
- Factors: `[0.85, 1.15]` (±15%)

### Fitting Weights
```matlab
% Log1p residuals with weights
weights = [1, 1, 0.9, 1.0, 2.0];
%          ↑  ↑  ↑    ↑    ↑
%          H2 CO2 CH4 H2S  SO4

% SO4 has 2.0 weight → Emphasize sulfate plateau
```

---

## Özet Tablo

| # | Parametre | Birim | Tipik | Kaynak | Hassasiyet |
|---|-----------|-------|-------|--------|-----------|
| 1 | k_m | day⁻¹ | 0.06 | Fit | ⭐⭐⭐⭐⭐ |
| 2 | k_s | day⁻¹ | 0.08 | Fit | ⭐⭐⭐⭐⭐ |
| 3 | k_a | day⁻¹ | 0.03 | Fit | ⭐⭐⭐ |
| 4 | Y_m | - | 0.06 | Fit | ⭐⭐ |
| 5 | Y_s | - | 0.05 | Fit | ⭐⭐ |
| 6 | Y_a | - | 0.05 | Fit | ⭐⭐ |
| 7 | KI_m | mM | 0.20 | Fit | ⭐⭐⭐⭐ |
| 8 | KI_s | mM | 0.20 | Fit | ⭐⭐⭐⭐ |
| 9 | KI_a | mM | 0.20 | Fit | ⭐⭐⭐ |
| 10 | k_prec | day⁻¹ | 0.02 | Fit | ⭐⭐⭐⭐ |
| 11 | HS_sat | mM | 0.10 | Fit | ⭐⭐⭐⭐ |
| 12 | H2_th | mM | 0.02 | Fit | ⭐⭐⭐ |
| 13 | DG_th | kJ/mol | -12 | Fit | ⭐⭐ |
| 14 | K_H2 | mM | 0.50 | Fit | ⭐⭐⭐⭐ |
| 15 | K_SO4 | mM | 0.50 | Fit | ⭐⭐⭐⭐ |
| 16 | K_CO2 | mM | 0.80 | Fit | ⭐⭐⭐⭐ |
| 17 | kla_H2 | day⁻¹ | 10.0 | Fit | ⭐⭐⭐⭐⭐ |
| 18 | kla_CO2 | day⁻¹ | 10.0 | Fit | ⭐⭐⭐⭐⭐ |
| 19 | kla_H2S | day⁻¹ | 25.0 | Fit | ⭐⭐⭐⭐⭐ |
| 20 | b | day⁻¹ | 0.01 | Fit | ⭐⭐ |
| 21 | t_lag | day | 3.0 | Fit | ⭐⭐⭐⭐ |
| 22 | w_lag | day | 0.7 | Fit | ⭐⭐⭐ |
| 23 | k_diss_gyp | day⁻¹ | 0.12 | Kayaç | ⭐⭐⭐⭐⭐ |
| 24 | beta_SO4_m | mM⁻¹ | 0.10 | Fit | ⭐⭐⭐⭐ |
| 25 | phi_H2 | - | 1.00 | Fit | ⭐⭐⭐⭐ |
| 26 | phi_CO2 | - | 1.00 | Fit | ⭐⭐⭐⭐ |
| 27 | phi_H2S | - | 1.00 | Fit | ⭐⭐⭐⭐⭐ |
| 28 | alpha_H2S | - | 1.00 | Fit | ⭐⭐⭐⭐⭐ |

---

**Hazırlayan**: Hasan Arı  
**Tarih**: 6 Ocak 2026  
**Kaynak**: Kod satır 139-201 (parametre tanımları)
