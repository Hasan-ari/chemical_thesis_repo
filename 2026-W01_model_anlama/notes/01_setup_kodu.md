# Setup Kısmı - Satır Satır Kod Açıklaması

**Bölüm**: Satır 1-113 (Setup)
**Amaç**: Veri okuma, başlangıç koşulları, parametre tanımlama

---

## Satır 10: Ana Fonksiyon Tanımı

```matlab
function anaerobic_model_two_phase_mixedSR_25C_v4
```

**Ne yapıyor?**
- MATLAB fonksiyonu tanımla
- Input yok, output yok (self-contained)
- Çalıştırmak için: `anaerobic_model_two_phase_mixedSR_25C_v4` yaz

---

## Satır 24-27: Sabitler

```matlab
Vg = 0.14;   % headspace volume [L] ~ 140 mL
Vl = 0.015;  % liquid volume   [L] ~ 15 mL
T  = 298.15; % K (25°C)
R_gas = 0.082057; % L·atm/(mol·K)
```

**Satır satır:**
- **24**: `Vg = 0.14` → Gaz fazı hacmi (Litre), ODE'de kullanılacak
- **25**: `Vl = 0.015` → Sıvı fazı hacmi (Litre)
- **26**: `T = 298.15` → Sıcaklık (Kelvin), 25°C = 298.15 K
- **27**: `R_gas = 0.082057` → Gaz sabiti, `p*V = n*R*T` için gerekli

**Neden gerekli?**
- Gaz-sıvı dönüşümlerde kullanılacak (Henry yasası)
- ODE içinde `env` struct ile pass edilecek

---

## Satır 30-40: Henry Sabitleri

```matlab
Hcp_H2_base  = 0.78;  % H2  (25 °C) (mmol/L/atm)
Hcp_CO2_base = 34.0;  % CO2 (25 °C) (mmol/L/atm)
Hcp_H2S_base = 90.0;  % H2S (25 °C) (mmol/L/atm)

phi_H2  = 1.00;
phi_H2 = 1.00;

Hcp_H2_eff  = phi_H2  * Hcp_H2_base;
Hcp_CO2_eff = phi_CO2 * Hcp_CO2_base;
Hcp_H2S_eff = Hcp_H2S_base;
```

**Satır satır:**
- **30-32**: Base değerler (sabit, literatürden)
  - `Hcp_H2_base = 0.78` → H₂ çözünürlük sabiti
  - `Hcp_CO2_base = 34.0` → CO₂ çözünürlük sabiti
  - `Hcp_H2S_base = 90.0` → H₂S çözünürlük sabiti
- **35-36**: Scale faktörleri (fit sırasında değişecek)
  - `phi_H2 = 1.00` → Başlangıç çarpanı (±15% değişebilir)
  - `phi_CO2 = 1.00` → Başlangıç çarpanı
- **38-40**: Effective değerler (kullanılacak)
  - `Hcp_H2_eff = phi_H2 * Hcp_H2_base` → Base × phi
  - `Hcp_CO2_eff = phi_CO2 * Hcp_CO2_base`
  - `Hcp_H2S_eff = Hcp_H2S_base` → H2S için phi yok (başta)

**Kod mantığı:**
- Henry: `C_eq [mmol/L] = Hcp [mmol/L/atm] × p [atm]`
- Phi ile fine-tune yapabilmek için ayrıldı

---

## Satır 42-50: Deneysel Veri Okuma

```matlab
raw = readmatrix('Muller_2024_H2_Sandstone_at_25C.txt');
t_exp       = raw(:,1);  % days
nH2_g_exp   = raw(:,2) / 1000; % µmol -> mmol
nCO2_g_exp  = raw(:,3) / 1000; % µmol -> mmol
nCH4_g_exp  = raw(:,4) / 1000; % µmol -> mmol
nH2S_g_exp  = raw(:,5) / 1000; % µmol -> mmol
pH_exp      = raw(:,6);        % pH
SO4_exp     = raw(:,7);        % sulfate mM (mmol/L)
data_exp    = [nH2_g_exp, nCO2_g_exp, nCH4_g_exp, nH2S_g_exp, SO4_exp];
```

**Satır satır:**
- **42**: `readmatrix(...)` → .txt dosyasını oku, matrix döndür
  - `raw`: N×7 matrix (N satır, 7 kolon)
- **43**: `t_exp = raw(:,1)` → İlk kolon = zaman (gün)
  - `:` → Tüm satırlar, `1` → 1. kolon
- **44-47**: Gaz miktarları (µmol → mmol)
  - `raw(:,2)` → 2. kolon (H₂ µmol)
  - `/1000` → µmol → mmol çevir
  - `nH2_g_exp`, `nCO2_g_exp`, ... → Vector'ler
- **48-49**: pH ve SO4 (birim dönüşümü yok)
  - `pH_exp = raw(:,6)` → pH (0-14 arası)
  - `SO4_exp = raw(:,7)` → Sülfat (mmol/L)
- **50**: Target data matrix oluştur
  - `data_exp = [nH2_g, nCO2_g, nCH4_g, nH2S_g, SO4]`
  - 5 kolon → Fit'te bu değerlere match edecek

**Örnek veri:**
```
t_exp = [0; 1; 2; 3; ...]  (gün)
nH2_g_exp = [150; 140; 130; ...]  (mmol)
```

---

## Satır 52-53: pH İnterpolasyon Fonksiyonu

```matlab
pH_fun = @(t) max(0, interp1(t_exp, pH_exp, t, 'linear', 'extrap'));
```

**Satır satır:**
- `@(t)` → Anonymous function (input: `t`, output: pH değeri)
- `interp1(t_exp, pH_exp, t, 'linear', 'extrap')`:
  - `t_exp`: Bilinen zaman noktaları (örnek: [0, 1, 2, ..., 20])
  - `pH_exp`: Bilinen pH değerleri (örnek: [7.2, 7.1, 7.0, ...])
  - `t`: İstenen zaman (örnek: 3.5)
  - `'linear'`: Linear interpolasyon kullan
  - `'extrap'`: Eğer `t` aralık dışında → Extrapolate et
- `max(0, ...)`: pH < 0 olmasın (güvenlik)

**Kullanım:**
```matlab
pH_at_3_5_days = pH_fun(3.5);  % Örnek: 7.05
```

**Neden gerekli?**
- ODE solver her timestep'te farklı `t` soracak
- `t_exp` noktalarında değil, sürekli `t` için pH lazım

---

## Satır 55-60: Başlangıç Gaz Basınçları ve Sıvı Konsantrasyonları

```matlab
pH2  = (nH2_g_exp(1)/1000)  * R_gas * T / Vg; % atm
pCO2 = (nCO2_g_exp(1)/1000) * R_gas * T / Vg; % atm
pH2S = (nH2S_g_exp(1)/1000) * R_gas * T / Vg; % atm

H2_aq0  = Hcp_H2_eff  * pH2;   % mmol/L
CO2_aq0 = Hcp_CO2_eff * pCO2;  % mmol/L
```

**Satır satır:**

### İdeal Gaz Yasası (satır 55-57)
```matlab
pH2 = (nH2_g_exp(1)/1000) * R_gas * T / Vg;
```
- `nH2_g_exp(1)` → İlk zaman noktasındaki H₂ (mmol)
- `/1000` → mmol → mol
- `* R_gas * T / Vg` → `p = n*R*T/V` (ideal gaz)
- Sonuç: `pH2` (atm)

### Henry Dengesi (satır 59-60)
```matlab
H2_aq0 = Hcp_H2_eff * pH2;
```
- `Hcp_H2_eff` → 0.78 mmol/L/atm
- `pH2` → ~1.5 atm (örnek)
- `H2_aq0 = 0.78 × 1.5 = 1.17` mmol/L

**Neden gerekli?**
- ODE başlangıç koşulu: `y0(5) = H2_aq0`, `y0(6) = CO2_aq0`

---

## Satır 62-66: Diagnostics (Debug için)

```matlab
Ptot_0 = ((nH2_g_exp(1)+nCO2_g_exp(1)+nCH4_g_exp(1)+nH2S_g_exp(1))/1000) * R_gas * T / Vg;
fprintf('\n[Henry/Pressure @ t0] P_tot=%.3f atm \n p_H2=%.3f, p_CO2=%.3f, p_H2S=%.4f atm \n Ceq: H2=%.4f, CO2=%.3f mmol/L\n', ...
        Ptot_0, pH2, pCO2, pH2S, H2_aq0, CO2_aq0);
```

**Satır satır:**
- **62-63**: Toplam basınç hesapla
  - `(nH2 + nCO2 + nCH4 + nH2S) / 1000` → Toplam mol
  - `* R*T/V` → Toplam basınç (atm)
- **64-66**: Ekrana yazdır
  - `fprintf(...)` → Formatlanmış print
  - `%.3f` → 3 ondalık basamak
  - `\n` → Yeni satır

**Çıktı örneği:**
```
[Henry/Pressure @ t0] P_tot=2.150 atm
 p_H2=1.850, p_CO2=0.280, p_H2S=0.0050 atm
 Ceq: H2=1.4430, CO2=9.520 mmol/L
```

---

## Satır 68-77: Başlangıç Durumu Vektörü (y0)

```matlab
S_tot0   = 1;  % tiny sulfide seed
Fe_pool0 = 0.10;  % initial dissolved Fe(II) pool (mmol/L)

y0 = [ nH2_g_exp(1), nCO2_g_exp(1), nCH4_g_exp(1), nH2S_g_exp(1), ...
       H2_aq0, CO2_aq0, SO4_exp(1), ...
       0.01, 0.01, 0, 0, S_tot0, 0, ...
       Fe_pool0 ];
```

**Satır satır:**
- **73**: `S_tot0 = 1` → Başlangıç sülfür (mmol/L), seed value
- **74**: `Fe_pool0 = 0.10` → Başlangıç Fe²⁺ (mmol/L)
- **76-79**: 14 elemanlı vector oluştur
  - `y0(1:4)`: Gaz fazı (deneyden oku)
    - `nH2_g_exp(1)`, `nCO2_g_exp(1)`, `nCH4_g_exp(1)`, `nH2S_g_exp(1)`
  - `y0(5:6)`: Sıvı H2, CO2 (Henry'den hesapla)
    - `H2_aq0`, `CO2_aq0`
  - `y0(7)`: SO4 (deneyden oku)
    - `SO4_exp(1)`
  - `y0(8:13)`: Küçük seed değerler
    - `0.01` (FeS), `0.01` (X), `0` (Acetate), `0` (HCO3), `1` (S_tot), `0` (Lag)
  - `y0(14)`: Fe pool
    - `Fe_pool0 = 0.10`

**Örnek:**
```matlab
y0 = [150, 50, 2, 1, 1.17, 9.5, 10, 0.01, 0.01, 0, 0, 1, 0, 0.10]
     % [nH2_g, nCO2_g, nCH4_g, nH2S_g, H2_aq, CO2_aq, SO4, FeS, X, Ac, HCO3, S_tot, Lag, Fe_pool]
```

---

## Satır 79-100: Parametre Vektörü (p0, lb, ub)

```matlab
p0 = [ 0.06, 0.08, 0.03, ...  % k_m, k_s, k_a (3)
       0.06, 0.05, 0.05, ...  % Y_m, Y_s, Y_a (3)
       0.20, 0.20, 0.20, ...  % KI_m, KI_s, KI_a (3)
       0.02, 0.10, 0.02, -12, ...  % k_prec, HS_sat, H2_th, DG_th (4)
       0.50, 0.50, 0.80, ...  % K_H2, K_SO4, K_CO2 (3)
       10.0, 10.0, 25.0, ...  % kla_H2, kla_CO2, kla_H2S (3)
       0.01, 3.0, 0.7, ...    % b, t_lag, w_lag (3)
       0.12, 0.10, ...        % k_diss_gyp, beta_SO4_m (2)
       1.00, 1.00, 1.00, ...  % phi_H2, phi_CO2, phi_H2S (3)
       1.00 ];                % alpha_H2S (1)
                              % TOPLAM: 28 parametre
```

**Satır satır:**
- **80-88**: `p0` → Initial guess (28 eleman)
  - `p0(1:3)` → Kinetik hız sabitleri
  - `p0(4:6)` → Biyokütle verim
  - `p0(7:9)` → İnhibisyon sabitleri
  - `p0(10:13)` → Çökelme + aktivasyon
  - `p0(14:16)` → Monod sabitleri
  - `p0(17:19)` → kLa (kütle transferi)
  - `p0(20:22)` → Biyokütle decay + lag
  - `p0(23:24)` → Kayaç özel + rekabet
  - `p0(25:27)` → Henry faktörleri
  - `p0(28)` → H2S degassing faktörü

### Lower Bounds (satır 90-97)
```matlab
lb = [ 1e-4, 1e-4, 1e-4, ...  % k_m, k_s, k_a (min çok küçük)
       0.01, 0.01, 0.01, ...  % Y_m, Y_s, Y_a
       1e-3, 1e-3, 1e-3, ...  % KI_m, KI_s, KI_a
       0.0, 0.0, 0.0, -50, ...  % k_prec, HS_sat, H2_th, DG_th
       1e-3, 1e-3, 1e-3, ...  % K_H2, K_SO4, K_CO2
       0.1, 0.1, 0.1, ...     % kla_H2, kla_CO2, kla_H2S
       0, 0, 0.1, ...         % b, t_lag, w_lag
       0.01, 0.00, ...        % k_diss_gyp, beta_SO4_m
       0.85, 0.85, 0.90, ...  % phi_H2, phi_CO2, phi_H2S (±15%)
       0.70];                 % alpha_H2S
```

### Upper Bounds (satır 99-107)
```matlab
ub = [ 5, 5, 5, ...          % k_m, k_s, k_a (max)
       0.5, 0.5, 0.5, ...    % Y_m, Y_s, Y_a
       5, 5, 5, ...          % KI_m, KI_s, KI_a
       1.0, 5.0, 1.0, 0, ... % k_prec, HS_sat, H2_th, DG_th
       20, 20, 20, ...       % K_H2, K_SO4, K_CO2
       200, 200, 200, ...    % kla_H2, kla_CO2, kla_H2S
       0.2, 10, 2.0, ...     % b, t_lag, w_lag
       2.00, 1.00, ...       % k_diss_gyp, beta_SO4_m
       1.15, 1.15, 1.10, ... % phi_H2, phi_CO2, phi_H2S
       3.00 ];               % alpha_H2S
```

**Optimizer kullanımı:**
- `lsqnonlin` bu bounds içinde arayacak
- `p0` başlangıç → `p_fit` sonuç

---

## Satır 109: Fitting Options

```matlab
fit_opts = optimoptions('lsqnonlin','Display','iter','MaxFunctionEvaluations',6000);
```

**Satır satır:**
- `optimoptions(...)` → Options struct oluştur
- `'lsqnonlin'` → Hangi optimizer için
- `'Display','iter'` → Her iterasyonu ekrana bas
- `'MaxFunctionEvaluations',6000` → Max 6000 fonksiyon çağrısı

**Çıktı örneği:**
```
Iteration  Func-count     Residual         Step-size
    0         29          1.234e+02
    1         58          8.912e+01      1.23e+00
    2         87          6.543e+01      2.34e+00
    ...
   150       4350         1.234e+00      1.23e-03
```

---

## Satır 111-119: Environment Struct

```matlab
env.Vg = Vg; env.Vl = Vl; env.T = T; env.Rgas = R_gas;
env.Hcp_H2_eff  = Hcp_H2_eff;
env.Hcp_CO2_eff = Hcp_CO2_eff;
env.Hcp_H2S_eff = Hcp_H2S_eff;
env.pH_fun      = pH_fun;
env.pKa_H2S     = 7.05;
env.SO4_sat_gyp = 15.0;
```

**Satır satır:**
- **111**: Hacim ve sıcaklık sabitlerini struct'a koy
  - `env.Vg = Vg` → `env` struct'ının `Vg` field'ı
  - `env.Vl = Vl`, `env.T = T`, `env.Rgas = R_gas`
- **112-114**: Henry sabitleri
  - `env.Hcp_H2_eff`, `env.Hcp_CO2_eff`, `env.Hcp_H2S_eff`
- **115**: pH fonksiyonu
  - `env.pH_fun = pH_fun` → Function handle pass ediliyor
- **116**: pKa sabiti
  - `env.pKa_H2S = 7.05` → H₂S ↔ HS⁻ denge sabiti (25°C)
- **119**: Gypsum SO4 saturasyon seviyesi
  - `env.SO4_sat_gyp = 15.0` → Sandstone için 15 mM

**Neden struct?**
- ODE fonksiyonuna pass edilecek: `model_mixed(t, y, p, env)`
- Global variable kullanmadan temiz kod

**Kullanım:**
```matlab
% ODE içinde:
Vg_value = env.Vg;
pH_value = env.pH_fun(t);  % Function call
```

---

## Özet: Setup Kısmının Görevi

```
1. Sabitler tanımla (Vg, Vl, T, R_gas)
2. Henry sabitleri ayarla (Hcp_H2_eff, Hcp_CO2_eff, Hcp_H2S_eff)
3. Deneysel veriyi oku (.txt → matrix)
4. pH interpolasyon fonksiyonu oluştur (pH_fun)
5. Başlangıç gaz basınçlarını hesapla (ideal gaz yasası)
6. Başlangıç sıvı konsantrasyonlarını hesapla (Henry yasası)
7. y0 vektörünü oluştur (14 state başlangıç değerleri)
8. p0, lb, ub vektörlerini oluştur (28 parametre)
9. Fitting options ayarla (lsqnonlin için)
10. env struct oluştur (ODE'ye pass edilecek sabitler)
```

**Sonraki adım:** Bu setup ile `lsqnonlin` çağrılacak (Fitting bölümü)

---

**Hazırlayan**: Hasan Arı
**Tarih**: 6 Ocak 2026
**Amaç**: Setup kodunu satır satır anlamak
