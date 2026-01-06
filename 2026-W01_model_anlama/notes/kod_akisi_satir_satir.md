# MATLAB Kodu - Satır Satır Açıklama

**Dosya**: `anaerobic_model_two_phase_mixedSR_25C_v4.m`  
**Amaç**: Kod mantığını anlamak (kimya değil!)  
**Tarih**: 6 Ocak 2026

---

## Ana Fonksiyon Yapısı

```matlab
function anaerobic_model_two_phase_mixedSR_25C_v4
    % 1. Setup (satır 1-130)
    % 2. Fitting (satır 131-140)
    % 3. Simulation (satır 141-150)
    % 4. Post-processing (satır 151-300)
end
```

---

## BÖLÜM 1: SETUP (Satır 1-130)

### Satır 24-27: Sabitler
```matlab
Vg = 0.14;   % headspace volume [L] ~ 140 mL
Vl = 0.015;  % liquid volume   [L] ~ 15 mL
T  = 298.15; % K (25°C)
R_gas = 0.082057; % L·atm/(mol·K)
```

**Ne yapıyor?**
- `Vg`, `Vl`: Hacim sabitleri (ODE'de kullanılacak)
- `T`, `R_gas`: Sıcaklık ve gaz sabiti (ideal gaz yasası için)

---

### Satır 29-40: Henry Sabitleri

```matlab
% Henry solubility constants Hscp (c/p) @ 25°C, mmol/L/atm
Hcp_H2_base  = 0.78;  % H2  (25 °C) (mmol/L/atm)
Hcp_CO2_base = 34.0;  % CO2 (25 °C) (mmol/L/atm)
Hcp_H2S_base = 90.0;  % H2S (25 °C) (mmol/L/atm)

% Scale factors (fit edilecek)
phi_H2  = 1.00;
phi_CO2 = 1.00;

% Effective Henry constants
Hcp_H2_eff  = phi_H2  * Hcp_H2_base;
Hcp_CO2_eff = phi_CO2 * Hcp_CO2_base;
Hcp_H2S_eff = Hcp_H2S_base;
```

**Ne yapıyor?**
- `Hcp_*_base`: Sabit değerler (literatürden)
- `phi_*`: Çarpanlar (başta 1.0, sonra fit edilecek)
- `Hcp_*_eff`: Kullanılacak nihai değer = base × phi

**Neden?**
- Fit sırasında `phi_H2`, `phi_CO2` ayarlanacak
- Henry sabitlerini ±15% hareket ettirebilmek için

---

### Satır 42-50: Deneysel Veri Okuma

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

**Ne yapıyor?**
- `.txt` dosyasını oku → `raw` matrix
- Kolonları ayır → `t_exp`, `nH2_g_exp`, vb.
- `/1000`: µmol → mmol çevir
- `data_exp`: Fit'te kullanılacak target değerler (5 kolon)

**Önemli:**
- `t_exp`: Zaman noktaları (örnek: [0, 1, 2, ..., 20] gün)
- `data_exp`: Her zaman noktasında ölçülen değerler

---

### Satır 52-53: pH İnterpolasyonu

```matlab
% pH interpolant
pH_fun = @(t) max(0, interp1(t_exp, pH_exp, t, 'linear', 'extrap'));
```

**Ne yapıyor?**
- Anonymous function: `pH_fun(t)` → pH değeri döndürür
- `interp1`: Linear interpolasyon
  - `t_exp` noktalarında `pH_exp` değerleri var
  - Herhangi bir `t` için interpolate et
- `max(0, ...)`: Negatif pH olmasın

**Neden?**
- ODE solver her timestep'te farklı `t` soracak
- `pH_fun(3.5)` → pH at t=3.5 gün

---

### Satır 55-60: Başlangıç Dengesi

```matlab
pH2  = (nH2_g_exp(1)/1000)  * R_gas * T / Vg; % atm
pCO2 = (nCO2_g_exp(1)/1000) * R_gas * T / Vg; % atm
pH2S = (nH2S_g_exp(1)/1000) * R_gas * T / Vg; % atm

H2_aq0  = Hcp_H2_eff  * pH2;   % mmol/L
CO2_aq0 = Hcp_CO2_eff * pCO2;  % mmol/L
```

**Ne yapıyor?**
1. **İdeal gaz yasası**: `p = n*R*T/V`
   - `nH2_g_exp(1)` mmol → `/1000` → mol
   - `p_H2` atm hesapla
2. **Henry yasası**: `C_eq = Hcp * p`
   - Dengede olması gereken sıvı konsantrasyonu

**Neden?**
- ODE başlangıç koşulu: `y0` için `H2_aq0`, `CO2_aq0` gerekli

---

### Satır 62-66: Diagnostics (İsteğe Bağlı)

```matlab
Ptot_0 = ((nH2_g_exp(1)+nCO2_g_exp(1)+nCH4_g_exp(1)+nH2S_g_exp(1))/1000) * R_gas * T / Vg;
fprintf('\n[Henry/Pressure @ t0] P_tot=%.3f atm \n p_H2=%.3f, p_CO2=%.3f, p_H2S=%.4f atm \n Ceq: H2=%.4f, CO2=%.3f mmol/L\n', ...
        Ptot_0, pH2, pCO2, pH2S, H2_aq0, CO2_aq0);
```

**Ne yapıyor?**
- Toplam basınç hesapla
- Ekrana yazdır (debugging için)

**Çıktı örneği:**
```
[Henry/Pressure @ t0] P_tot=2.150 atm 
 p_H2=1.850, p_CO2=0.280, p_H2S=0.0050 atm 
 Ceq: H2=1.4430, CO2=9.520 mmol/L
```

---

### Satır 68-77: Başlangıç Durumu (y0)

```matlab
S_tot0   = 1;  % tiny sulfide seed
Fe_pool0 = 0.10;  % initial dissolved Fe(II) pool (mmol/L)

y0 = [ nH2_g_exp(1), nCO2_g_exp(1), nCH4_g_exp(1), nH2S_g_exp(1), ...
       H2_aq0, CO2_aq0, SO4_exp(1), ...
       0.01, 0.01, 0, 0, S_tot0, 0, ...
       Fe_pool0 ];
```

**Ne yapıyor?**
- 14 elemanlı vector oluştur:
  - `y0(1:4)`: Gaz fazı (deneyden oku)
  - `y0(5:6)`: Sıvı H2, CO2 (Henry'den hesapla)
  - `y0(7)`: SO4 (deneyden oku)
  - `y0(8:13)`: Küçük seed değerler (0.01, 0, vb.)
  - `y0(14)`: Fe_pool başlangıç değeri

**Neden?**
- `ode15s(odes, [0 t_end], y0, opts)` → `y0` gerekli

---

### Satır 79-100: Parametre Vektörü (p0, lb, ub)

```matlab
p0 = [ 0.06, 0.08, 0.03, ...  % k_m, k_s, k_a
       0.06, 0.05, 0.05, ...  % Y_m, Y_s, Y_a
       ...
       1.00 ];                % alpha_H2S (28 parametre)

lb = [ 1e-4, 1e-4, 1e-4, ...  % Lower bounds
       ...
       0.70];

ub = [ 5, 5, 5, ...           % Upper bounds
       ...
       3.00 ];
```

**Ne yapıyor?**
- `p0`: Initial guess (28 eleman)
- `lb`: Lower bounds (alt sınır)
- `ub`: Upper bounds (üst sınır)

**Neden?**
- Fitting algoritması bu aralıkta arar
- `lsqnonlin(fun, p0, lb, ub, opts)`

---

### Satır 102: Fitting Options

```matlab
fit_opts = optimoptions('lsqnonlin','Display','iter','MaxFunctionEvaluations',6000);
```

**Ne yapıyor?**
- `Display','iter'`: Her iterasyonu ekrana yaz
- `MaxFunctionEvaluations',6000`: Max 6000 fonksiyon çağrısı

**Çıktı örneği:**
```
Iteration  Func-count     Residual         Step-size       optimality
    0         29           1.23456e+02                      5.67e+01
    1         58           8.91234e+01      1.234e+00       3.45e+01
    ...
```

---

### Satır 104-112: Environment Struct

```matlab
env.Vg = Vg; env.Vl = Vl; env.T = T; env.Rgas = R_gas;
env.Hcp_H2_eff  = Hcp_H2_eff;
env.Hcp_CO2_eff = Hcp_CO2_eff;
env.Hcp_H2S_eff = Hcp_H2S_eff;
env.pH_fun      = pH_fun;
env.pKa_H2S     = 7.05;
env.SO4_sat_gyp = 15.0;
```

**Ne yapıyor?**
- Struct oluştur: `env.Vg`, `env.Vl`, vb.
- ODE fonksiyonuna pass edilecek

**Neden?**
- ODE fonksiyonu: `model_mixed(t, y, p, env)`
- `env` ile sabitler taşınır (global variable kullanmadan)

---

## BÖLÜM 2: FITTING (Satır 114-116)

```matlab
[p_fit,~,~,~,~,~,~] = lsqnonlin(@(p) residuals_full(p, t_exp, data_exp, y0, env), p0, lb, ub, fit_opts);
save('best_fit_params_Sandstone_25C.mat','p_fit','env','y0');
```

**Ne yapıyor?**

### Satır 114: lsqnonlin Çağrısı
```matlab
[p_fit, ...] = lsqnonlin(@(p) residuals_full(...), p0, lb, ub, fit_opts);
```

**Adım adım:**
1. `@(p)`: Anonymous function (p → residuals)
2. `residuals_full(p, ...)`: Objective function
   - Input: `p` (28 parametre)
   - Output: Residual vector (hata)
3. `lsqnonlin`: Least-squares optimizer
   - `p0` başlangıç → `p_fit` sonuç bulur
   - `lb`, `ub` sınırları zorlar

**Çıktı:**
- `p_fit`: En iyi 28 parametre

### Satır 115: Kaydet
```matlab
save('best_fit_params_Sandstone_25C.mat','p_fit','env','y0');
```

**Ne yapıyor?**
- `.mat` dosyasına yaz (MATLAB binary format)
- Daha sonra `load(...)` ile yüklenebilir

---

## BÖLÜM 3: FINAL SIMULATION (Satır 118-124)

```matlab
odes = @(t,y) model_mixed(t,y,p_fit,env);
opts = odeset('NonNegative',1:14,'RelTol',1e-8,'AbsTol',1e-10,'MaxStep',0.5);
[t_sim, y_sim] = ode15s(odes, [0 t_exp(end)], y0, opts);
```

**Ne yapıyor?**

### Satır 118: ODE Function Handle
```matlab
odes = @(t,y) model_mixed(t,y,p_fit,env);
```
- Anonymous function: `(t, y)` alır → `dydt` döner
- `p_fit`: Fitted parametreler artık sabit

### Satır 119: ODE Options
```matlab
opts = odeset('NonNegative',1:14, ...);
```
- `NonNegative',1:14`: Tüm 14 state ≥ 0 zorla
- `RelTol',1e-8`: Relative tolerance (hassaslık)
- `AbsTol',1e-10`: Absolute tolerance
- `MaxStep',0.5`: Max timestep = 0.5 gün

### Satır 120: ODE Solver
```matlab
[t_sim, y_sim] = ode15s(odes, [0 t_exp(end)], y0, opts);
```
- `ode15s`: Stiff ODE solver
- `[0 t_exp(end)]`: t = 0 → t_final (örnek: 20 gün)
- Output:
  - `t_sim`: Zaman vektörü (dense, örnek: 500 nokta)
  - `y_sim`: State matrix (500 × 14)

---

## BÖLÜM 4: POST-PROCESSING (Satır 126-300)

### Satır 126-127: pH Speciation

```matlab
[H2S_aq, HS_aq] = speciate_sulfide(y_sim(:,12), env.pH_fun(t_sim), env.pKa_H2S);
```

**Ne yapıyor?**
- `y_sim(:,12)`: State 12 = S_tot (tüm timesteps)
- `env.pH_fun(t_sim)`: Her timestep için pH
- `speciate_sulfide`: Helper function
  - Input: `S_tot`, `pH`, `pKa`
  - Output: `H2S_aq`, `HS_aq` (pH'ya göre böl)

**Kod (satır 405-410):**
```matlab
function [H2S_aq, HS_aq] = speciate_sulfide(S_tot, pH, pKa)
frac_HS  = 1 ./ (1 + 10.^(pKa - pH));
HS_aq    = S_tot .* frac_HS;
H2S_aq   = S_tot - HS_aq;
end
```

**Mantık:**
- `frac_HS`: HS⁻ fraksiyonu (0-1 arası)
- Element-wise operations (`.` operatörü)

---

### Satır 129-164: Sulfur Mass Balance Diagnostic

```matlab
S_gas_mmol = y_sim(:,4);                 % nH2S_g
S_aq_mmol  = y_sim(:,12) * env.Vl;       % S_tot * Vl
S_FeS_mmol = y_sim(:,8)  * env.Vl;       % FeS * Vl

S_total_model = S_gas_mmol + S_aq_mmol + S_FeS_mmol;
```

**Ne yapıyor?**
- Kütle dengesi check:
  - Gaz fazında: `nH2S_g` (mmol)
  - Sıvıda: `S_tot × Vl` (mmol/L → mmol)
  - Katıda: `FeS × Vl` (mmol/L → mmol)
- Toplam: `S_total_model`

**Sonra:**
```matlab
rates_over = zeros(length(t_sim),4);
for k = 1:length(t_sim)
    rates_over(k,:) = rate_out_mixed(t_sim(k), y_sim(k,:), p_fit, env);
end
r_sulf_vec   = rates_over(:,2);  % Sulfate reduction rate
S_prod_cum   = cumtrapz(t_sim, r_sulf_vec) * env.Vl;  % Cumulative production
```

**Ne yapıyor?**
1. Loop: Her timestep için reaction rate hesapla
2. `r_sulf_vec`: Sülfat indirgeme hızı (mmol/L/day)
3. `cumtrapz`: Kümülatif integral (üretilen toplam sülfür)

**Expected:**
```matlab
S_total_expected = S_total0 + S_prod_cum;
```

**Plot:**
```matlab
figure; subplot(2,1,1);
plot(t_sim, S_total_model, 'b-', t_sim, S_total_expected, 'r--');
legend('Model', 'Expected');
```

**Neden?**
- Eğer mavi ve kırmızı çakışırsa → Kütle korunuyor ✅
- Ayrılırsa → Kod hatası var ❌

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
- `H2S_aq`: Sıvı H2S konsantrasyonu
- Henry yasası ile dengede olması gereken gaz miktarı hesapla
- Karşılaştır:
  - Mavi: Model (ODE'den gelen `nH2S_g`)
  - Kırmızı: Henry'den beklenen

**Neden?**
- Eğer ayrılırsa → `kla_H2S` veya `phi_H2S` yanlış

---

### Satır 183-189: Reaction Rates (Plotting için)

```matlab
rates = zeros(length(t_sim), 4);
for i = 1:length(t_sim)
    rates(i,:) = rate_out_mixed(t_sim(i), y_sim(i,:), p_fit, env);
end
```

**Ne yapıyor?**
- Her timestep için rate hesapla
- `rates`: 4 kolonlu matrix
  - `rates(:,1)`: r_meth (methanogenesis)
  - `rates(:,2)`: r_sulf (sulfate reduction)
  - `rates(:,3)`: r_prec (precipitation)
  - `rates(:,4)`: r_aceto (homoacetogenesis)

---

### Satır 191-205: Write .dat File

```matlab
fileID = fopen('Sandstone_25C_inc_rates.dat','w');
fprintf(fileID, 'Time(days) nH2_g nCO2_g ... r_meth r_sulf r_prec r_aceto\n');

for i = 1:length(t_sim)
    fprintf(fileID, '%10.6f %12.6g %12.6g ... %12.6g %12.6g %12.6g %12.6g\n', ...
        t_sim(i), y_sim(i,1), y_sim(i,2), ..., rates(i,1), rates(i,2), rates(i,3), rates(i,4));
end
fclose(fileID);
```

**Ne yapıyor?**
- ASCII text dosyası yaz
- Format: Space-separated columns
- Her satır: 1 timestep (örnek: 500 satır)

**Çıktı örneği:**
```
Time(days) nH2_g nCO2_g nCH4_g nH2S_g H2_aq ... r_meth r_sulf r_prec r_aceto
0.000000   1.234e+02 5.678e+01 1.234e+00 ...  0.0123 0.0456 0.0001 0.0012
0.040812   1.233e+02 5.677e+01 1.240e+00 ...  0.0125 0.0460 0.0001 0.0013
...
```

---

### Satır 207-224: Plotting (Figure 1)

```matlab
species = {'nH2_g','nCO2_g','nCH4_g','nH2S_g','H2(aq)', ...};
figure('Name','Gases & Aqueous - Sandstone (25 °C)');
for i = 1:length(species)
    subplot(7,2,i)
    if i <= 4
        plot(t_exp, data_exp(:,i), 'ko', 'DisplayName','Exp'); hold on;
        plot(t_sim, y_sim(:,i), 'b-', 'DisplayName','Model');
        ylabel('mmol (gas)'); xlabel('days');
    elseif i == 7
        plot(t_exp, data_exp(:,5), 'ko'); hold on;
        plot(t_sim, y_sim(:,7), 'b-');
        ylabel('mmol/L'); xlabel('days');
    else
        plot(t_sim, y_sim(:,i), 'b-');
        ylabel('mmol/L'); xlabel('days');
    end
    title(species{i}); legend;
end
```

**Ne yapıyor?**
- 14 subplot (7×2 grid)
- Her subplot: 1 state variable
- İlk 4: Deneysel data ile karşılaştır (siyah 'o')
- Diğerleri: Sadece model (mavi çizgi)

---

### Satır 240-256: RMSE Hesaplama

```matlab
yH2_on_exp  = interp1(t_sim, y_sim(:,1), t_exp, 'linear', 'extrap');
yCO2_on_exp = interp1(t_sim, y_sim(:,2), t_exp, 'linear', 'extrap');
fprintf('RMSE (gas moles on t_exp): H2=%.4f mmol CO2=%.4f mmol\n', ...
        rmse_equal(yH2_on_exp, data_exp(:,1)), rmse_equal(yCO2_on_exp, data_exp(:,2)));
```

**Ne yapıyor?**
1. `interp1`: Model output (`t_sim`) → deneysel noktalar (`t_exp`) interpolate et
2. `rmse_equal`: RMSE hesapla

**Helper function (satır 412-414):**
```matlab
function r = rmse_equal(a,b)
r = sqrt(mean((a(:)-b(:)).^2,'omitnan'));
end
```

**Çıktı örneği:**
```
RMSE (gas moles on t_exp): H2=2.3456 mmol CO2=1.2345 mmol
```

---

## Helper Functions

### 1. residuals_full (Satır 258-276)

```matlab
function res = residuals_full(p, t_exp, data_exp, y0, env)
odes = @(t,y) model_mixed(t,y,p,env);
opts = odeset('NonNegative',1:14,'RelTol',1e-8,'AbsTol',1e-10,'MaxStep',0.5);
[~, y_sim] = ode15s(odes, t_exp, y0, opts);

sim_mat = [y_sim(:,1), y_sim(:,2), y_sim(:,3), y_sim(:,4), y_sim(:,7)];
log_sim = log1p(sim_mat);
log_exp = log1p([data_exp(:,1), data_exp(:,2), data_exp(:,3), data_exp(:,4), data_exp(:,5)]);

weights = [1, 1, 0.9, 1.0, 2.0];
res = (log_sim - log_exp) .* weights;

if any(y_sim(:) < -1e-9), res = res + 1e3 * abs(min(y_sim(:))); end
res = res(:);
end
```

**Ne yapıyor?**

#### Satır 260-261: ODE Çöz
```matlab
[~, y_sim] = ode15s(odes, t_exp, y0, opts);
```
- `t_exp` noktalarında çöz (dense grid değil!)
- `y_sim`: `length(t_exp) × 14` matrix

#### Satır 263-265: Log Transform + Weights
```matlab
log_sim = log1p(sim_mat);  % log(1 + x)
log_exp = log1p(data_exp);
res = (log_sim - log_exp) .* weights;
```

**Neden log1p?**
- Küçük ve büyük değerleri normalize eder
- `log1p(x) = log(1+x)` → x=0 için güvenli

**Weights:**
- `[1, 1, 0.9, 1.0, 2.0]` → H2, CO2, CH4, H2S, SO4
- SO4 weight = 2.0 → Daha önemli!

#### Satır 273-274: Penalty for Negative States
```matlab
if any(y_sim(:) < -1e-9)
    res = res + 1e3 * abs(min(y_sim(:)));
end
```

**Neden?**
- Eğer state < 0 olursa → Fiziksel olarak invalid
- Büyük penalty ekle → Optimizer o parametreyi kullanmasın

#### Satır 275: Reshape
```matlab
res = res(:);
```
- Matrix → Vector (lsqnonlin vector ister)

---

### 2. model_mixed (ODE Function) (Satır 278-360)

```matlab
function dydt = model_mixed(t, y, p, env)
```

**Signature:**
- Input: `t` (scalar time), `y` (14×1 vector), `p` (28×1 params), `env` (struct)
- Output: `dydt` (14×1 derivatives)

**Yapısı:**
1. **Unpack** (satır 279-287): `env`, `y`, `p` değişkenlerini al
2. **Guards** (satır 312-315): Negatif değerleri `eps` ile değiştir
3. **Partial Pressures** (satır 317-321): İdeal gaz yasası
4. **Henry Equilibrium** (satır 323-327): `Ceq = Hcp * p`
5. **Mass Transfer** (satır 329-335): `J = kla * (Ceq - C_aq)`
6. **Speciation** (satır 337-341): pH bağımlı
7. **Inhibitions** (satır 343-349): Monod + inhibisyon
8. **Rates** (satır 351-354): Reaction rates hesapla
9. **Derivatives** (satır 356-360): `dydt` vektörü oluştur

#### Örnek Kod Bloğu: Partial Pressures

```matlab
% Satır 317-321
pH2  = (nH2_g /1000)  * Rgas * T / Vg;
pCO2 = (nCO2_g/1000)  * Rgas * T / Vg;
pH2S = (nH2S_g/1000)  * Rgas * T / Vg;
```

**Mantık:**
- `nH2_g` (mmol) → `/1000` → mol
- `p = n*R*T/V` (ideal gaz)

#### Örnek Kod Bloğu: Mass Transfer

```matlab
% Satır 329-331
J_H2  = kla_H2  * (Ceq_H2  - H2_aq);
J_CO2 = kla_CO2 * (Ceq_CO2 - CO2_aq);
```

**Mantık:**
- Eğer `H2_aq < Ceq_H2` → `J_H2 > 0` → Gaz çözünüyor
- Eğer `H2_aq > Ceq_H2` → `J_H2 < 0` → Gaz uçuyor

#### Örnek Kod Bloğu: Derivatives

```matlab
% Satır 356-360
dnH2_g  = - J_H2  * Vl;
dnCO2_g = - J_CO2 * Vl;
dnCH4_g = + r_meth * Vl;
dnH2S_g = + Jout_H2S * Vl;

dH2_aq  = + J_H2  - 4*r_meth - 4*r_sulf - 4*r_aceto;
dCO2_aq = + J_CO2 - 1*r_meth - 2*r_aceto;
...
dydt = [dnH2_g; dnCO2_g; dnCH4_g; dnH2S_g; dH2_aq; dCO2_aq; ...];
```

**Mantık:**
- Gaz balansı: `J_H2 * Vl` (mmol/L/day × L → mmol/day)
- Sıvı balansı: `J_H2 - 4*r_meth - ...` (mmol/L/day)
- Son satır: 14 derivative'i birleştir

---

### 3. rate_out_mixed (Diagnostic Rates) (Satır 362-403)

```matlab
function dr = rate_out_mixed(t, y, p, env)
```

**Amaç:**
- Plotting için reaction rates hesapla
- ODE içindeki rate hesaplamalarını tekrar yap

**Neden ayrı fonksiyon?**
- ODE içinde rates hesaplanıyor ama return edilmiyor
- Post-processing için tekrar lazım

**Çıktı:**
```matlab
dr = [r_meth, r_sulf, r_prec, r_aceto];  % 4-element vector
```

---

## Kod Akış Diyagramı

```
MAIN FUNCTION
│
├─ SETUP (satır 1-113)
│  ├─ Read experimental data (.txt)
│  ├─ Initialize y0 (14 states)
│  ├─ Initialize p0, lb, ub (28 params)
│  └─ Create env struct
│
├─ FITTING (satır 114-116)
│  ├─ Call lsqnonlin
│  │   ├─ Calls residuals_full
│  │   │   ├─ Solves ODE at t_exp
│  │   │   │   └─ Calls model_mixed (ODE function)
│  │   │   ├─ Computes log residuals
│  │   │   └─ Returns residual vector
│  │   └─ Finds p_fit (best params)
│  └─ Save p_fit to .mat
│
├─ SIMULATION (satır 118-124)
│  ├─ Solve ODE with p_fit (dense grid)
│  │   └─ Calls model_mixed
│  └─ Get t_sim, y_sim
│
└─ POST-PROCESSING (satır 126-256)
   ├─ Speciation (H2S_aq, HS_aq)
   ├─ Mass balance diagnostics
   ├─ Compute reaction rates
   ├─ Write .dat file
   ├─ Plot figures
   └─ Print RMSE
```

---

## Önemli Kod Kalıpları

### 1. Anonymous Function

```matlab
pH_fun = @(t) interp1(t_exp, pH_exp, t, 'linear', 'extrap');
odes   = @(t,y) model_mixed(t,y,p_fit,env);
```

**Ne işe yarar?**
- Fonksiyon handle oluşturur
- Parametreleri "dondurmak" için kullanılır

### 2. Element-wise Operations

```matlab
frac_HS = 1 ./ (1 + 10.^(pKa - pH));  % ./ ve .^ dikkat!
```

**Neden `.` gerekli?**
- `pH` vector ise → element-wise böl
- `./` yerine `/` kullanırsan matrix division olur (hata!)

### 3. Struct Usage

```matlab
env.Vg = Vg;
env.pH_fun = pH_fun;

% Sonra:
Vg_value = env.Vg;
pH_value = env.pH_fun(3.5);
```

**Neden?**
- Global variable'dan daha temiz
- Fonksiyonlar arası veri taşıma

### 4. Loop with Pre-allocation

```matlab
rates = zeros(length(t_sim), 4);  % Pre-allocate
for i = 1:length(t_sim)
    rates(i,:) = rate_out_mixed(t_sim(i), y_sim(i,:), p_fit, env);
end
```

**Neden pre-allocate?**
- MATLAB loop'ta matrix büyütmek çok yavaş
- `zeros` ile önce yer ayır → Hızlı!

---

## Python'a Çevirirken Dikkat!

### 1. Indexing
```matlab
y(1)      % MATLAB: 1-indexed
y_sim(:,1) % All rows, column 1
```
```python
y[0]       # Python: 0-indexed
y_sim[:,0] # All rows, column 0
```

### 2. ODE Solver
```matlab
[t, y] = ode15s(@odefun, tspan, y0, opts);
```
```python
from scipy.integrate import solve_ivp
sol = solve_ivp(odefun, tspan, y0, method='BDF', **options)
t, y = sol.t, sol.y.T
```

### 3. Optimization
```matlab
p_fit = lsqnonlin(@resfun, p0, lb, ub, opts);
```
```python
from scipy.optimize import least_squares
result = least_squares(resfun, p0, bounds=(lb, ub), **opts)
p_fit = result.x
```

### 4. Element-wise
```matlab
frac = 1 ./ (1 + 10.^(pKa - pH));
```
```python
frac = 1 / (1 + 10**(pKa - pH))  # NumPy broadcasts
```

---

**Hazırlayan**: Hasan Arı  
**Tarih**: 6 Ocak 2026  
**Amaç**: Kod mantığını anlamak (kimya değil!)
