# LatestCodes Klasörü - Detaylı Teknik Referans

## 1. Klasör Hiyerarşisi

```
LatestCodes/
│
├── Low_pressure_muller_2024_best_fits/
│   │
│   ├── Basalt/
│   │   ├── 25C/
│   │   │   ├── anaerobic_model_two_phase_mixedSR_25C_v4.m    [527 satır]
│   │   │   ├── best_fit_params_Basalt_25C.mat
│   │   │   ├── Muller_2024_H2_Basalt_at_25C.txt
│   │   │   ├── Basalt_25C_inc_rates.dat
│   │   │   └── [PNG plot dosyaları]
│   │   ├── 34C/
│   │   │   └── [aynı yapı]
│   │   └── 40C/
│   │       └── [aynı yapı]
│   │
│   ├── Calcite/
│   │   ├── 25C/
│   │   ├── 34C/
│   │   └── 40C/two_phase_apprx/
│   │
│   ├── Gypsum/
│   │   ├── 25C/
│   │   ├── 34C/two_phase_apprx/
│   │   └── 40C/
│   │
│   └── Sandstone/
│       ├── 25C/
│       ├── 34C/
│       └── 40C/
│
├── High_pressure_mura_2024/
│   └── mura_2024_matlab_code/
│       ├── mura_2024_fit_two_phase_clean.m
│       ├── Mura2024_Fig1_readoff_with_calcium.txt
│       └── [PDF ve PNG dosyaları]
│
└── Dokümantasyon Dosyaları
    ├── henrys law calculations for gases.docx
    └── Summary_Explanation.docx
```

## 2. MATLAB Dosyası Yapısı (anaerobic_model_two_phase_mixedSR_*_v4.m)

### Ana Fonksiyon Bölümleri

```matlab
function anaerobic_model_two_phase_mixedSR_25C_v4()
    %% PART 1: SETUP - Sabitler ve veri yükleme
    %% PART 2: INITIAL CONDITIONS - y0 ve parametre tanımları
    %% PART 3: FITTING - lsqnonlin optimizasyonu
    %% PART 4: SIMULATION - ODE çözümü ve görselleştirme
end

function dydt = model_mixed(t, y, p, env)
    % Ana ODE fonksiyonu - 14 diferansiyel denklem
end

function res = residuals_full(p, env, t_exp, data_exp, weights)
    % Optimizasyon için residual hesaplama
end

function [r_meth, r_sulf, r_precip, r_aceto] = rate_out_mixed(y, p, env, t)
    % Reaksiyon hızları hesaplama
end

function [H2S_aq, HS_aq] = speciate_sulfide(S_tot, pH, pKa)
    % pH-bağımlı sülfür spesiyasyonu
end
```

### Part 1: Setup Detayları

```matlab
% Fiziksel sabitler
R_gas = 8.314e-2;     % L·bar/(mol·K)
T_C = 25;             % Sıcaklık (°C)
T_K = T_C + 273.15;   % Sıcaklık (K)
Vg = 0.140;           % Headspace hacmi (L)
Vl = 0.015;           % Sıvı hacmi (L)
P_head = 2.0;         % Headspace basıncı (bar)

% Henry sabitleri (mmol/L/atm)
Hcp_H2_base  = 0.77;
Hcp_CO2_base = 33.0;
Hcp_H2S_base = 88.0;

% pH için pKa
pKa_H2S = 7.05;

% Veri yükleme
data = readmatrix('Muller_2024_H2_Basalt_at_25C.txt');
t_exp = data(:,1);              % Zaman (gün)
nH2_exp = data(:,2) / 1000;     % µmol → mmol
nCO2_exp = data(:,3) / 1000;
nCH4_exp = data(:,4) / 1000;
nH2S_exp = data(:,5) / 1000;
pH_exp = data(:,6);
SO4_exp = data(:,7);            % mM
```

### Part 2: Initial Conditions

```matlab
% Başlangıç koşulları (y0)
y0 = zeros(14, 1);

% Gaz fazı (mmol) - deneysel veriden
y0(1) = nH2_exp(1);
y0(2) = nCO2_exp(1);
y0(3) = nCH4_exp(1);
y0(4) = nH2S_exp(1);

% Sulu faz (mmol/L) - Henry yasasından hesaplanan
P_H2 = (y0(1) * R_gas * T_K) / Vg;
y0(5) = Hcp_H2_base * P_H2;      % H2_aq

P_CO2 = (y0(2) * R_gas * T_K) / Vg;
y0(6) = Hcp_CO2_base * P_CO2;    % CO2_aq

y0(7) = SO4_exp(1);              % SO4
y0(8) = 0.0;                     % FeS
y0(9) = 0.01;                    % X (biyokütle)
y0(10) = 0.0;                    % Acetate
y0(11) = 10.0;                   % HCO3 (sabit)
y0(12) = 0.001;                  % S_tot
y0(13) = 0.0;                    % Lag
y0(14) = 0.1;                    % Fe_pool
```

### Part 3: Parametre Tanımları

```matlab
% 28 parametre için alt ve üst sınırlar
%       [k_m,   k_s,   k_a,   Y_m,  Y_s,  Y_a,   KI_m, KI_s, KI_a, ...]
p_lb = [0.001, 0.001, 0.001, 0.01, 0.01, 0.01,  0.1,  0.1,  0.1,  ...];
p_ub = [0.5,   0.5,   0.5,   0.5,  0.5,  0.5,   10,   10,   10,   ...];
p_init = (p_lb + p_ub) / 2;  % Başlangıç tahmini

% Optimizasyon
options = optimoptions('lsqnonlin', ...
    'MaxFunctionEvaluations', 10000, ...
    'MaxIterations', 1000, ...
    'Display', 'iter');

p_fit = lsqnonlin(@(p) residuals_full(p, env, t_exp, data_exp, weights), ...
    p_init, p_lb, p_ub, options);
```

## 3. ODE Model Detayları (model_mixed fonksiyonu)

### Durum Değişkenleri Çıkarma
```matlab
function dydt = model_mixed(t, y, p, env)
    % Gaz fazı (mmol)
    nH2_g  = y(1);
    nCO2_g = y(2);
    nCH4_g = y(3);
    nH2S_g = y(4);
    
    % Sulu faz (mmol/L)
    H2_aq   = y(5);
    CO2_aq  = y(6);
    SO4     = y(7);
    FeS     = y(8);
    X       = y(9);
    Acetate = y(10);
    HCO3    = y(11);
    S_tot   = y(12);
    Lag     = y(13);
    Fe_pool = y(14);
```

### Henry Yasası ile Denge Hesabı
```matlab
    % Kısmi basınçlar
    n_tot = nH2_g + nCO2_g + nCH4_g + nH2S_g;
    y_H2 = nH2_g / n_tot;
    P_H2 = y_H2 * P_head;
    
    % Denge konsantrasyonları
    Hcp_H2_eff = phi_H2 * Hcp_H2_base;
    C_eq_H2 = Hcp_H2_eff * P_H2;
    
    % Kütle transfer akısı (mmol/L/gün)
    J_H2 = kla_H2 * (C_eq_H2 - H2_aq);
```

### Reaksiyon Hızları
```matlab
    % Monod kinetiği
    m_H2 = H2_aq / (K_H2 + H2_aq);
    m_SO4 = SO4 / (K_SO4 + SO4);
    m_CO2 = CO2_aq / (K_CO2 + CO2_aq);
    
    % Sülfür spesiyasyonu ve inhibisyon
    pH_t = interp1(env.t_pH, env.pH_data, t);
    frac_HS = 1 / (1 + 10^(pKa - pH_t));
    HS_aq = S_tot * frac_HS;
    f_inh_m = KI_m / (KI_m + HS_aq);
    
    % Aktivasyon fonksiyonları
    f_act = H2_aq / (H2_th + H2_aq);
    f_lag = 1 / (1 + exp(-(t - t_lag) / w_lag));
    
    % Reaksiyon hızları (mmol/L/gün)
    r_meth = k_m * X * m_H2 * m_CO2 * f_inh_m * f_act * Lag;
    r_sulf = k_s * X * m_H2 * m_SO4 * f_inh_s * f_act * Lag;
    r_aceto = k_a * X * m_H2 * m_CO2^2 * f_inh_a * f_act * Lag;
```

### Diferansiyel Denklemler
```matlab
    dydt = zeros(14, 1);
    
    % Gaz fazı (mmol/gün)
    dydt(1) = -J_H2 * Vl;                          % dnH2_g/dt
    dydt(2) = -J_CO2 * Vl;                         % dnCO2_g/dt
    dydt(3) = +r_meth * Vl;                        % dnCH4_g/dt
    dydt(4) = +J_out_H2S * Vl;                     % dnH2S_g/dt
    
    % Sulu faz (mmol/L/gün)
    dydt(5) = +J_H2 - 4*r_meth - 4*r_sulf - 4*r_aceto;     % dH2_aq/dt
    dydt(6) = +J_CO2 - 1*r_meth - 2*r_aceto;               % dCO2_aq/dt
    dydt(7) = -1*r_sulf + r_diss_gyp;                       % dSO4/dt
    dydt(8) = +r_prec;                                      % dFeS/dt
    dydt(9) = +Y_m*r_meth + Y_s*r_sulf + Y_a*r_aceto - b*X; % dX/dt
    dydt(10) = +r_aceto;                                    % dAcetate/dt
    dydt(11) = 0;                                           % dHCO3/dt (sabit)
    dydt(12) = +r_sulf - r_prec - J_out_H2S;               % dS_tot/dt
    dydt(13) = (f_lag - Lag) / w_lag;                      % dLag/dt
    dydt(14) = -r_prec;                                    % dFe_pool/dt
end
```

## 4. Deneysel Veri Dosyaları (.txt)

### Örnek: Basalt 25°C
```
Time(days)  H2(µmol)  CO2(µmol)  CH4(µmol)  H2S(µmol)  pH    SO4(mM)
0.0         8988      2436       0          1          6.8   5.5
1.1         8324      2263       0          1          6.9   4.3
2.0         7985      2156       5          3          6.9   3.8
3.0         7507      2060       25         67         7.1   3.2
5.0         6234      1823       125        156        7.3   2.3
7.0         4521      1456       285        198        7.5   1.8
9.0         2741      1012       445        167        7.8   1.5
11.0        1523      678        578        123        8.0   1.2
14.0        756       345        645        78         8.3   1.0
17.0        412       156        689        45         8.5   0.8
19.0        288       89         705        27         8.7   0.6
```

### Veri Özellikleri (Mineral Bazında)

| Mineral | H₂ Yarı Ömrü | CH₄ Max | SO₄ Tükenme | pH Aralığı |
|---------|--------------|---------|-------------|------------|
| Basalt | ~8 gün | ~700 µmol | ~19 gün | 6.8 → 8.7 |
| Calcite | ~9 gün | ~500 µmol | ~19 gün | 6.8 → 8.7 |
| Gypsum | ~4 gün | ~1200 µmol | Tükenmez | 6.5 → 7.8 |
| Sandstone | ~3 gün | ~1600 µmol | ~10 gün | 6.5 → 8.5 |

## 5. Best Fit Parametre Dosyaları (.mat)

### Dosya İçeriği
```matlab
% best_fit_params_Basalt_25C.mat
p_fit  = [1x28 double];  % Fitted parametre vektörü
env    = struct;         % Ortam yapısı
y0     = [14x1 double];  % Başlangıç koşulları

% env yapısı:
env.Vg = 0.14;           % Headspace hacmi (L)
env.Vl = 0.015;          % Sıvı hacmi (L)
env.T_K = 298.15;        % Sıcaklık (K)
env.R_gas = 0.08314;     % Gaz sabiti
env.P_head = 2.0;        % Basınç (bar)
env.Hcp_H2_base = 0.77;  % Henry sabitleri
env.Hcp_CO2_base = 33.0;
env.Hcp_H2S_base = 88.0;
env.pKa_H2S = 7.05;
env.t_pH = [...];        % pH interpolasyon zamanları
env.pH_data = [...];     % pH değerleri
env.SO4_sat = 15.0;      % Jips doygunluğu (mM)
```

### Tipik Parametre Değerleri (Basalt 25°C örneği)
```matlab
p_fit = [
    0.08,   % k_m - metanojenez hızı (1/gün)
    0.12,   % k_s - sülfat indirgeme hızı (1/gün)
    0.03,   % k_a - asetojenez hızı (1/gün)
    0.05,   % Y_m - metanojen verimi
    0.08,   % Y_s - SRB verimi
    0.03,   % Y_a - asetojen verimi
    2.0,    % KI_m - sülfür inhibisyonu (mmol/L)
    3.0,    % KI_s
    1.5,    % KI_a
    0.0,    % k_prec - FeS çökelme (devre dışı)
    0.5,    % HS_sat
    0.05,   % H2_th - H2 eşiği
    -5.0,   % DG_th - termodinamik eşik
    0.5,    % K_H2 - yarı doygunluk
    2.0,    % K_SO4
    5.0,    % K_CO2
    10.0,   % kla_H2 - kütle transfer (1/gün)
    8.0,    % kla_CO2
    25.0,   % kla_H2S
    0.02,   % b - bozunma hızı
    3.0,    % t_lag - lag merkezi (gün)
    0.5,    % w_lag - lag genişliği
    0.1,    % k_diss_gyp - jips çözünme
    0.05,   % beta_SO4_m - rekabet faktörü
    1.0,    % phi_H2 - Henry ölçeği
    1.0,    % phi_CO2
    1.0,    % phi_H2S
    1.5     % alpha_H2S - degassing ölçeği
];
```

## 6. Model Çıktı Dosyası (.dat)

### Sütun Yapısı (21 sütun)
```
Sütun  Değişken     Birim        Açıklama
-----  --------     -----        --------
1      Time         gün          Simülasyon zamanı
2      nH2_g        mmol         Gaz fazı H2
3      nCO2_g       mmol         Gaz fazı CO2
4      nCH4_g       mmol         Gaz fazı CH4
5      nH2S_g       mmol         Gaz fazı H2S
6      H2_aq        mmol/L       Çözünmüş H2
7      CO2_aq       mmol/L       Çözünmüş CO2
8      SO4          mmol/L       Sülfat
9      FeS          mmol/L       Demir sülfür
10     X            mmol/L       Biyokütle
11     Acetate      mmol/L       Asetat
12     HCO3         mmol/L       Bikarbonat
13     S_tot        mmol/L       Toplam sülfür
14     H2S_aq       mmol/L       Çözünmüş H2S (spesiye edilmiş)
15     HS           mmol/L       Bisülfür iyonu
16     Lag          -            Aktivasyon (0-1)
17     Fe_pool      mmol/L       Kullanılabilir Fe
18     r_meth       mmol/L/gün   Metanojenez hızı
19     r_sulf       mmol/L/gün   Sülfat indirgeme hızı
20     r_precip     mmol/L/gün   FeS çökelme hızı
21     r_aceto      mmol/L/gün   Asetojenez hızı
```

### Örnek Çıktı (İlk 5 satır)
```
0.000  8.988  2.436  0.000  0.001  1.209  14.05  5.50  0.000  0.010  0.000  10.0  0.001  0.0006  0.0004  0.00  0.10  0.006  0.003  0.000  0.010
0.013  8.975  2.432  0.001  0.001  1.207  14.02  5.49  0.000  0.010  0.000  10.0  0.002  0.0012  0.0008  0.01  0.10  0.007  0.004  0.000  0.011
0.027  8.961  2.428  0.002  0.002  1.204  13.99  5.48  0.000  0.011  0.001  10.0  0.003  0.0018  0.0012  0.02  0.10  0.008  0.005  0.000  0.012
...
```

## 7. Python'a Port için Kritik Noktalar

### scipy.integrate.solve_ivp Kullanımı
```python
from scipy.integrate import solve_ivp

def model_mixed(t, y, p, env):
    # MATLAB model_mixed fonksiyonunun Python karşılığı
    dydt = np.zeros(14)
    # ... (aynı denklemler)
    return dydt

sol = solve_ivp(
    lambda t, y: model_mixed(t, y, p_fit, env),
    t_span=[0, t_end],
    y0=y0,
    method='BDF',  # ode15s karşılığı (stiff sistem)
    dense_output=True,
    max_step=0.1
)
```

### .mat Dosyası Okuma
```python
import scipy.io as sio

mat_data = sio.loadmat('best_fit_params_Basalt_25C.mat')
p_fit = mat_data['p_fit'].flatten()
y0 = mat_data['y0'].flatten()
# env struct'ı dict olarak gelir
```

## 8. LSTM için Veri Hazırlama Notları

### Zaman Serisi Formatı
- **Girdi:** (batch, sequence_length, 14 features)
- **Çıktı:** (batch, 14 features) veya (batch, sequence_length, 14 features)

### Normalizasyon
```python
# Her değişken için mean/std normalizasyonu
mean = data.mean(axis=0)
std = data.std(axis=0)
data_norm = (data - mean) / std
```

### Sliding Window
```python
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)
```
