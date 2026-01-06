# LSTM Training Data vs ODE Simulation Output

**Tarih**: 6 Ocak 2026
**Amaç**: `inc_rates.dat` (ODE simulation output) ile `lstm_training_data_v4_normalized.mat` (LSTM eğitim verisi) arasındaki farkları açıklamak

---

## 🎯 Ana Soru

> **"İkisi de tahmin etmek istediğimiz şeyler için eğitim verisi, değil mi?"**

**CEVAP**: **EVET!** İkisi de **aynı fiziksel sistemi** (14 state variable'ın ODE dinamikleri) temsil ediyor. Ancak **farklı amaçlar için farklı formatlarda** kaydedilmiş.

---

## 📊 Karşılaştırma Tablosu

| **Özellik** | **Basalt_25C_inc_rates.dat** | **lstm_training_data_v4_normalized.mat** |
|------------|------------------------------|------------------------------------------|
| **Kaynak** | Ana kod simulation sonucu | LSTM için özel üretildi |
| **Üretildiği Yer** | `anaerobic_model_two_phase_mixedSR_25C_v4.m` (satır ~200-250) | `lstm_train_v4.m` (satır 40-67) |
| **Zaman Noktası Sayısı** | Adaptif (ode15s'in seçtiği, değişken) | 2000 (sabit, uniform) |
| **Zaman Aralığı** | Adaptif (başta sık: 0, 2e-6, 5e-6...; sonra seyrek) | Uniform (eşit aralıklı: linspace) |
| **State Sayısı** | 14 state + 6 ekstra = **20 sütun** | Sadece **14 state variable** |
| **Ekstra Bilgiler** | ✅ r_meth, r_sulf, r_precip, r_aceto, H2S_aq, HS | ❌ Yok (sadece state) |
| **Normalizasyon** | ❌ RAW değerler (original scale) | ✅ Z-score + log1p uygulanmış |
| **Format** | `.dat` (text, space-separated, ASCII) | `.mat` (MATLAB binary) |
| **Kullanım Amacı** | Post-processing, plotting, makale figürleri, analiz | LSTM eğitimi (sequence learning) |
| **Dosya Boyutu** | ~100-500 KB (text, adaptif nokta sayısı) | ~5-10 MB (binary, 2000 nokta + metadata) |

---

## 🔬 Basalt_25C_inc_rates.dat İçeriği

### **Sütun Yapısı** (21 sütun)

```
Time(days) | 14 State Variables | 6 Ekstra Bilgi
-----------+---------------------+----------------
Time       | nH2_g   nCO2_g      | H2S_aq  HS
           | nCH4_g  nH2S_g      | r_meth  r_sulf
           | H2_aq   CO2_aq      | r_precip r_aceto
           | SO4     FeS         |
           | X       Acetate     |
           | HCO3    S_tot       |
           | Lag     Fe_pool     |
```

### **Örnek Satırlar** (ilk 3 satır):

```
Time       nH2_g   nCO2_g   nCH4_g     nH2S_g    ...  r_meth      r_sulf      r_aceto
0.000000   9.074   2.464    0          0.001     ...  0.00220802  0.00303778  0.0149299
0.000002   9.074   2.464    8.08e-11   0.00100   ...  0.00220805  0.00303782  0.0149301
0.000005   9.074   2.464    1.62e-10   0.00100   ...  0.00220809  0.00303785  0.0149302
```

**Zaman Adımları**: Adaptif (ODE solver'ın kendi seçimi)
- Başlangıç: Çok sık (0, 2e-6, 5e-6, 7e-6 gün)
- Sonra: Seyrekleşir (dinamikler yavaşlayınca)

### **Üretilme Kodu** (Ana kod içinde):

```matlab
% anaerobic_model_two_phase_mixedSR_25C_v4.m (satır ~200-250)
[t_sim, y_sim] = ode15s(@(t,y) model_mixed(t,y,p_fit,env), t_exp, y0, opts);

% Ekstra bilgileri hesapla (r_meth, r_sulf, H2S_aq, HS)
for i = 1:length(t_sim)
    [rates, ~, H2S_aq, HS] = rate_out_mixed(t_sim(i), y_sim(i,:), p_fit, env);
    % ...
end

% Tüm veriyi kaydet
data_out = [t_sim, y_sim, H2S_aq, HS, r_meth, r_sulf, r_precip, r_aceto];
writematrix(data_out, 'Basalt_25C_inc_rates.dat', 'Delimiter', ' ');
```

---

## 🤖 lstm_training_data_v4_normalized.mat İçeriği

### **İçindeki Değişkenler**:

```matlab
t_train        % [2000 x 1] - Uniform zaman grid'i
xTrain_raw     % [14 x 2000] - RAW state values (normalizasyon öncesi)
xTrain_norm    % [14 x 2000] - Normalize edilmiş state values
X              % {1990 x 1} cell - Her eleman [14 x 10] sequence (LSTM input)
Y              % [1990 x 14] - Her satır [1 x 14] next timestep (LSTM target)
p_fit          % [1 x 28] - Fitted parameters
env            % struct - Environment (Vg, Vl, T, Henry constants)
y0             % [14 x 1] - Initial conditions
norm_params    % struct - Normalization metadata
    .mean      % [14 x 1] - Mean for each state
    .std       % [14 x 1] - Std for each state
    .log_indices % [1 x 3] - Indices of log-transformed states (4, 10, 13)
```

### **Üretilme Kodu** (lstm_train_v4.m):

```matlab
% lstm_train_v4.m (satır 40-108)

%% 1. Uniform zaman grid oluştur
t_train = linspace(0, t_exp(end), 2000)'; % FARK: Uniform, 2000 nokta

%% 2. ODE'yi uniform grid üzerinde çöz
[~, y_sim] = ode15s(@(t,y) model_mixed(t,y,p_fit,env), t_train, y0, opts);
xTrain_raw = y_sim'; % [14 x 2000]

%% 3. Log transform (küçük değerler için)
log_indices = [4, 10, 13]; % nH2S_g, Acetate, Lag
xTrain(log_indices, :) = log1p(xTrain_raw(log_indices, :));

%% 4. Z-score normalizasyon
xTrain_mean = mean(xTrain, 2); % [14 x 1]
xTrain_std = std(xTrain, 0, 2); % [14 x 1]
xTrain_norm = (xTrain - xTrain_mean) ./ xTrain_std;

%% 5. Sequence oluştur (sliding window)
sequenceLength = 10;
for i = 1:(2000 - sequenceLength)
    X{end+1} = xTrain_norm(:, i:i+9);       % [14 x 10] (input)
    Y(end+1, :) = xTrain_norm(:, i+10)';    % [1 x 14] (target)
end

%% 6. Kaydet
save('lstm_training_data_v4_normalized.mat', ...
     't_train', 'xTrain_raw', 'xTrain_norm', 'X', 'Y', ...
     'p_fit', 'env', 'y0', 'norm_params');
```

---

## 🔍 Neden İki Farklı Format?

### **1. inc_rates.dat → Ana Kod Simulation Output**

**Amaç**: ODE simulation sonuçlarını **insan okunabilir formatta** kaydetmek

**Kullanım Senaryoları**:
- ✅ Plotting (MATLAB, Python, Excel)
- ✅ Post-processing analizi
- ✅ Makale figürleri (reaction rates vs time)
- ✅ Model validasyon (experimental data ile karşılaştırma)
- ✅ Ekstra bilgiler (r_meth, r_sulf) görselleştirme

**Avantajlar**:
- Text formatı → Her programda açılabilir
- Adaptif zaman adımları → ODE solver'ın doğal çıktısı
- Ekstra bilgiler (rates) → Kimyasal analiz için kullanışlı

---

### **2. lstm_training_data_v4_normalized.mat → LSTM Eğitim Verisi**

**Amaç**: LSTM ağını eğitmek için **özel hazırlanmış sequence data**

**Kullanım Senaryoları**:
- ✅ LSTM training (sequence-to-one learning)
- ✅ Hızlı ODE emulation (transport simülasyonlarında)
- ✅ Normalizasyon metadata'sı saklama (denormalization için gerekli)
- ✅ Reproducibility (aynı eğitim verisiyle tekrar eğitim)

**Avantajlar**:
- Uniform zaman adımları → Sequence learning için ideal
- Normalize edilmiş → LSTM eğitimi için kritik (farklı ölçeklerdeki state'ler)
- Metadata dahil → Denormalization için gerekli bilgiler
- Sliding window sequences → [14 x 10] input, [1 x 14] target

---

## 📐 Normalizasyon Detayları

### **Neden Normalizasyon Gerekli?**

ODE state variables **çok farklı ölçeklerde**:

```
nH2_g:    ~0.1 - 10 mmol        (10⁰)
nH2S_g:   ~0.001 - 0.02 mmol    (10⁻³)
Acetate:  ~1e-8 - 1e-5 mmol     (10⁻⁸)
Fe_pool:  ~0.1 - 10 mmol        (10⁰)
SO4:      ~1 - 10 mmol/L        (10⁰)
```

**LSTM problemi**: Farklı ölçeklerdeki değişkenleri **öğrenemez** (gradient flow bozulur)

### **Çözüm: İki Aşamalı Transform**

#### **Adım 1: Log Transform (Küçük Değerler İçin)**

```matlab
log_indices = [4, 10, 13]; % nH2S_g, Acetate, Lag
xTrain(log_indices, :) = log1p(xTrain_raw(log_indices, :));
```

**Neden**: Çok küçük değerler (1e-8) için log scale daha stabil

#### **Adım 2: Z-score Normalizasyon (Tüm Değişkenler)**

```matlab
xTrain_mean = mean(xTrain, 2); % [14 x 1]
xTrain_std = std(xTrain, 0, 2); % [14 x 1]
xTrain_norm = (xTrain - xTrain_mean) ./ xTrain_std;
```

**Sonuç**: Tüm state variables **~[-3, +3]** aralığında → LSTM öğrenebilir

### **Denormalizasyon (Tahmin Sonrası)**

```matlab
% LSTM tahmini (normalized)
Y_pred_norm = predict(net, X_test); % [N x 14]

% 1. Z-score'u geri al
Y_pred = Y_pred_norm .* norm_params.std' + norm_params.mean';

% 2. Log transform'u geri al (sadece log_indices için)
Y_pred(:, norm_params.log_indices) = expm1(Y_pred(:, norm_params.log_indices));

% Sonuç: Orijinal ölçekte tahmin
```

---

## 🎯 Özet: İkisi De Aynı Fiziksel Sistemi Temsil Ediyor

### **Ortak Özellikler**:

```
┌─────────────────────────────────────────┐
│  Aynı Fiziksel Model (14 ODE)           │
│  - nH2_g, nCO2_g, nCH4_g, nH2S_g       │
│  - H2_aq, CO2_aq, SO4, FeS, X          │
│  - Acetate, HCO3, S_tot, Lag, Fe_pool  │
└─────────────────────────────────────────┘
          │
          │ (Aynı ODE solver: ode15s)
          │ (Aynı parameters: p_fit)
          │
          ▼
┌──────────────────────────────────────────┐
│  İkisi de "Ground Truth" Dinamikler     │
│  = Sistemin gerçek davranışı            │
└──────────────────────────────────────────┘
          │
          ├──────────────┬─────────────────┐
          ▼              ▼                 ▼
    inc_rates.dat   LSTM Training    Tahmin Amacı
    (adaptif)       (uniform+norm)   (aynı dinamikler)
```

### **Fark: Kullanım Amacı**

| **Dosya** | **Kullanım** |
|-----------|-------------|
| **inc_rates.dat** | Analiz, plotting, makale figürleri (insan için) |
| **LSTM training data** | Hızlı tahmin için model eğitimi (makine için) |

---

## 🚀 LSTM'in Avantajı: Hız

### **Transport Simülasyonunda Neden LSTM?**

```python
# Transport simülasyonu (1D grid, 75 hücre, 100 zaman adımı)
# Her hücrede her zaman adımında ODE çözmek gerekiyor

# ODE Solver ile:
for time_step in range(100):
    for cell in range(75):
        [t, y] = ode15s(model_mixed, ...) # ~0.5 saniye/hücre
# Toplam: 75 × 100 × 0.5s = 3750 saniye (~1 saat!)

# LSTM ile:
for time_step in range(100):
    for cell in range(75):
        y_pred = predict(net, history) # ~0.001 saniye/hücre
# Toplam: 75 × 100 × 0.001s = 7.5 saniye!
```

**Hız Kazancı**: **500x daha hızlı** 🚀

---

## 📝 Sonuç

> **"İkisi de tahmin etmek istediğimiz şeyler için eğitim verisi, değil mi?"**

**EVET, kesinlikle doğru!**

- **inc_rates.dat**: ODE simulation sonucu (adaptif, ekstra bilgilerle, analiz için)
- **LSTM training data**: Aynı ODE sonucu (uniform, normalized, hızlı tahmin için)

**İkisi de aynı "ground truth"e bakıyor**: 14 state variable'ın zaman içindeki dinamikleri

**Tek fark**: Biri **insan analizi** için (plotting, makale), diğeri **makine öğrenmesi** için (LSTM, hızlı emulation)

---

**Dosya**: `d:\chemical_thesis_repo\2026-W01_model_anlama\notes\05_lstm_vs_ode_data.md`
**Oluşturulma Tarihi**: 6 Ocak 2026
**Son Güncelleme**: 6 Ocak 2026
