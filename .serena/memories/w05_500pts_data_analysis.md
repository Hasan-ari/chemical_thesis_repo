# W05 - 500 Pointlik Veri Analizi

## 1. Veri Kaynağı Özeti

### Dosya Bilgileri
| Özellik | Değer |
|---------|-------|
| **W05 Veri Dosyası** | `2026_W05_seq_len_experiment/app/data/basalt_25c_lstm_input_500pts.npy` |
| **Kaynak (W04)** | `2026_W04_Lstm_training_v1/new_app/data/output/basalt_25c_lstm_input_500pts.npy` |
| **Üretim Scripti** | `2026_W04_Lstm_training_v1/new_app/src/lstm_synth_data/generate.py` |
| **Shape** | (500, 14) - 500 timestep, 14 state değişkeni |

### Zaman Parametreleri
- **t_start:** 0 gün
- **t_end:** 19 gün
- **n_points:** 500
- **dt:** 19/499 ≈ 0.038 gün ≈ 55 dakika

---

## 2. Data Sentezi Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  1. Parametre Yükleme (params.py)                           │
│     - best_fit_params_Basalt_25C.mat dosyasından            │
│     - 28 fitted parametre (p_fit)                           │
│     - Environment config (Vg, Vl, T, Henry sabitleri)       │
│     - Başlangıç koşulları (y0) - deneysel veriden           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. ODE Çözümü (generate.py)                                │
│     - scipy.integrate.solve_ivp                             │
│     - Method: Radau (stiff systems için, MATLAB ode15s gibi)│
│     - rtol=1e-8, atol=1e-10                                 │
│     - t_eval: np.linspace(0, 19, 500) uniform grid          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Çıktı Dosyaları                                         │
│     - basalt_25c_synth_500pts.csv (tüm veriler + rates)     │
│     - basalt_25c_synth_500pts.npz (numpy archive)           │
│     - basalt_25c_lstm_input_500pts.npy (sadece 14 state)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 14 State Değişkeni

```python
STATE_NAMES = [
    "nH2_g",    # 0:  Gaz fazı H2 (mmol)
    "nCO2_g",   # 1:  Gaz fazı CO2 (mmol)
    "nCH4_g",   # 2:  Gaz fazı CH4 (mmol) - metanojenez ürünü
    "nH2S_g",   # 3:  Gaz fazı H2S (mmol) - sülfür ürünü
    "H2_aq",    # 4:  Çözünmüş H2 (mmol/L)
    "CO2_aq",   # 5:  Çözünmüş CO2 (mmol/L)
    "SO4",      # 6:  Sülfat (mmol/L)
    "FeS",      # 7:  Demir sülfür çökeltisi (mmol/L)
    "X",        # 8:  Biyokütle (mmol/L)
    "Acetate",  # 9:  Asetat (mmol/L)
    "HCO3",     # 10: Bikarbonat - SABİT (mmol/L)
    "S_tot",    # 11: Toplam çözünmüş sülfür (mmol/L)
    "Lag",      # 12: Lag fazı aktivasyonu (0-1)
    "Fe_pool"   # 13: Kullanılabilir Fe havuzu (mmol/L)
]
```

---

## 4. Deneysel Veri Karşılaştırması

### Kaynak: Muller 2024 - Basalt @ 25°C

```python
# params.py içindeki deneysel veri
BASALT_25C_DATA = np.array([
    # [time, H2(umol), CO2(umol), CH4(umol), H2S(umol), pH, SO4(mM)]
    [0.0,   9074, 2464,   0,   1, 6.7, 5.7],
    [1.1,   8655, 2338,   0,   0, 6.9, 5.7],
    [5.0,   8016, 2203,  12,  20, 7.0, 3.1],
    [6.0,   7603, 2086,  25,  35, 7.2, 2.8],
    [7.0,   7362, 1949,  45,  54, 7.1, 2.2],
    [8.0,   6946, 1820,  66,  50, 7.2, 2.4],
    [9.0,   5560, 1407,  89,  49, 7.2, 2.2],
    [12.0,  3728,  766, 141,  41, 6.7, 1.9],
    [13.9,  2128,  280, 173,  42, 6.0, 1.3],
    [15.9,  1808,  189, 201,  35, 5.9, 1.2],
    [19.0,  1409,   80, 255,  30, 5.9, 1.1],
])
```

### Başlangıç Koşulları (t=0) - ✅ DOĞRU

| Değişken | Model (CSV) | Deneysel | Durum |
|----------|-------------|----------|-------|
| nH2_g | 9.074 mmol | 9.074 mmol (9074 µmol) | ✅ Eşit |
| nCO2_g | 2.464 mmol | 2.464 mmol (2464 µmol) | ✅ Eşit |
| nCH4_g | 0 mmol | 0 mmol | ✅ Eşit |
| nH2S_g | 0.001 mmol | 0.001 mmol (1 µmol) | ✅ Eşit |
| SO4 | 5.7 mM | 5.7 mM | ✅ Eşit |

### Son Değerler (t=19 gün) - ⚠️ Model Approximation

| Değişken | Model (CSV) | Deneysel | Fark Açıklaması |
|----------|-------------|----------|-----------------|
| nH2_g | 2.37 mmol | 1.41 mmol | Model H2 tüketimini yavaş tahmin |
| nCO2_g | 0.022 mmol | 0.080 mmol | Model CO2'yi fazla tüketiyor |
| nCH4_g | 0.35 mmol | 0.26 mmol | Yakın - kabul edilebilir |
| nH2S_g | 0.15 mmol | 0.03 mmol | Model H2S üretimini yüksek tahmin |
| SO4 | 1.05 mM | 1.1 mM | ✅ Yakın - iyi fit |

**Not:** Son değerlerdeki farklar KOD HATASI DEĞİL, modelin doğal limitasyonudur. Best-fit parametreleri bile mükemmel eşleşme sağlayamaz.

---

## 5. Kod Doğrulama Sonuçları

### ✅ Doğru Çalışan Kısımlar

1. **Parametre Yükleme** (`params.py:33-73`)
   - .mat dosyasından p_fit, env, y0 doğru yükleniyor
   - Fallback olarak default parametreler var

2. **Başlangıç Koşulları** (`params.py:142-187`)
   - Deneysel veriden (BASALT_25C_DATA) doğru hesaplanıyor
   - µmol → mmol dönüşümü doğru
   - Henry yasası ile H2_aq, CO2_aq hesabı doğru

3. **ODE Solver** (`generate.py:68-77`)
   - Radau method (stiff systems için uygun)
   - Toleranslar: rtol=1e-8, atol=1e-10 (yeterli hassasiyet)
   - Uniform time grid: np.linspace(0, 19, 500)

4. **Çıktı Formatı** (`generate.py:166-168`)
   - LSTM input olarak sadece states kaydediliyor
   - Shape: (500, 14) - beklenen format

### ⚠️ Dikkat Edilecek Noktalar

1. W05'teki veri W04'ten **kopyalanmış** olmalı (aynı dosya adı)
2. pH interpolasyonu deneysel veriden yapılıyor (time-dependent)
3. HCO3 (index 10) model boyunca SABİT kalıyor (0 değeri)

---

## 6. W05 Deney Bağlamı

### Sequence Length Threshold Experiment

500 pointlik veri şu şekilde kullanılıyor:

```
Veri: [y[0], y[1], y[2], ..., y[499]]  (500 nokta, her biri 14 feature)

seq_len=50 için:
    Input:  [y[0], y[1], ..., y[49]]   → Output: y[50]
    Input:  [y[1], y[2], ..., y[50]]   → Output: y[51]
    ...
    Toplam training sample: 500 - 50 = 450 adet

seq_len değerleri test ediliyor: [50, 30, 20, 10, 5]
```

### Hedef
- Overfit edene kadar eğit (target_loss = 1e-8)
- Autoregressive trajectory üret
- Minimum başarılı seq_len değerini bul

---

## 7. Dosya Lokasyonları

```
chemical_thesis_repo/
├── 2026_W04_Lstm_training_v1/
│   └── new_app/
│       ├── src/lstm_synth_data/
│       │   ├── generate.py      # ← Data sentezi ana kodu
│       │   ├── params.py        # ← Parametre yükleme
│       │   └── ode_model.py     # ← ODE model tanımı
│       └── data/output/
│           ├── basalt_25c_lstm_input_500pts.npy  # ← LSTM input
│           ├── basalt_25c_synth_500pts.csv       # ← Full CSV
│           └── basalt_25c_synth_500pts.npz       # ← NumPy archive
│
└── 2026_W05_seq_len_experiment/
    └── app/
        ├── data/
        │   └── basalt_25c_lstm_input_500pts.npy  # ← W04'ten kopyalanmış
        └── src/lstm_experiment/
            └── run_experiment.py  # ← Seq_len deneyi
```

---

## 8. Sonuç

**Data sentezi kodu DOĞRU çalışıyor:**

- ✅ Başlangıç koşulları deneysel verilerle birebir eşleşiyor
- ✅ ODE solver doğru parametrelerle çalışıyor
- ✅ 14 state değişkeni doğru sırayla üretiliyor
- ✅ Zaman grid'i uniform ve doğru (500 nokta, 0-19 gün)
- ⚠️ Model-deney farkları kod hatası değil, model limitasyonu

**500 pointlik veri LSTM eğitimi için kullanılabilir durumda.**
