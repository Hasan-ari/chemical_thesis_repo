# Chemical Thesis Project - Comprehensive Overview

## 1. Ekip ve Roller

| Rol | Kişi | Sorumluluk |
|-----|------|------------|
| **Kimya Doçenti** | Tez Danışmanı #1 | MATLAB kodları, parametreler, deneysel veri analizi |
| **CS Doçenti** | Tez Danışmanı #2 | Makine öğrenmesi danışmanlığı |
| **Öğrenci** | CS Yüksek Lisans | PyTorch LSTM/RNN surrogate model geliştirme |

## 2. Proje Amacı

**Ana Hedef:** Yeraltı ortamlarında mikrobiyal H₂ üretimini modelleyen ODE sisteminin **surrogate modeli** olarak LSTM/RNN geliştirmek.

**Neden Surrogate Model?**
- ODE çözümü hesaplama maliyetli
- Tekrarlayan (time-series) veri yapısı → LSTM/RNN uygun
- Hızlı tahmin için neural network avantajlı

## 3. Araştırma Konusu

### Bilimsel Bağlam
- **Alan:** Yeraltı H₂ depolama ve mikrobiyal metabolizma
- **Süreç:** Anaerobik ortamda H₂ tüketimi ve CH₄/H₂S üretimi
- **Veri Kaynağı:** Muller 2024 laboratuvar deneyleri

### Modellenen Kimyasal Reaksiyonlar
1. **Metanojenez:** 4H₂ + CO₂ → CH₄ + 2H₂O
2. **Sülfat İndirgeme:** 4H₂ + SO₄²⁻ + 2H⁺ → H₂S + 4H₂O
3. **Homoasetojenez:** 4H₂ + 2CO₂ → CH₃COOH + 2H₂O
4. **FeS Çökelme:** Fe²⁺ + HS⁻ → FeS(s) + H⁺

## 4. Deneysel Koşullar

### Muller 2024 (Düşük Basınç)
- **Mineraller:** Basalt, Calcite, Gypsum, Sandstone
- **Sıcaklıklar:** 25°C, 34°C, 40°C
- **Toplam:** 4 mineral × 3 sıcaklık = 12 koşul
- **Headspace:** 140 mL (Vg = 0.14 L)
- **Sıvı:** 15 mL (Vl = 0.015 L)
- **Süre:** ~19 gün

### Mura 2024 (Yüksek Basınç)
- **Basınç:** ~60 bar
- **Sıcaklık:** 36°C
- **Süre:** ~105 gün

## 5. Veri Yapısı

### Deneysel Veri Formatı (.txt)
**Dosya:** `Muller_2024_H2_[Mineral]_at_[Temp]C.txt`

| Sütun | Parametre | Birim | Açıklama |
|-------|-----------|-------|----------|
| 1 | Time | gün | Örnekleme zamanı |
| 2 | H₂(g) | µmol | Gaz fazı hidrojen |
| 3 | CO₂(g) | µmol | Gaz fazı karbondioksit |
| 4 | CH₄(g) | µmol | Metan üretimi |
| 5 | H₂S(g) | µmol | Hidrojen sülfür |
| 6 | pH | - | Çözelti pH'ı |
| 7 | SO₄ | mM | Sülfat konsantrasyonu |

### Model Çıktısı (.dat)
**Dosya:** `[Mineral]_[Temp]C_inc_rates.dat`
- 21 sütun, 1433 zaman noktası
- Tüm durum değişkenleri + reaksiyon hızları

## 6. MATLAB Model Yapısı (v4 Two-Phase)

### 14 Durum Değişkeni
```matlab
y = [nH2_g,    % 1: Gaz fazı H₂ (mmol)
     nCO2_g,   % 2: Gaz fazı CO₂ (mmol)
     nCH4_g,   % 3: Gaz fazı CH₄ (mmol)
     nH2S_g,   % 4: Gaz fazı H₂S (mmol)
     H2_aq,    % 5: Çözünmüş H₂ (mmol/L)
     CO2_aq,   % 6: Çözünmüş CO₂ (mmol/L)
     SO4,      % 7: Sülfat (mmol/L)
     FeS,      % 8: Demir sülfür çökeltisi (mmol/L)
     X,        % 9: Biyokütle (mmol/L)
     Acetate,  % 10: Asetat (mmol/L)
     HCO3,     % 11: Bikarbonat (sabit) (mmol/L)
     S_tot,    % 12: Toplam çözünmüş sülfür (mmol/L)
     Lag,      % 13: Lag fazı aktivasyonu (0-1)
     Fe_pool]  % 14: Kullanılabilir Fe²⁺ (mmol/L)
```

### 28 Parametre
| İndeks | Parametre | Birim | Açıklama |
|--------|-----------|-------|----------|
| 1-3 | k_m, k_s, k_a | 1/gün | Maksimum metabolik hızlar |
| 4-6 | Y_m, Y_s, Y_a | mmolX/mmol | Biyokütle verimleri |
| 7-9 | KI_m, KI_s, KI_a | mmol/L | Sülfür inhibisyon sabitleri |
| 10-11 | k_prec, HS_sat | 1/gün, mmol/L | FeS çökelme parametreleri |
| 12-13 | H2_th, DG_th | mmol/L, kJ/mol | Aktivasyon eşikleri |
| 14-16 | K_H2, K_SO4, K_CO2 | mmol/L | Monod yarı-doygunluk |
| 17-19 | kla_H2, kla_CO2, kla_H2S | 1/gün | Kütle transfer katsayıları |
| 20-22 | b, t_lag, w_lag | 1/gün, gün, gün | Bozunma, lag zamanlaması |
| 23 | k_diss_gyp | 1/gün | Jips çözünme hızı |
| 24 | beta_SO4_m | mM⁻¹ | SO₄-metanojen rekabeti |
| 25-27 | phi_H2, phi_CO2, phi_H2S | - | Henry ölçek faktörleri |
| 28 | alpha_H2S | - | H₂S degassing ölçeği |

### Henry Sabitleri (25°C)
| Gaz | Değer | Birim |
|-----|-------|-------|
| H₂ | 0.77 | mmol/L/atm |
| CO₂ | 33.0 | mmol/L/atm |
| H₂S | 88.0 | mmol/L/atm |

## 7. Dosya Yapısı

### LatestCodes Klasörü
```
LatestCodes/
├── Low_pressure_muller_2024_best_fits/
│   ├── Basalt/
│   │   ├── 25C/
│   │   │   ├── anaerobic_model_two_phase_mixedSR_25C_v4.m
│   │   │   ├── best_fit_params_Basalt_25C.mat
│   │   │   ├── Muller_2024_H2_Basalt_at_25C.txt
│   │   │   └── Basalt_25C_inc_rates.dat
│   │   ├── 34C/
│   │   └── 40C/
│   ├── Calcite/
│   ├── Gypsum/
│   └── Sandstone/
│
└── High_pressure_mura_2024/
    └── mura_2024_matlab_code/
```

### Best Fit Dosyaları (.mat)
Her mineral/sıcaklık için:
- `p_fit`: 28 parametre vektörü
- `env`: Ortam sabitleri (Henry, pH fonksiyonu)
- `y0`: 14 elemanlı başlangıç koşulu vektörü

## 8. Sentetik Veri Üretim Akışı

```
1. .mat dosyası yükle (p_fit, env, y0)
         ↓
2. ode15s ile ODE çöz
   - dydt = model_mixed(t, y, p, env)
   - Timespan: [0, t_exp_end]
   - 1433 zaman noktası
         ↓
3. Post-processing
   - Sülfür spesiyasyonu hesapla
   - Reaksiyon hızları hesapla
         ↓
4. .dat dosyasına kaydet (21 sütun)
```

## 9. LSTM Surrogate Model Hedefleri

### Girdi/Çıktı Yapısı
- **Girdi:** Zaman serisi (14 durum değişkeni)
- **Çıktı:** Bir sonraki zaman adımı tahminleri

### Mimari (Planlanan)
- Stacked LSTM: 128 → 64 birim
- Sequence length: 100 adım
- Eğitim/Test: 80/20 split

### Doğrulama Stratejisi
- Free-running (chain) prediction
- Teacher forcing olmadan
- 150 adım recursive forecast
- Hata analizi: 50, 100, 150 adımlarda

## 10. Önemli Klasörler

| Klasör | İçerik |
|--------|--------|
| `CURRENT/` | Aktif çalışma dizini |
| `LatestCodes/` | Kimya hocasının MATLAB kodları |
| `2026-W01_model_anlama/` | Model dokümantasyonu |
| `2026_W04_Lstm_training_v1/` | LSTM eğitim kodları |

## 11. Teknik Notlar

### ODE Çözücü
- **Algoritma:** ode15s (stiff sistem için)
- **Toleranslar:** RelTol=1e-8, AbsTol=1e-10
- **NonNegative:** Tüm 14 durum için aktif

### Parametre Optimizasyonu
- **Yöntem:** lsqnonlin (least-squares)
- **Residual:** Log-weighted (log_sim - log_exp)
- **SO₄ Ağırlığı:** 2.0× (plato önemli)

### Sülfür Spesiyasyonu
```
pKa(H₂S) = 7.05 @ 25°C
HS⁻ fraction = 1 / (1 + 10^(pKa - pH))
```

## 12. Sonraki Adımlar

1. ✅ MATLAB kodlarını anla
2. ✅ Veri formatlarını öğren
3. 🔄 Python'a ODE modeli port et
4. 🔄 Sentetik veri üret
5. ⏳ LSTM modeli eğit (PyTorch)
6. ⏳ Doğrulama ve karşılaştırma
