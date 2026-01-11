# LSTM Recursive Forecast - Multi-Step Horizon

## Hocamın İstekleri ve Uygulama

### Toplantı Notları (Özet)
```
- Chain correction yaparak gidelim
- 3 seferde gibi (10, 20, 30 step)
- Her adıma bakmayalım
- 100 adım sequence length
- 150 adım tahmin yapsın
- Random yerden alabiliriz
- Step size fixlensin
- Uzadıkça kopma olacak mı kontrol
```

---

## Problem: Identity Mapping

### Neden 1-Step Prediction Çalışmıyor?

```
dt = 20 gün / 2500 nokta = 0.008 gün/step

y(t+1) ≈ y(t) + 0.008 × dy/dt

Değişim çok küçük olduğu için:
- Model: "y(t+1) ≈ y(t)" öğreniyor
- Sonuç: Düz çizgi (identity mapping)
```

### Çözüm: Multi-Step Horizon

```
Horizon = 10:
  y(t+10) = y(t) + 10 × dt × dy/dt
  Değişim 10x daha büyük → Model dynamics öğrenmek ZORUNDA

Horizon = 20:
  Değişim 20x daha büyük

Horizon = 30:
  Değişim 30x daha büyük
```

---

## Uygulama Detayları

### 1. Config

```python
SEQ_LEN = 100              # Hocam: 100 adım sequence
PRED_HORIZON = 10          # Default horizon
HORIZONS_TO_TEST = [10, 20, 30]  # 3 senaryo
FORECAST_STEPS = 150       # Hocam: 150 adım tahmin
```

### 2. Sequence Creation

```python
# Eski (1-step): 
Y = data[i + seq_len]              # t+1

# Yeni (multi-step):
Y = data[i + seq_len + horizon - 1]  # t+horizon
```

### 3. Recursive Forecast

```python
# Her chain adımında:
1. Context (100 adım) → Model → t+horizon tahmini
2. Tahmin context'e eklenir
3. Context 1 adım kayar
4. Tekrar tahmin
5. 150 adım tamamlanana kadar devam
```

### 4. Karşılaştırmalı Analiz

```
Her horizon için:
- Ayrı model eğit
- Aynı başlangıç noktasından forecast
- RMSE hesapla
- En iyi horizon belirle
```

---

## Çıktılar

| Dosya | Açıklama |
|-------|----------|
| `horizon_comparison.png` | 3 horizon karşılaştırma grafiği |
| `horizon_comparison_report.txt` | Detaylı RMSE raporu |
| `lstm_horizon_XX.pt` | En iyi modelin ağırlıkları |
| `scaler.pkl` | Normalizasyon scaler |

---

## Neden Bu Yaklaşım Mantıklı?

### 1. Fiziksel Anlamlılık
- ODE sistemleri derivatives (dy/dt) ile tanımlanır
- Küçük dt'de Δy ≈ 0, model derivatives öğrenemiyor
- Büyük horizon'da Δy anlamlı, dynamics görünür

### 2. Akademik Geçerlilik
- Multi-step direct prediction vs iterative prediction
- Zaman serisi literatüründe standart yaklaşım
- Fair comparison için aynı test koşulları

### 3. Hocamın Yönlendirmesi
- "10 step, 20 step gidelim" → Multi-horizon testi
- "Uzadıkça kopma olacak mı" → Divergence analizi
- "3 seferde gibi" → 3 farklı senaryo

---

## Sonraki Adımlar

1. **Bu kodu çalıştır** → En iyi horizon belirle
2. **Delta Learning ile karşılaştır** → Hangi yaklaşım daha iyi?
3. **Physics-Informed Loss** → Fizik kısıtlamaları ekle

---

## Çalıştırma

```bash
# Local
python lstm_recursive_forecast.py

# Colab
%run lstm_recursive_forecast.py
```

---

## Dosya Yapısı

```
lstm_recursive_forecast/
├── lstm_recursive_forecast.py   # Ana kod (multi-horizon)
├── run_in_colab.ipynb           # Colab notebook
├── README.md                    # Bu dosya
└── (outputs after running)
    ├── chain_test.png
    ├── horizon_comparison.png
    └── ...
```

---

**Son güncelleme:** 2026-W02
