# W05 - Sequence Length Experiments (Güncel)

## Özet

Bu hafta iki ana deney yapıldı:
1. **500pts ile seq_len threshold** - ✅ Tamamlandı (local Mac)
2. **n_points vs seq_len scaling** - 🔄 Colab'da çalışıyor (A100 GPU)

---

## 1. İlk Deney: 500pts ile seq_len Threshold (✅ Tamamlandı)

### Lokasyon
`2026_W05_seq_len_experiment/app/`

### Soru
> "seq_len'i minimum kaça düşürebiliriz ki model hala iyi trajectory üretebilsin?"

### Sonuçlar

| seq_len | RMSE | Durum |
|---------|------|-------|
| 50 | 0.148 | ✅ |
| **30** | **0.063** | ✅ **En iyi** |
| 20 | 0.096 | ✅ |
| 10 | 0.307 | ✅ Sınırda |
| 5 | 0.560 | ❌ RMSE > 0.5 |

### Ana Bulgular
- **Minimum çalışan seq_len = 10**
- **Optimal seq_len = 30** (50'den daha iyi!)
- **En zor değişkenler:** Acetate, X (biyokütle)
- **En kolay değişkenler:** HCO3, FeS, Lag, Fe_pool

### Çıktılar
```
outputs/seq_len_experiment/
├── seq_len_{50,30,20,10,5}/
│   ├── model.pt
│   ├── result.json
│   └── trajectory.npz
└── all_results.json

figures/all_variables/
├── rmse_heatmap.png
├── var_XX_*.png (14 adet)
└── summary_key_variables.png
```

---

## 2. İkinci Deney: n_points vs seq_len Scaling (🔄 Colab'da Çalışıyor)

### Lokasyon
`2026_W05_npoints_seq_len_experiment/app/`

### Soru
> "Veri boyutu arttıkça, minimum çalışan seq_len nasıl değişiyor?"

### Deney Matrisi

| n_points | Test edilen seq_len |
|----------|---------------------|
| 500 | [50, 30, 20, 10, 5] |
| 1000 | [50, 30, 20, 10, 5, 3] |
| 2500 | [50, 30, 20, 10, 5, 3, 2] |

### Ara Sonuçlar (n=1000, Mac'te yarım kaldı)

| seq_len | RMSE | Durum |
|---------|------|-------|
| 50 | 0.193 | ✅ |
| 30 | 0.262 | ✅ |
| 20 | 0.299 | ✅ |
| 10 | 0.956 | ❌ |
| 5 | 0.540 | ❌ |
| 3 | 1.577 | ❌ |

**İlginç bulgu:** n=1000'de seq_len=10 çalışmıyor ama n=500'de çalışıyordu!

### Dosyalar
```
2026_W05_npoints_seq_len_experiment/app/
├── colab_experiment.py      ← Colab için ana script
├── plot_results.py          ← Figür oluşturma
├── data/
│   ├── basalt_25c_lstm_input_500pts.npy
│   ├── basalt_25c_lstm_input_1000pts.npy
│   └── basalt_25c_lstm_input_2500pts.npy
└── outputs_colab/           ← Sonuçlar buraya kaydedilecek
```

### Colab'da Çalıştırma
```python
# 1. data/ klasörüne dosyaları yükle (sürükle-bırak):
#    - colab_experiment.py
#    - plot_results.py
#    - basalt_25c_lstm_input_500pts.npy
#    - basalt_25c_lstm_input_1000pts.npy
#    - basalt_25c_lstm_input_2500pts.npy

# 2. Runtime → GPU seç (A100 ideal)

# 3. Deneyi çalıştır
%run data/colab_experiment.py

# 4. Figürleri oluştur
%run data/plot_results.py outputs_colab

# 5. Sonuçları indir
!zip -r results.zip outputs_colab/
from google.colab import files
files.download("results.zip")
```

---

## 3. Model Mimarisi

```
Input: (batch, seq_len, 14)
         │
         ▼
┌─────────────────────┐
│ LSTM Layer 1: 14→128│
│ LSTM Layer 2: 128→128│  ← 2 katman, sigmoid+tanh (içeride)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Linear: 128→14      │  ← Aktivasyon YOK
└─────────────────────┘
         │
         ▼
Output: (batch, 14)
```

**Linear layer (128→14):**
- Matris çarpımı: (1×128) @ (128×14) = (1×14)
- Her çıktı = 128 girdinin ağırlıklı toplamı
- Parametreler: 128×14 + 14 bias = 1806

---

## 4. Başarı Kriterleri

| Kriter | Değer |
|--------|-------|
| RMSE threshold | < 0.5 |
| Collapse | NaN/Inf yok |

---

## 5. Öğrenilenler

1. **seq_len=30 optimal** - Daha uzun her zaman daha iyi değil
2. **Acetate ve X en zor** - Karmaşık büyüme dinamikleri
3. **A100 GPU** - Mac MPS'den ~6x hızlı
4. **Thermal pressure: Moderate** - Mac için normal

---

## 6. Oluşturulan Figürler

İlk deney (500pts):
- `figures/all_variables/rmse_heatmap.png`
- `figures/all_variables/var_XX_*.png`
- `figures/seq_len_5_detailed.png`

İkinci deney (Colab sonrası):
- `outputs_colab/scaling_analysis.png`
- `outputs_colab/rmse_heatmap.png`
- `outputs_colab/trajectories_n{500,1000,2500}.png`
- `outputs_colab/rmse_per_variable.png`
