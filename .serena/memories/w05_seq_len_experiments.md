# W05 - Sequence Length Experiments

## 1. İlk Deney: 500pts ile seq_len Threshold (Tamamlandı)

### Lokasyon
`2026_W05_seq_len_experiment/app/`

### Soru
> "seq_len'i minimum kaça düşürebiliriz ki model hala iyi trajectory üretebilsin?"

### Deney Matrisi
- **Veri:** 500 nokta, 14 state değişkeni
- **Test edilen seq_len:** [50, 30, 20, 10, 5]
- **Target loss:** 1e-8 (strong overfit)

### Sonuçlar

| seq_len | Training Loss | RMSE | Durum |
|---------|---------------|------|-------|
| 50 | 1.85e-06 | 0.148 | ✅ Çalışıyor |
| **30** | 3.69e-06 | **0.063** | ✅ **En iyi** |
| 20 | 1.90e-06 | 0.096 | ✅ Çalışıyor |
| 10 | 1.16e-05 | 0.307 | ✅ Sınırda |
| 5 | 4.88e-06 | 0.560 | ❌ Başarısız (RMSE > 0.5) |

### Ana Bulgular
1. **Minimum çalışan seq_len = 10**
2. **Optimal seq_len = 30** (seq_len=50'den daha iyi!)
3. **En problemli değişkenler:** Acetate (RMSE=1.75), X/biyokütle (RMSE=1.07)
4. **En kararlı değişkenler:** HCO3, FeS, Lag, Fe_pool (RMSE < 0.002)

### Çıktılar
- `outputs/seq_len_experiment/` - model ve trajectory dosyaları
- `figures/all_variables/` - tüm değişkenler için grafikler
- `figures/all_variables/rmse_heatmap.png` - özet heatmap

---

## 2. İkinci Deney: n_points vs seq_len Scaling (Bekliyor)

### Lokasyon
`2026_W05_npoints_seq_len_experiment/app/`

### Soru
> "Veri boyutu arttıkça, minimum çalışan seq_len nasıl değişiyor?"

### Hipotez
```
n=500  → min_seq_len ≈ 10 (ilk deneyden biliniyor)
n=1000 → min_seq_len ≈ 5?
n=2500 → min_seq_len ≈ 3?
```

### Deney Matrisi

| n_points | Test edilecek seq_len | Training samples (seq_len=10) |
|----------|----------------------|-------------------------------|
| 500 | [50, 30, 20, 10, 5] | 490 |
| 1000 | [50, 30, 20, 10, 5, 3] | 990 |
| 2500 | [50, 30, 20, 10, 5, 3, 2] | 2490 |

### Çalıştırma
```bash
cd 2026_W05_npoints_seq_len_experiment/app
source .venv/bin/activate
python -m experiment.run_npoints_experiment
```

### Beklenen Çıktılar
- `outputs/n500/`, `outputs/n1000/`, `outputs/n2500/` - her n_points için sonuçlar
- `figures/scaling_analysis.png` - n_points vs min_seq_len scaling grafiği
- `figures/summary_heatmap.png` - tüm kombinasyonların RMSE heatmap'i

---

## 3. Klasör Yapısı

```
chemical_thesis_repo/
├── 2026_W05_seq_len_experiment/          ← İlk deney (tamamlandı)
│   └── app/
│       ├── data/basalt_25c_lstm_input_500pts.npy
│       ├── outputs/seq_len_experiment/
│       ├── figures/all_variables/
│       └── src/lstm_experiment/
│           ├── run_experiment.py
│           ├── plot_all_variables.py
│           └── plot_seq_len_5.py
│
└── 2026_W05_npoints_seq_len_experiment/  ← İkinci deney (bekliyor)
    └── app/
        ├── data/
        │   ├── basalt_25c_lstm_input_500pts.npy
        │   ├── basalt_25c_lstm_input_1000pts.npy
        │   └── basalt_25c_lstm_input_2500pts.npy
        ├── outputs/
        ├── figures/
        └── src/experiment/
            └── run_npoints_experiment.py
```

---

## 4. Başarı Kriterleri (Her İki Deney İçin)

1. **Training loss** < 1e-7 (target_loss × 10)
2. **Trajectory RMSE** < 0.5
3. **No collapse** (NaN, Inf yok)

---

## 5. Tez İçin Önem

Bu deneyler şu soruları yanıtlıyor:
1. LSTM surrogate model için **minimum context window** ne kadar?
2. **Veri miktarı** ile **gerekli context** arasındaki ilişki nedir?
3. Hangi kimyasal değişkenler **daha zor** tahmin ediliyor?

Scaling law bulunursa: "X nokta veri ile Y seq_len yeterli" pratik kuralı çıkarılabilir.
