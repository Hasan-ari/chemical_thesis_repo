# N_POINTS vs SEQ_LEN Experiment

## Araştırma Sorusu

> Veri boyutu (n_points) arttıkça, minimum çalışan seq_len nasıl değişiyor?

## Hipotez

```
Daha fazla veri noktası → Daha fazla training sample → Daha düşük seq_len yeterli

n=500  → min_seq_len ≈ 10 (W05 ilk deneyden biliyoruz)
n=1000 → min_seq_len ≈ 5?
n=2500 → min_seq_len ≈ 3?
```

## Deney Matrisi

| n_points | Test edilecek seq_len | Training samples (seq_len=10) |
|----------|----------------------|-------------------------------|
| 500 | [50, 30, 20, 10, 5] | 490 |
| 1000 | [50, 30, 20, 10, 5, 3] | 990 |
| 2500 | [50, 30, 20, 10, 5, 3, 2] | 2490 |

## Klasör Yapısı

```
2026_W05_npoints_seq_len_experiment/
└── app/
    ├── data/
    │   ├── basalt_25c_lstm_input_500pts.npy
    │   ├── basalt_25c_lstm_input_1000pts.npy
    │   └── basalt_25c_lstm_input_2500pts.npy
    │
    ├── outputs/
    │   ├── n500/
    │   │   ├── seq_len_50/
    │   │   ├── seq_len_30/
    │   │   └── ...
    │   ├── n1000/
    │   └── n2500/
    │
    ├── figures/
    │   ├── scaling_analysis.png
    │   └── summary_heatmap.png
    │
    └── src/experiment/
        └── run_npoints_experiment.py
```

## Çalıştırma

```bash
cd app
uv venv && source .venv/bin/activate
uv pip install -e .
python -m experiment.run_npoints_experiment
```

## Beklenen Çıktılar

1. **scaling_analysis.png**: n_points vs min_seq_len ilişkisi
2. **summary_heatmap.png**: Tüm kombinasyonların RMSE heatmap'i
3. **all_results.json**: Tüm deney sonuçları

## Başarı Kriterleri

1. Training loss < 1e-7
2. Trajectory RMSE < 0.5
3. No collapse (NaN, Inf)
