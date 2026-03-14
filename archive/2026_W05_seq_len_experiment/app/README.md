# W05 - Sequence Length Threshold Experiment

## Amaç

CS hocamızın istediği: **Minimum seq_len değerini bulmak** - overfit olmadan ne kadar düşürebiliriz?

## Deney Tasarımı

```
seq_len = 50 → 30 → 20 → 10 → 5
```

Her değer için:
1. LSTM'i overfit et (target_loss = 1e-8)
2. Autoregressive trajectory üret
3. RMSE hesapla
4. Collapse kontrolü yap

## Kurulum

```bash
cd 2026_W05_seq_len_experiment/app

# Virtual environment oluştur (uv ile)
uv venv

# Bağımlılıkları yükle
uv pip install -e .
```

## Çalıştırma

```bash
# Activate venv
source .venv/bin/activate

# Deneyi çalıştır
python -m lstm_experiment.run_experiment

# Veya özel parametrelerle
python -m lstm_experiment.run_experiment --seq_lengths 50 30 20 10 5 --epochs 10000
```

## Çıktılar

```
outputs/seq_len_experiment/
├── config.json              # Deney konfigürasyonu
├── all_results.json         # Tüm sonuçlar
├── comparison.png           # Karşılaştırma grafikleri
├── trajectories.png         # Trajectory grafikleri
├── experiment_*.log         # Log dosyası
│
├── seq_len_50/
│   ├── model.pt             # Eğitilmiş model
│   ├── result.json          # Sonuçlar
│   ├── trajectory.npz       # Üretilen trajectory
│   └── history.json         # Training history
│
├── seq_len_30/
│   └── ...
│
└── seq_len_20/
    └── ...
```

## Başarı Kriterleri

| Kriter | Değer |
|--------|-------|
| Training Loss | < 1e-8 |
| Trajectory RMSE | < 0.5 |
| Collapse | Yok |

## Beklenen Sonuç

seq_len küçüldükçe:
- Daha fazla training sample (iyi)
- Daha az context (kötü)
- Bir noktada model öğrenemez → overfit başarısız
