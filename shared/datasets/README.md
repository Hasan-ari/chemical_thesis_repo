# Shared Datasets

Bu klasör tüm haftalık çalışmalarda ve CURRENT/ klasöründe kullanılan ortak deneysel veri setlerini içerir.

## Muller 2024 Dataset

**Kaynak**: Muller et al. (2024) - Experimental H₂ production data

**Format**: `.dat` files (tab-separated values)

**İçerik**: 12 farklı deneysel koşul

| Kayaç | 25°C | 34°C | 40°C |
|-------|------|------|------|
| **Basalt** | ✅ H2_Basalt_at_25C.dat | ✅ H2_Basalt_at_34C.dat | ✅ H2_Basalt_at_40C.dat |
| **Calcite** | ✅ H2_Calcite_at_25C.dat | ✅ H2_Calcite_at_34C.dat | ✅ H2_Calcite_at_40C.dat |
| **Gypsum** | ✅ H2_Gypsum_at_25C.dat | ✅ H2_Gypsum_at_34C.dat | ✅ H2_Gypsum_at_40C.dat |
| **Sandstone** | ✅ H2_Sandstone_at_25C.dat | ✅ H2_Sandstone_at_34C.dat | ✅ H2_Sandstone_at_40C.dat |

### Klasör Yapısı
```
shared/datasets/muller_2024/
├── basalt/
│   ├── H2_Basalt_at_25C.dat
│   ├── H2_Basalt_at_34C.dat
│   └── H2_Basalt_at_40C.dat
├── calcite/
│   ├── H2_Calcite_at_25C.dat
│   ├── H2_Calcite_at_34C.dat
│   └── H2_Calcite_at_40C.dat
├── gypsum/
│   ├── H2_Gypsum_at_25C.dat
│   ├── H2_Gypsum_at_34C.dat
│   └── H2_Gypsum_at_40C.dat
└── sandstone/
    ├── H2_Sandstone_at_25C.dat
    ├── H2_Sandstone_at_34C.dat
    └── H2_Sandstone_at_40C.dat
```

## Kullanım

### Python'da
```python
import pandas as pd

# Örnek: Sandstone 25°C verisi
data = pd.read_csv('shared/datasets/muller_2024/sandstone/H2_Sandstone_at_25C.dat', 
                   sep='\t')
```

### MATLAB'da
```matlab
% Örnek: Basalt 34°C verisi
data = readmatrix('shared/datasets/muller_2024/basalt/H2_Basalt_at_34C.dat');
```

## Notlar

- Bu veriler **tüm haftalık çalışmalarda ortak referans** olarak kullanılır
- CURRENT/ klasöründeki kodlar bu verileri kullanır
- Veriler değiştirilmemeli (read-only)
- Yeni deneysel veriler eklendiğinde bu klasöre eklenir

---

**Son Güncelleme**: 5 Ocak 2026  
**Kaynak**: reactions/dataset_for_training_different_rocks_at_25C-34C-40C/
