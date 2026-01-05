# CURRENT - Şu An Çalışılacak Materyaller

⭐ **Bu klasördeki dosyalar profesörün 3 Ocak 2026 akşam mailinden (107 dosya)**

## 🎯 Ne ile çalışmalısın?

### v4 İki Fazlı Model (Two-Phase)
**Kod Versiyonu**: v4 (en güncel)

**Özellikler**:
- 14 state variables (v3'te 13'tü)
- 28 parameters (v3'te 13'tü)
- İki fazlı (gaz + sıvı) sistem
- Henry yasası implementasyonu
- pH bağımlı sülfür türlenmesi
- Fe pool limitation (Gypsum için kritik)

### 📊 12 Başarılı Fit

| Kayaç | 25°C | 34°C | 40°C |
|-------|------|------|------|
| **Sandstone** | ✅ | ✅ | ✅ |
| **Basalt** | ✅ | ✅ | ✅ |
| **Calcite** | ✅ | ✅ | ✅ |
| **Gypsum** | ✅ | ✅ | ✅ |

**Her durum için** (örnek: `code/v4_two_phase/sandstone_25C/`):
- `anaerobic_model_two_phase_mixedSR_25C_v4.m` - v4 kodu
- `best_fit_params_Sandstone_25C.mat` - Fitted parametreler
- Deneysel veri: `data/muller_2024/Muller_2024_H2_Sandstone_at_25C.txt`
- Sonuçlar: `results/fitted_outputs/` klasöründe

### 🚀 Sıradaki Adımlar

#### HAFTA 1-2: Anlama
```bash
1. Sandstone 25°C kodunu aç:
   CURRENT/code/v4_two_phase/sandstone_25C/anaerobic_model_two_phase_mixedSR_25C_v4.m

2. Satır satır incele:
   - 14 state variable nedir?
   - 28 parameter ne anlama geliyor?
   - Henry yasası nasıl implement edilmiş?
   - pH türlenmesi nasıl çalışıyor?

3. MATLAB'da çalıştır ve figürleri incele
```

#### HAFTA 3-4: Python Çevirisi
```bash
1. v4'ü Python'a çevir:
   - 14 state ODE system
   - 28 parameter
   - scipy.integrate.solve_ivp (ode15s equivalent)

2. Sandstone 25°C için test et:
   - MATLAB vs Python RMSE < 0.1
```

#### AY 2-3: Tüm Kayaçlar + GPU
```bash
1. 12 durum için Python kodu
2. PyTorch + CUDA acceleration
3. LSTM/PINN eğitimi hazırlığı
```

### ⚠️ KULLANMA!

**Eski Versiyonlar:**
- ❌ v1: Tek fazlı model (yanlış fizik)
- ❌ v2: Tek fazlı model (yanlış fizik)
- ❌ v3: Tek fazlı model + pH türlenmesi (hala yanlış)
- ✅ **v4: İki fazlı model (DOĞRU!)** ← BU İLE ÇALIŞ

### 📚 Önemli Dokümanlar

`docs/` klasöründe:
1. **henrys_law_calculations.docx**: Henry sabitleri hesaplamaları
2. **addition_of_Fe_pool.docx**: Fe havuzu neden eklendi?
3. **explanations_to_fit.docx**: Fit açıklamaları

### 📂 Klasör Yapısı

```
CURRENT/
├── code/v4_two_phase/
│   ├── sandstone_25C/   [.m + .mat]
│   ├── sandstone_34C/   [.m + .mat]
│   ├── sandstone_40C/   [.m + .mat]
│   ├── basalt_25C/      [.m + .mat]
│   ├── basalt_34C/      [.m + .mat]
│   ├── basalt_40C/      [.m + .mat]
│   ├── calcite_25C/     [.m + .mat]
│   ├── calcite_34C/     [.m + .mat]
│   ├── calcite_40C/     [.m + .mat]
│   ├── gypsum_25C/      [.m + .mat]
│   ├── gypsum_34C/      [.m + .mat]
│   └── gypsum_40C/      [.m + .mat]
├── data/muller_2024/    [12 .txt dosyası]
├── results/fitted_outputs/  [.dat + .png]
└── docs/                [3 .docx]
```

### 🎓 Profesörden Mesaj

> "Artık tüm kayaçlar için algoritmaları standard hale getirdim ki **senin işin daha kolay olsun**."

**Ne anlama geliyor?**
- Fiziksel model tamamen doğru ✅
- 12 durum için fitted parametreler ✅
- Kod standardize ✅
- **Sadece neural network eğitimi kaldı!** ✅

---

**Son Güncelleme**: 5 Ocak 2026
**Kaynak**: Profesör maili (3 Ocak 2026, 19:18)
