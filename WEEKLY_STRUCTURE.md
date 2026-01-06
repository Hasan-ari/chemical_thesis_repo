# Haftalık Çalışma Klasör Yapısı

## 📋 Sistem: Cumulative ISO + Konu Bazlı

### İsimlendirme Kuralı
```
YYYY-Www_konu_aciklamasi
```

**Örnekler:**
- `2026-W01_model_anlama` (30 Aralık 2025 - 5 Ocak 2026)
- `2026-W02_model_anlama_devam` 
- `2026-W03_python_cevirisi`
- `2026-W05_tum_kayaclar_python`
- `2026-W07_pytorch_lstm`

### Her Haftalık Klasörün İç Yapısı

```
2026-W01_model_anlama/
├── README.md              # Haftalık özet
├── code/                  # O hafta yazılan kodlar
│   ├── matlab/           # MATLAB denemeleri
│   └── python/           # Python denemeleri (varsa)
├── results/              # Çıktılar
│   ├── figures/          # Grafikler (.png, .pdf)
│   ├── tables/           # Tablolar (.xlsx, .csv)
│   └── data/             # Üretilen veri dosyaları
├── notes/                # Öğrenme notları
│   ├── parameters.md     # Parametre açıklamaları
│   ├── equations.md      # Denklem türetmeleri
│   └── questions.md      # Sorular ve cevaplar
└── PROGRESS.md           # Haftalık ilerleme takibi
```

---

## 📅 Haftalık Çalışma Planı (Örnek: İlk 10 Hafta)

### 🔵 HAFTA 1-2: Model Anlama
**Klasör:** `2026-W01_model_anlama`, `2026-W02_model_anlama_devam`

**Hedefler:**
- Sandstone 25°C v4 kodunu satır satır anlamak
- 14 state variable ve 28 parametreyi tanımlamak
- Henry yasası ve pH speciation'ı kavramak
- Diğer kayaçlarla karşılaştırma yapmak

**Çıktılar:**
- `notes/state_variables.md` - 14 değişkenin açıklaması
- `notes/parameters_detailed.md` - 28 parametrenin ne anlama geldiği
- `results/figures/sandstone_25C_annotated.png` - Açıklamalı grafik
- `results/tables/rock_comparison_25C.xlsx` - Kayaç karşılaştırması

---

### 🔵 HAFTA 3-4: Python'a Çevirme
**Klasör:** `2026-W03_python_cevirisi`, `2026-W04_python_validation`

**Hedefler:**
- v4 MATLAB kodunu Python'a çevirmek (scipy.integrate.solve_ivp)
- MATLAB vs Python doğrulama (RMSE < 0.1)
- Henry sabitleri ve kLa hesaplamalarını implement etmek

**Çıktılar:**
- `code/python/anaerobic_model_v4.py` - Ana Python sınıfı
- `code/python/test_vs_matlab.py` - Doğrulama scripti
- `results/figures/python_vs_matlab_comparison.png`
- `results/tables/validation_rmse.xlsx`
- `notes/scipy_vs_matlab_ode_solvers.md`

---

### 🔵 HAFTA 5-6: Tüm Kayaçlar Python'da
**Klasör:** `2026-W05_all_rocks_python`, `2026-W06_optimization`

**Hedefler:**
- 12 durum için Python kodu (loop ile)
- Parametre optimizasyonu (scipy.optimize.least_squares)
- Toplu görselleştirme (4x3 subplot)

**Çıktılar:**
- `code/python/run_all_rocks.py`
- `results/figures/all_12_conditions.png` (4 kayaç × 3 sıcaklık)
- `results/tables/rmse_summary_all.xlsx`
- `notes/optimization_strategy.md`

---

### 🔵 HAFTA 7-8: PyTorch Dataset Hazırlığı
**Klasör:** `2026-W07_pytorch_dataset`, `2026-W08_data_preprocessing`

**Hedefler:**
- PyTorch Dataset sınıfı oluşturmak
- Normalizasyon ve sequence oluşturma
- Train/val/test split stratejisi

**Çıktılar:**
- `code/python/dataset.py` - H2ProductionDataset sınıfı
- `code/python/preprocessing.py` - Veri ön işleme fonksiyonları
- `results/figures/data_distribution.png`
- `notes/pytorch_best_practices.md`

---

### 🔵 HAFTA 9-10: İlk LSTM Modeli
**Klasör:** `2026-W09_lstm_baseline`, `2026-W10_lstm_training`

**Hedefler:**
- Basit LSTM modelini kurmak
- GPU üzerinde eğitim
- İlk baseline sonuçları almak

**Çıktılar:**
- `code/python/lstm_model.py`
- `code/python/train.py`
- `results/figures/training_curves.png`
- `results/models/lstm_baseline_epoch50.pth`
- `notes/hyperparameters.md`

---

## 📝 README.md Template (Her Hafta İçin)

```markdown
# Hafta X (YYYY-Www): [Konu Başlığı]

**Tarih Aralığı**: DD Ay - DD Ay YYYY  
**Ana Hedef**: [Tek cümle ile bu haftanın hedefi]

---

## 🎯 Hedefler

- [ ] Hedef 1
- [ ] Hedef 2
- [ ] Hedef 3

---

## 📊 Ne Yapıldı?

### Gün 1-2: [Alt Başlık]
- Sandstone 25°C kodunu okudum
- State variables'ları anladım
- ...

### Gün 3-4: [Alt Başlık]
- Parametreleri Excel'e aktardım
- Henry sabitlerini hesapladım
- ...

### Gün 5-7: [Alt Başlık]
- Tüm kayaçları karşılaştırdım
- ...

---

## 🔬 Sonuçlar

### Başarılar ✅
- MATLAB kodunu %100 anladım
- Tüm parametrelerin anlamını öğrendim
- ...

### Zorluklar ⚠️
- pH speciation formülünü anlamak zaman aldı
- kLa değerlerinin fiziksel anlamı tam net değil
- ...

### Çözümler 💡
- Profesörün Henry yasası dokümanını okudum
- Literatürden kLa hesaplama makaleleri buldum
- ...

---

## 📁 Üretilen Dosyalar

### Kod
- `code/matlab/sandstone_25C_annotated.m` - Açıklamalı kod

### Sonuçlar
- `results/figures/H2_production_comparison.png`
- `results/tables/parameter_table.xlsx`

### Notlar
- `notes/state_variables_explained.md`
- `notes/questions_for_professor.md`

---

## 📌 Sonraki Hafta İçin

- [ ] Python'a çeviriye başla
- [ ] scipy.integrate.solve_ivp araştır
- [ ] Henry sabiti sıcaklık bağımlılığını implement et

---

## 🤔 Sorular / Belirsizlikler

1. **Soru 1**: kLa değerleri nasıl ölçülüyor?
   - **Cevap**: Literatürde empirik korelasyonlar var

2. **Soru 2**: Fe_pool başlangıç değeri nereden geliyor?
   - **Cevap**: TBD - Profesöre soracağım

---

**Git Commit Hash**: a1b2c3d (Bu haftanın commit'i)
```

---

## 📝 PROGRESS.md Template

```markdown
# İlerleme Takip - Hafta X

## Günlük Log

### Pazartesi (DD/MM)
- ⏰ 09:00-12:00: Sandstone kodunu okudum (3 saat)
- ✅ State variables listesini çıkardım
- 📝 Not: pH hesaplaması karmaşık, daha fazla zaman gerek

### Salı (DD/MM)
- ⏰ 10:00-15:00: Parametreleri analiz ettim (5 saat)
- ✅ 28 parametrenin Excel tablosunu oluşturdum
- ⚠️ Problem: Bazı parametrelerin fiziksel anlamı belirsiz

### Çarşamba (DD/MM)
- ⏰ 13:00-18:00: Henry yasası implementasyonunu anladım (5 saat)
- ✅ Sıcaklık bağımlılığı formüllerini çıkardım
- 💡 Fikir: Van't Hoff denklemi kullanılmış

...

---

## Toplam Çalışma Saati: XX saat

## Hedeflere Ulaşma Durumu

- ✅ Hedef 1: Tamamlandı (%100)
- 🔄 Hedef 2: Devam ediyor (%60)
- ⏳ Hedef 3: Başlamadı (%0)
```

---

## 🔄 Cumulative Yaklaşım

**İlke:** Her hafta bir öncekinin devamıdır.

### Örnek Akış:

**Hafta 1 (2026-W01_model_anlama):**
```
code/matlab/
  └── sandstone_25C_annotated.m  (açıklamalı MATLAB kodu)
notes/
  └── state_variables.md  (14 değişken açıklaması)
```

**Hafta 2 (2026-W02_model_anlama_devam):**
```
code/matlab/
  └── all_rocks_comparison.m  (4 kayacı karşılaştıran kod)
notes/
  └── parameters_detailed.md  (28 parametre açıklaması)
  └── henry_law_notes.md  (yeni eklendi)
```

**Hafta 3 (2026-W03_python_cevirisi):**
```
code/
  ├── matlab/  (önceki haftalardakiler)
  └── python/  (YENİ!)
      └── anaerobic_model_v4.py
notes/
  └── ...  (öncekiler + yeni Python notları)
```

**👉 Her hafta kendi içinde self-contained ama önceki haftaların üzerine inşa ediyor.**

---

## ⚙️ Git Workflow

### Her Hafta Sonunda:

```bash
# 1. Haftalık çalışmaları commit'le
git add 2026-W01_model_anlama/
git commit -m "feat(week1): Model anlama tamamlandı

- Sandstone 25°C kodunu anladım
- 14 state variable ve 28 parametre açıklandı  
- Tüm kayaçlar karşılaştırıldı

Closes #week1
"

# 2. Tag ekle (opsiyonel)
git tag -a week1-complete -m "Hafta 1 tamamlandı"

# 3. Push
git push origin main --tags
```

---

## 🎯 Sonraki Adımlar

1. **İlk haftalık klasörü oluştur:**
   ```bash
   mkdir 2026-W01_model_anlama
   cd 2026-W01_model_anlama
   mkdir code results notes
   touch README.md PROGRESS.md
   ```

2. **README.md'yi template ile doldur**

3. **Çalışmaya başla!** 🚀

---

**Oluşturulma Tarihi**: 5 Ocak 2026  
**Son Güncelleme**: 5 Ocak 2026
