**Tez Projesi**: Biyokimyasal Reaktif Transport için Derin Öğrenme
**Dönem**: Aralık 2025 - Devam Ediyor
**Son Güncelleme**: 5 Ocak 2026

---

## 📋 Genel Bakış

Bu proje, yer altı hidrojen (H₂) depolamadaki biyokimyasal reaksiyonları modellemek için LSTM tabanlı sinir ağlarını kullanıyor.

**Hedefler**:
- MATLAB RNN/LSTM modelini PyTorch'a dönüştürme
- 4 kayaç türü × 3 sıcaklıkta (25°C, 34°C, 40°C) eğitim
- Physics-Informed Neural Networks (PINN) araştırması
- Parametre optimizasyonu ve karşılaştırma

---

## 📁 Repo Yapısı

```
chemical_thesis_repo/
├── CURRENT/              # ⭐ ŞU AN ÇALIŞILACAK MATERYALLER (v4 kodları)
├── shared/               # Ortak kullanılan kaynaklar
│   └── datasets/         # 12 deneysel veri seti (Muller 2024)
├── archive/              # Eski çalışmalar (v1, v2, v3 - KULLANMA!)
└── [haftalık klasörler]  # Gelecekte: 2025-W50, 2025-W51, vb.
```

### CURRENT/ - Şu An Çalışılacak Dosyalar

** En önemli klasör!** 3 Ocak 2026 akşam mailinden:

```
CURRENT/
├── code/v4_two_phase/     # 12 MATLAB kodu (.m + .mat fitted params)
│   ├── sandstone_25C/
│   ├── sandstone_34C/
│   ├── sandstone_40C/
│   ├── basalt_25C/
│   ├── basalt_34C/
│   ├── basalt_40C/
│   ├── calcite_25C/
│   ├── calcite_34C/
│   ├── calcite_40C/
│   ├── gypsum_25C/
│   ├── gypsum_34C/
│   └── gypsum_40C/
├── data/muller_2024/      # Deneysel veriler (.txt)
├── results/fitted_outputs/ # Fit sonuçları (.dat + .png)
└── docs/                  # Açıklama dokümanları
```

**v4 İki Fazlı Model Özellikleri**:
- ✅ 14 state variables (v3'te 13'tü)
- ✅ 28 parameters (v3'te 13'tü)
- ✅ İki fazlı sistem (gaz + sıvı)
- ✅ Henry yasası + pH türlenmesi
- ✅ 12 başarılı fit

👉 **Detaylar için**: [CURRENT/README.md](CURRENT/README.md)

### 📊 shared/datasets/ - Ortak Veri Setleri

Tüm haftalık çalışmalarda kullanılacak 12 deneysel veri seti:

```
shared/datasets/muller_2024/
├── basalt/      # 25C, 34C, 40C (.dat)
├── calcite/     # 25C, 34C, 40C (.dat)
├── gypsum/      # 25C, 34C, 40C (.dat)
└── sandstone/   # 25C, 34C, 40C (.dat)
```

👉 **Detaylar için**: [shared/datasets/README.md](shared/datasets/README.md)

### 🗄️ archive/ - Eski Çalışmalar

❌ **KULLANMA!** Sadece referans için:
- `reactions_old/`: Eski tek fazlı modeller (v1, v2, v3)
- Yanlış fizik içeriyor - profesör v4 ile güncelledi

👉 **Detaylar için**: [archive/README.md](archive/README.md)

---

## 🚀 Hızlı Başlangıç

### Gereksinimler

**Python**:
```bash
pip install -r requirements.txt
```

**MATLAB**: Detaylar için [MATLAB_SETUP.md](MATLAB_SETUP.md) dosyasına bakın

### Çalıştırma

**v4 MATLAB modeli** (Şu an çalışılması gereken):
```matlab
cd CURRENT/code/v4_two_phase/sandstone_25C
anaerobic_model_two_phase_mixedSR_25C_v4
```

**Tüm 12 durumu görüntüleme**:
```matlab
% 12 folder'ı tek tek aç ve çalıştır
% Her biri .m ve .mat dosyası içerir
```

---

## 📊 Haftalık İlerleme Günlüğü

> **Not**: ISO hafta bazlı klasörleme sistemi gelecekte eklenecek (2025-W50, W51, W52, 2026-W01)

### 🔄 5 Ocak 2026: Repo Reorganizasyonu

**Ne Yapıldı**:
- ✅ CURRENT/ klasörü oluşturuldu (profesörün v4 kodları)
- ✅ shared/datasets/ ortak veri klasörü hazırlandı
- ✅ archive/ eski çalışmalar arşivlendi
- ✅ 12 rock/temperature kombinasyonu organize edildi

---

### 🗄️ Eski Haftalık Çalışmalar (Arşivlendi)

**Not**: Aşağıdaki çalışmalar v1/v2/v3 (tek fazlı, yanlış fizik) ile yapılmıştır.
Profesör 3 Ocak'ta v4 (iki fazlı, doğru fizik) gönderdi.

**Hafta 3** (23 Aralık): PINN finalizasyonu → `archive/reactions_old/`
**Hafta 2** (16 Aralık): Keşif ve deneyler → `archive/reactions_old/`
**Hafta 1** (9 Aralık): MATLAB baseline → `archive/reactions_old/`

👉 **Eski çalışma detayları**: Eğer gerekirse `archive/reactions_old/` içinde mevcuttur

---

## 📚 Dataset Açıklaması

**Kaynak**: Muller et al. 2024 (Low pressure H₂ storage experiments)

**Kayaç Türleri**:
- Basalt (bazik volkanik)
- Calcite (karbonat)
- Gypsum (sülfat)
- Sandstone (kumtaşı)

**Sıcaklıklar**: 25°C, 34°C, 40°C
**Basınç**: 2 bar (düşük basınç deneyleri)
**State Variables**: H₂, CO₂, CH₄, H₂S, SO₄, FeS, X_meth, X_sulf, X_aceto, Acetate

---

## 📖 Alıntı

Bu kodu kullanıyorsanız:

```bibtex
@software{h2_reactive_transport_2025,
  author = {Hasan Arı},
  title = {H₂ Biyokimyasal Reaktif Transport: MATLAB'dan PyTorch'a},
  year = {2026},
  url = {https://github.com/lynxrafu/chemical_thesis_repo}
}
```

---

## 📄 Lisans

MIT License - Detaylar için [LICENSE](LICENSE) dosyasına bakın

---

## 🔄 Son Güncellemeler

- **5 Ocak 2026**: Repo reorganizasyonu - CURRENT/ (v4), shared/datasets/, archive/ eklendi
- **3 Ocak 2026**: Profesör v4 two-phase model gönderdi (12 başarılı fit)
- **23 Aralık 2025**: PINN finalizasyonu (v3 ile, arşivlendi)
- **16 Aralık 2025**: Keşif ve deney haftası (v2/v3 ile, arşivlendi)
- **9 Aralık 2025**: İlk MATLAB baseline (v1 ile, arşivlendi)

---

## 📧 İletişim

Öğrenci: Hasan Arı
