# H₂ Biyokimyasal Reaktif Transport Modellemesi: MATLAB'dan PyTorch'a Dönüşüm

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
├── src/              # Kaynak kod (MATLAB + PyTorch)
├── data/             # 12 eğitim dataset'i (kayaç/sıcaklık kombinasyonları)
├── results/          # Model çıktıları, figürler, parametreler
├── docs/             # Referanslar ve dokümantasyon
└── weekly_work/      # Gelişim snapshot'ları
```

**Detaylı açıklama**:
- [src/matlab/core/](src/matlab/core/): Ana MATLAB modeli (RNN, LSTM, ODE)
- [src/python/](src/python/): PyTorch implementasyonu
- [src/notebooks/](src/notebooks/): Jupyter notebook'lar
- [data/training/](data/training/): Basalt, Calcite, Gypsum, Sandstone verileri
- [results/trained_models/](results/trained_models/): Eğitilmiş model ağırlıkları
- [weekly_work/](weekly_work/): Her haftanın tam snapshot'ı

---

## 🚀 Hızlı Başlangıç

### Gereksinimler

**Python**:
```bash
pip install -r requirements.txt
```

**MATLAB**: Detaylar için [MATLAB_SETUP.md](MATLAB_SETUP.md) dosyasına bakın

### Çalıştırma

**PyTorch modeli**:
```bash
cd src/python
python matlab_to_pytorch_complete.py
```

**MATLAB modeli**:
```matlab
addpath(genpath('src/matlab/core'));
addpath(genpath('data/training'));
rnn_transport_multiguild_uq_v3
```

---

## 📊 Haftalık İlerleme Günlüğü

> **Not**: Her hafta yeni bölüm eklenecek - kronolojik sırayla

### Hafta 3 (23 Aralık 2025): PINN Finalizasyonu ✅

**Ne Yapıldı**:
- ✅ Biyokimyasal denklemler düzeltildi
- ✅ MATLAB'a karşı doğrulama yapıldı (parametreler %1 içinde)
- ✅ Karşılaştırma grafikleri oluşturuldu

**Sonuçlar**:
- MSE: 0.00038 (MATLAB: 0.00042)
- Parametre uyumu: %99.2

**Önemli Dosyalar**:
- [weekly_work/week_03_dec23_pinn_finalization/](weekly_work/week_03_dec23_pinn_finalization/)
- [weekly_work/week_03_dec23_pinn_finalization/README.md](weekly_work/week_03_dec23_pinn_finalization/README.md) (detaylı haftalık rapor)

---

### Hafta 2 (16 Aralık 2025): Keşif ve Deneyler ✅

**Ne Yapıldı**:
- ✅ ODE solver'ları test edildi (ode15s, ode23)
- ✅ Python kimya kütüphaneleri araştırıldı (ChemPy, Cantera)
- ✅ PINN prototipleri geliştirildi (6 farklı deney)

**Deneyler**:
1. **Task 1**: ODE15 solver testi → ODE23 daha kararlı
2. **Task 2**: ChemPy entegrasyonu → Başarısız (kütüphane limitleri)
3. **Task 4**: İlk PINN implementasyonu → Çalışıyor!
4. **Task 5**: PINN + normalizasyon → İyileşme
5. **Task 5b**: Sequence length=50 testi → Optimal
6. **Task 6**: Veri downsampling → Eğitim hızlandı

**Karar**: Standart LSTM, PINN'den daha iyi performans (bu dataset için)

**Önemli Dosyalar**:
- [weekly_work/week_02_dec16_exploration/](weekly_work/week_02_dec16_exploration/)
- [weekly_work/week_02_dec16_exploration/README.md](weekly_work/week_02_dec16_exploration/README.md) (detaylı haftalık rapor)

---

### Hafta 1 (9 Aralık 2025): MATLAB Baseline ✅

**Ne Yapıldı**:
- ✅ MATLAB'da RNN/LSTM implementasyonu
- ✅ Sandstone 25°C üzerinde eğitim
- ✅ İlk başarılı fit

**Sonuçlar**:
- MSE: 0.00042
- Eğitim süresi: 45 dakika (CPU)

**Önemli Dosyalar**:
- [weekly_work/week_01_dec09_matlab_baseline/](weekly_work/week_01_dec09_matlab_baseline/)
- [weekly_work/week_01_dec09_matlab_baseline/README.md](weekly_work/week_01_dec09_matlab_baseline/README.md) (detaylı haftalık rapor)

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
  author = {Hasan Tahsin Öztürk},
  title = {H₂ Biyokimyasal Reaktif Transport: MATLAB'dan PyTorch'a},
  year = {2025},
  url = {https://github.com/lynxrafu/chemical_thesis_repo}
}
```

---

## 📄 Lisans

MIT License - Detaylar için [LICENSE](LICENSE) dosyasına bakın

---

## 🔄 Son Güncellemeler

- **5 Ocak 2026**: Repo reorganizasyonu tamamlandı (modüler yapı)
- **23 Aralık 2025**: PINN finalizasyonu
- **16 Aralık 2025**: Keşif ve deney haftası
- **9 Aralık 2025**: İlk MATLAB baseline

---

## 📧 İletişim

Öğrenci: Hasan Tahsin Öztürk
