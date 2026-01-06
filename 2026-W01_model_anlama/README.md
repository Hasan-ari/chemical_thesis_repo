# Hafta 1 (2026-W01): Model Anlama

**Tarih Aralığı**: 30 Aralık 2025 - 5 Ocak 2026
**Ana Hedef**: v4 iki fazlı modelini tam olarak anlamak

---

## 🎯 Hedefler

- [ ] Sandstone 25°C v4 kodunu satır satır okumak
- [ ] 14 state variable'ı tanımlamak ve anlamak
- [ ] 28 parametreyi listelemek ve açıklamak
- [ ] Henry yasası implementasyonunu kavramak
- [ ] pH bağımlı sülfür türlenmesini anlamak
- [ ] 4 kayacı aynı sıcaklıkta (25°C) karşılaştırmak

---

## 📊 Ne Yapıldı?

### Organizasyon ve Hazırlık (5 Ocak)
- ✅ Repo tamamen reorganize edildi
- ✅ CURRENT/ klasörü oluşturuldu (v4 kodları)
- ✅ shared/datasets/ oluşturuldu (12 .dat dosyası)
- ✅ archive/ oluşturuldu (eski v1/v2/v3 kodları)
- ✅ Eksik Calcite 40°C ve Gypsum 34°C verileri eklendi
- ✅ CURRENT/README.md detaylı çalışma rehberi ile güncellendi
- ✅ Haftalık klasörleme sistemi tasarlandı

---

## 🔬 Sonuçlar

### Başarılar ✅
- Repo yapısı profesyonel hale getirildi
- Tüm v4 kodları tek yerde toplandı (CURRENT/)
- 12/12 durum için veri tamamlandı
- Eski ve yeni kodlar net şekilde ayrıldı
- Haftalık çalışma yol haritası oluşturuldu

### Öğrenilenler 📚
- v4 modeli v3'ten çok farklı: 2 fazlı sistem (gaz + sıvı)
- State variables: 10 → 14'e çıktı (Fe_pool, HCO3, S_tot, Lag eklendi)
- Parameters: 13 → 28'e çıktı (Henry sabitleri, kLa değerleri, vb.)
- Fiziksel model artık doğru: Henry yasası + pH speciation

### Zorluklar ⚠️
- Eski klasörleme sistemi karmaşıktı (9_12_2025_calisma, vb.)
- Calcite 40°C ve Gypsum 34°C eksikti (şimdi eklendi)
- Hangi kodun hangi versiyon olduğu belirsizdi

### Çözümler 💡
- ISO hafta + konu bazlı klasörleme sistemi tasarlandı
- Cumulative yaklaşım seçildi (her hafta öncekinin devamı)
- WEEKLY_STRUCTURE.md dokümantasyonu oluşturuldu
- Tüm eski dosyalar archive/weekly_work_old/ altına taşındı

---

## 📁 Üretilen Dosyalar

### Dokümantasyon
- `WEEKLY_STRUCTURE.md` - Haftalık klasörleme sistem dokümantasyonu
- `CURRENT/README.md` - Detaylı çalışma rehberi (günlük adımlar + kod örnekleri)
- `archive/weekly_work_old/README.md` - Eski çalışmaların açıklaması

### Veri Organizasyonu
- `CURRENT/code/v4_two_phase/` - 12/12 klasör tam set
- `CURRENT/data/muller_2024/` - 12/12 .txt dosyası tam set
- `shared/datasets/muller_2024/` - 12/12 .dat dosyası tam set

---

## 📌 Sonraki Hafta İçin (2026-W02)

- [ ] MATLAB'ı aç, Sandstone 25°C kodunu çalıştır
- [ ] Kodu satır satır oku, comment'lerle açıkla
- [ ] State variables listesi çıkar (notes/state_variables.md)
- [ ] Parameters tablosu oluştur (Excel)
- [ ] Henry yasası hesaplamalarını anla
- [ ] pH speciation formüllerini çöz

---

## 🤔 Sorular / Belirsizlikler

1. **14 state variable tam olarak neler?**
   - **Cevap**: CURRENT/README.md'de liste var - sonraki hafta detaylandır

2. **28 parametre nasıl kategorize edilir?**
   - **Cevap**: Kinetik sabitler, Monod sabitleri, kLa değerleri, vb. - sonraki hafta ayır

3. **Henry sabitleri sıcaklığa nasıl bağlı?**
   - **Cevap**: Van't Hoff denklemi - kodda implement edilmiş, anlamak gerek

---

## 📈 İlerleme İstatistikleri

- **Toplam çalışma saati**: ~4 saat (organizasyon + dokümantasyon)
- **Tamamlanan hedefler**: 6/6 organizasyon hedefi ✅
- **Git commit sayısı**: 3 (reorganization, updates, structure)
- **Oluşturulan dosya sayısı**: 50+ (CURRENT/, shared/, archive/)

---

**Git Commit Hash**: (Bu hafta sonunda eklenecek)
**Son Güncelleme**: 5 Ocak 2026, 21:00
