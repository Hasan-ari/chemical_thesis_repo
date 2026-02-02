# Kimya Hocası (Selçuk Hoca) Açıklamaları

Bu dosya kimya doçentinin (tez danışmanı) mail yoluyla paylaştığı teknik açıklamaları içerir.

---

## 1. Muller 2024 - Düşük Basınç Deneyleri

### Genel Yaklaşım
- **Sistem:** Çift faz (two-phase gas-liquid)
- **Basınç:** ~2 bar (düşük basınç)
- **Fizik:** Henry's solubility yasası (ideal gaz)
- Kimyasal türler (H2S, H2 vs.) hem gaz hem sıvı halde bulunuyor

### Kayaç Özellikleri
- **4 kayaç türü:** Basalt, Calcite, Gypsum, Sandstone
- **3 sıcaklık:** 25°C, 34°C, 40°C
- Tüm kayaçlar tek algoritma ile hesaplanıyor
- Sadece parametre değerleri değiştiriliyor

### Gypsum (Jips) Kayacı Özel Durumu
```
Kimyasal formül: CaSO4·2H2O
```
- Çözündüğünde ortama çok fazla **SO4 (sülfat)** salıyor
- Bu durum **hydrogen sulfate reduction**'ı daha fazla tetikliyor
- Demiroksitle temasında çok erken **H2S gaz salınımı** oluşuyor
- Her kayaç ve sıcaklık için parametre ayarlaması gerekiyor

### Kritik Hassas Parametreler
Fitleri en çok etkileyen parametreler:
```matlab
Hcp_H2_base    % Henry sabiti - H2
Hcp_CO2_base   % Henry sabiti - CO2
Hcp_H2S_base   % Henry sabiti - H2S
S_tot0         % Başlangıç toplam sülfür
env.pKa_H2S    % H2S pKa değeri
env.SO4_sat_gyp % Jips SO4 doygunluğu
```

### Teknik Notlar
- PHREEQC gibi sofistike programlar bu derece iyi fit edemeyebilirdi
- Matematiksel background'a doğrudan müdahale edilebiliyor
- CPU kullanılarak çözülüyor (GPU ile hızlandırılabilir)

---

## 2. Mura 2024 - Yüksek Basınç Deneyleri

### Deney Koşulları
- **Basınç:** ~60 bar (yüksek basınç)
- **Sıcaklık:** 36°C
- **Süre:** ~100 gün
- **Kayaç:** Sandstone (kumtaşı)

### Sandstone Mineral Kompozisyonu
```
Quartz:     %90-95 (ana mineral)
Diğerleri:  %5-10  (kil, demiroksit, vb.)
```
- Özellikle %5-10 kısımdaki mineraller reaksiyona daha meyilli

### Fiziksel Yaklaşım Farkı

#### Düşük Basınç (Muller 2024):
```
İdeal gaz → Henry Yasası
```

#### Yüksek Basınç (Mura 2024):
```
Non-ideal (real) gaz → Peng-Robinson EOS + Henry-Sechenov
```

| Faz | Model |
|-----|-------|
| Gaz | Peng-Robinson Equation of State (fugacity) |
| Sıvı | Henry + Sechenov salinity modeli (ϕ–γ çerçevesi) |

### Cushion Gas (Yastık Gazı) Konsepti
- Gerçek yeraltı depolama yöntemi
- Hidrojen basılırken diğer gazlarla (CH4, CO2) karıştırılıyor
- Hidrojen molekülü çok ince (2 elektron) → difüzyonla kaçabiliyor
- Cushion gas basınç düşüşünü yavaşlatıyor

### Deney Protokolü Farkları

| Özellik | Muller 2024 | Mura 2024 |
|---------|-------------|-----------|
| Basınç | ~2 bar | ~60 bar |
| H2 basma | Saf hidrojen | Gaz karışımı |
| Süre | ~19 gün | ~100 gün |
| H2 kaybı | Hızlı | Yavaş |
| Kayaç bilgisi | Mineral bilgisi yok | Kompozisyon verilmiş |

### Kod Geliştirme
- ChatGPT destekli hazırlandı (Peng-Robinson adaptasyonu için)
- Tüm adımlar kod içinde yorumlarla açıklandı
- Summary dosyası oluşturuldu

---

## 3. PHREEQC - Jeokimyasal Modelleme

### PHREEQC Nedir?
- USGS tarafından geliştirilen jeokimyasal modelleme programı
- Notepad++ üzerinde çalışıyor
- Phreeqc'nin anlayabileceği bloklar yazılıyor

### MATLAB vs PHREEQC Karşılaştırması

| Özellik | MATLAB | PHREEQC |
|---------|--------|---------|
| Fizik/ODE | Görünür, müdahale edilebilir | Gömülü, müdahale edilemez |
| Fit yöntemi | Deneysel verilere otomatik fit | Manuel simülasyon |
| Çıktı | Monod-kinetic parametreleri | Trend eşleştirme |

### PHREEQC'de Yapılanlar
- Mura 2024 makalesini baz alarak kod yazıldı
- Redox reaksiyon tepkimeleri simüle ediliyor
- Mineral kinetikleri dikkate alınıyor
- Hidrojen, formate, calcium deneylerle yakın tutturuldu

### Mura 2024 Makalesindeki Belirsizlikler
- 60 bar basıncın nasıl sağlandığı belirsiz (N2? Toluene?)
- Headspace volume belli değil
- H2 injection 0.105 mmol ama partial pressure yok
- "99 CH4+CO2" değeri ile Figure 1'deki CO2=40 mmol tutarsız

### Gelecek Çalışmalar
1. Phreeqc + Python entegrasyonu
2. Parametre sensitivite analizi
3. Mineral yüzey alanı optimizasyonu (calcite, barite kritik)
4. Farklı mineral yüzdeleri ve initial condition'larla sentetik veri üretimi

---

## 4. Ekip Yapısı ve İş Bölümü

### Ana Ekip
| Kişi | Rol |
|------|-----|
| **Selçuk Hoca** | Kimya Doçenti, MATLAB kodları, PHREEQC |
| **Berkay, Zelal, Mehmet** | Phreeqc optimizasyonu, Python entegrasyonu |
| **Hasan** | CS Yüksek Lisans, PyTorch LSTM |
| **CS Doçenti** | Tez Danışmanı #2 |

### Veri Akışı
```
Selçuk Hoca:
  - MATLAB ile Muller/Mura verilerini fit et
  - Monod-kinetic parametreleri elde et
  - PHREEQC ile simülasyon yap
        ↓
Berkay/Zelal/Mehmet:
  - Phreeqc + Python entegrasyonu
  - Parametre sensitivitesi
  - Bolca sentetik veri üret
        ↓
Hasan:
  - Sentetik verileri al
  - PyTorch LSTM eğit
  - Surrogate model oluştur
```

---

## 5. Önemli Fiziksel Kavramlar

### Henry Yasası (Düşük Basınç)
```
C_aq = H × P_gas
```
- C_aq: Sulu fazdaki konsantrasyon
- H: Henry sabiti
- P_gas: Gaz kısmi basıncı

### Peng-Robinson EOS (Yüksek Basınç)
```
P = RT/(V-b) - a(T)/(V² + 2bV - b²)
```
- Non-ideal gaz davranışı için
- Fugacity hesabı yapılır
- 60 bar gibi yüksek basınçlarda gerekli

### Sechenov Etkisi
```
log(S/S0) = -K_s × I
```
- S: Tuzlu sudaki çözünürlük
- S0: Saf sudaki çözünürlük
- K_s: Sechenov sabiti
- I: İyonik güç

### Monod Kinetiği
```
r = r_max × S / (K_s + S)
```
- Mikrobiyal büyüme/reaksiyon hızı
- r_max: Maksimum hız
- K_s: Yarı doygunluk sabiti
- S: Substrat konsantrasyonu

---

## 6. Sonraki Adımlar (Kimya Hocasından)

### Kısa Vadeli
1. ✅ Muller 2024 fitleri tamamlandı (12 koşul)
2. ✅ Mura 2024 yüksek basınç fiti tamamlandı
3. 🔄 PHREEQC kodu optimize ediliyor
4. 🔄 Berkay/Zelal/Mehmet ile sensitivite analizi

### Orta Vadeli
1. ⏳ Phreeqc + Python entegrasyonu
2. ⏳ Farklı mineral kompozisyonlarıyla simülasyon
3. ⏳ Bolca sentetik veri üretimi

### Uzun Vadeli
1. ⏳ Tüm verilerin LSTM training için hazırlanması
2. ⏳ Neural network ile kimya hocasının kendi denemeleri

---

## 7. Kritik Notlar

### Fit Kalitesi
- Tüm kimyasal türler küçük RMSE hata paylarıyla tutturuldu
- En etkin fiziksel yaklaşım ve matematiksel çözüm elde edildi

### GPU Kullanımı
- MATLAB'da CPU kullanılıyor (zaman alıyor)
- GPU ile çok daha hızlı çözüm mümkün
- Python için GPU modifikasyonu önerildi

### PHREEQC Limitasyonları
- ChatGPT/Copilot ile PHREEQC kodu yazdırmak başarısız oldu
- "Tam bir fiyasko, tutarlılığı yok, yanlış yönlendiriyor"
- Manuel olarak 2 hafta uğraşıldı

### Veri Güvenilirliği
- Muller 2024: Kayaç mineral bilgisi verilmemiş
- Mura 2024: Bazı protokol detayları belirsiz/tutarsız
- Fit sonuçları deney koşullarına göre yorumlanmalı
