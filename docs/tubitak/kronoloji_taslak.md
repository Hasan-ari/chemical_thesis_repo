# TÜBİTAK Raporu — Kronoloji ve Teknik İçerik Taslağı

> Bu taslak git geçmişi, yerel ROADMAP kayıtları ve `results/` klasörlerinden derlendi
> (2026-08-21). `[DOLDUR]` işaretli yerler senin kaleminden çıkmalı; sayısal sonuçlar
> ilgili `results/.../metrics.json` dosyalarından alınabilir.

## 1. Kronolojik Çalışma Özeti

### Dönem 1 — Kasım 2025 – Ocak 2026: Başlangıç ve ilk LSTM denemeleri
- Proje deposu kuruldu, PHREEQC simülasyon çıktıları incelendi, ilk karşılaştırmalar yapıldı (2025-11-15).
- İlk LSTM eğitim denemeleri (2026-01-06) ve delta-learning / recursive-forecasting varyantları (2026-01-12): modelin mutlak değer yerine zaman adımları arası farkı öğrenmesi denendi, iki sürüm geliştirildi.
- Veri downsample edilerek eğitim süresi/bellek dengesi araştırıldı (2026-01-25).

### Dönem 2 — Şubat – Mart 2026: Yeniden yapılanma ve sistematik EDA
- Haftalık deney düzeni oturtuldu; eski kodlar ayıklandı (2026-02-02).
- Yeni PHREEQC v23 veri seti alındı; depo modüler dizin yapısına geçirildi (2026-03-14).
- PHREEQC v23 keşifsel veri analizi (EDA): kolon özetleri, dağılımlar, yörünge grafikleri (2026-03-16).
- Modüler LSTM eğitim hattı (pipeline) ve ilk deney sonuçları (2026-03-28).

### Dönem 3 — Nisan – Mayıs 2026: Literatüre hizalı değerlendirme
- Referans makaleye (Hidden RTNN) hizalı kapsamlı rollout değerlendirme altyapısı; Figür 1–12 üretim hattı (2026-04-20 → 2026-05-03).
- Farklı girdi pencereleriyle deney serisi: seq{3, 5, 10, 20}, hidden=128; değerlendirme artefaktları arşivlendi (2026-05-03).
- R² hesabında sıfıra bölme koruması gibi sayısal düzeltmeler; ruff + pre-commit ile kod kalite altyapısı (2026-05-03).
- LaTeX tez raporu iskeleti ve Veri bölümü taslağı (2026-05-03/04).

### Dönem 4 — Haziran 2026: Koşullu (conditional) LSTM'e geçiş
- Colab'a taşınabilir koşullu LSTM hattı: model artık tek bir yörüngeyi değil, deney koşullarından (sıcaklık, gözeneklilik, çözelti bileşimi, mineral miktarı/yüzeyi…) yörüngenin tamamını üretiyor (2026-06-29).
- İlk deney sonuçları ve öğrenme hızı (learning rate) taraması: 6 koşum, metrikleri `results/2026-06-29_lr_sweep_calcite/` altında (2026-06-29).

### Dönem 5 — Temmuz 2026: Dört-kayaç eğitimi
- Dört kayaç (Calcite, Dolomite, Halite, Trona) tek modelde birleştirildi; kayaç-farkında (rock-aware) veri bölme ile Colab eğitim hattı (2026-07-13).
- Kayaç-bazlı geçiş (transition) görselleştirmeleri ve özet grafikler (2026-07-13).
- Sonuçlar: [DOLDUR — dört-kayaç koşumunun test RMSE/R² özetini `results/` ve Drive koşum klasöründen al].

### Dönem 6 — Ağustos 2026: Sekiz kayaca genişleme hazırlığı ve 5-kayaç pilotu
- Depo disiplini: ham veri git'ten çıkarıldı (yedek Drive'da), deney = config + commit ilkesi, koşum artefaktları LFS/`results/` altında (2026-08-09, 2026-08-20).
- Sekiz kayaçlık veri atlası: 55.122 koşumun tamamı taranarak eski/yeni veri setlerinin alan varlığı, gerçek dağılımları ve üretim kodundaki hedef aralıkları karşılaştırıldı; birim uyumsuzluğu (mol/kgw ↔ mg/L, ~1000×) tespit edildi (2026-08-20).
- 5-kayaç pilotu: sabit 17-mineral sözlüğü girdi temsili tasarlandı ve uygulandı; 42.758 koşum yeni sözleşmeden hatasız geçti, 86 birim testi yeşil (2026-08-20, commit `0ffcd97f1`).
- Danışman onayı: "olmayan mineraller 0 olsun, doğrudan eğitime girilebilir" — tasarım onaylandı (2026-08-21).
- Beş-kayaç Colab eğitimi: [DOLDUR — eğitim koşulup sonuç alınınca].

## 2. Veri

- Kaynak: PHREEQC jeokimya simülasyonları (danışman tarafından üretilen koşum setleri).
- Kapsam: 4 eski kayaç (Calcite 9.875, Dolomite 9.882, Halite 9.425, Trona 9.899 koşum) + Sandstone 3.677; toplam 42.758 koşum pilotta. 3 şist seti (12.364 koşum) birim kararı sonrası eklenecek.
- Her koşum: `{ANAHTAR} değer` satırlarından oluşan girdi dosyası + 301 zaman adımlı, 32 hedef kolonlu çıktı tablosu.

## 3. Önişleme

- Girdi: 17 skaler koşul + sabit 17-mineral sözlüğünden 34 slot (mineral_MOLES/mineral_AREA; koşumda olmayan mineral 0) + normalize zaman kanalı → adım başına 52 özellik. Kayaç etiketi modele hiçbir zaman verilmiyor.
- Hedefler: 32 kimyasal değişken; çarpık dağılımlı değişkenlere log1p, ardından tüm değişkenlere z-score normalizasyonu.
- Sıfır-varyanslı sütunlar (kullanılmayan mineral slotları) için bölme koruması (std→1).
- Veri bölme: kayaç-farkında, koşum seviyesinde %80/%10/%10 (seed 42) — aynı koşumun adımları asla iki kümeye bölünmüyor.
- Önişlenmiş veri tek seferlik, sürümlü npz önbelleğine yazılıyor (`five_rocks_pilot_v1`).

## 4. Model Mimarisi

- Koşullu LSTM: gizli boyut 128, 2 katman, dropout 0.
- Girdi (koşum, 301, 52) → çıktı (koşum, 301, 32): model, deney koşullarından tüm kimyasal yörüngeyi üretiyor.
- [DOLDUR — istersen buraya bir mimari şeması; benden isteyebilirsin.]

## 5. Eğitim

- Colab GPU üzerinde; Adam, lr=0.001, batch 64, 100 epoch, gradient clipping 1.0, seed 42.
- Deney takibi: her koşum `config.yaml + git_commit.txt + metrics.json` üçlüsüyle `results/YYYY-AA-GG_<ad>/` altında saklanıyor.

## 6. Sonuçlar ve Değerlendirme

- Metrikler: özellik-bazlı ve kayaç-bazlı RMSE/R²; yörünge üst üste bindirme grafikleri.
- [DOLDUR — dört-kayaç ve beş-kayaç sayısal sonuç tabloları.]

## 7. TÜBİTAK biçimine uyarlama notları

- [DOLDUR — program türü (ör. 2209-A), rapor dönemi, proje adı/numarası, özet ve amaç bölümleri.]
- Bu taslaktaki dönem başlıkları rapor takvim çizelgesine (iş-zaman) birebir oturtulabilir.
