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
- Birim sorunu çözüldü ve sekiz-kayaç hattı kuruldu (2026-08-22 → 2026-08-27): kimya hocasının açıklaması (mg/L sadece şist GİRDİ su kimyasında; çıktılar PHREEQC tarafından mol/kgw'a çevrildiği için tutarlı) doğrulandı; `.phr` şablonları ve `phreeqc.dat`'tan PHREEQC'nin dönüşüm kuralı (mol/kgw = mg/L ÷ 1000 ÷ gfw) çıkarıldı, Ca ve Fe katsayıları çıktı verisi üzerinden %0.05 hata ile doğrulandı. Ayrı bir birim modülü (`conditional_model_v1/units.py`) yazıldı; veri setleri config'de `input_units` ile etiketleniyor. Skaler girdi sözlüğü 17→23 (K, Mn, Fe(2), Si, Al, gaz basıncı eklendi; şablonda olmayan alan 0), hedef seti 32→26 (sekiz sette ortak kolonlar). 55.122 koşumluk sekiz-kayaç önbelleği kuruldu.
- Sekiz-kayaç Colab eğitimi: [DOLDUR — eğitim koşulup sonuç alınınca].

### Dönem 7 — Eylül 2026: Dönüşüm doğrulaması ve kayaç-dışı genelleme (LORO)
- Kimya hocasının elle mol/kgw'a çevirdiği şist şablonlarıyla çapraz kontrol: 11 türden 10'u birebir; Si'de hoca element kütlesi (28.08), PHREEQC ise `units mg/L` altında SiO2 (60.08) kullanıyor. Eldeki koşular PHREEQC'nin kendi dönüşümüyle üretildiği için 60.08 korundu; kimya hocası aynı gün mailde doğruladı: laboratuvarda SiO2 ölçülür, değer SiO2 olarak girilir ve PHREEQC SiO2 olarak işler. Böylece 11 katsayının tamamı doğrulandı, önbellek değişmedi (2026-09-03).
- Leave-one-rock-out (LORO) hattı: bir kayaç eğitimden ve validasyondan tamamen çıkarılıp yalnız o kayaçta test ediliyor; kayaç başına ayrı model (8 model + 8-kayaç referans). Aynı model hem görülmeyen kayaçta hem görülen kayaçların test payında ölçülüyor; sonuçlar kat başına koşu klasörü, toplu özet CSV/JSON, SQLite tablosu ve grafiklerle kaydediliyor (2026-09-03, commit `8e7e82f34`).
- LORO Colab sonuçları: [DOLDUR — 8 kat koşulunca `loro_summary.csv`'den unseen vs reference RMSE tablosu].

## 2. Veri

- Kaynak: PHREEQC jeokimya simülasyonları (danışman tarafından üretilen koşum setleri).
- Kapsam: 4 eski kayaç (Calcite 9.875, Dolomite 9.882, Halite 9.425, Trona 9.899 koşum) + Sandstone 3.677 (= 42.758, pilot) + 3 şist seti (Quartzite 8.069, Mica 1.346, Mica-carbonate 2.949 = 12.364); sekiz kayaçta toplam 55.122 koşum.
- Her koşum: `{ANAHTAR} değer` satırlarından oluşan girdi dosyası + 301 zaman adımlı çıktı tablosu (eski setlerde 32, şistlerde 26–28 kolon; 26'sı ortak).
- Birimler: eski setler ve Sandstone çözelti kimyasını mol/kgw, şist şablonları mg/L yazıyor. PHREEQC girdiyi içeride mol/kgw'a çevirdiği için çıktılar tutarlı; girdiler modele girmeden önce aynı kurala göre (mg/L ÷ 1000 ÷ gram formül ağırlığı) mol/kgw'a çevriliyor. Si (SiO2 olarak, 60.08) ve Alkalinite (`as SO4-2`, 96.06) katsayıları şablondan türetildi; hocanın çevrilmiş dosyalarıyla çapraz kontrolde Alkalinite ve diğer 9 tür birebir uyuştu, Si için hocanın şablonunda 28.08 (element) görüldü; hoca mailde değerin ölçülmüş SiO2 olduğunu ve PHREEQC'nin SiO2 olarak okuduğunu doğruladı, 60.08 kesinleşti (2026-09-03).

## 3. Önişleme

- Girdi: 23 skaler koşul (sekiz setin birleşimi; şablonda olmayan alan 0) + sabit 17-mineral sözlüğünden 34 slot (mineral_MOLES/mineral_AREA; koşumda olmayan mineral 0) + normalize zaman kanalı → adım başına 58 özellik. Kayaç etiketi modele hiçbir zaman verilmiyor. (5-kayaç pilotunda 17 skaler / 52 özellikti.)
- Hedefler: sekiz sette ortak 26 kimyasal değişken (pilotta 32; şistlerde bulunmayan HCO3_mol, Na_tot, Mg_tot, Cl_tot, Ca_tot, S6_tot çıkarıldı); çarpık dağılımlı değişkenlere log1p, ardından tüm değişkenlere z-score normalizasyonu.
- Sıfır-varyanslı sütunlar (kullanılmayan mineral slotları) için bölme koruması (std→1).
- Veri bölme: kayaç-farkında, koşum seviyesinde %80/%10/%10 (seed 42) — aynı koşumun adımları asla iki kümeye bölünmüyor.
- Önişlenmiş veri tek seferlik, sürümlü npz önbelleğine yazılıyor (`five_rocks_pilot_v2`, `eight_rocks_v1`).

## 4. Model Mimarisi

- Koşullu LSTM: gizli boyut 128, 2 katman, dropout 0.
- Girdi (koşum, 301, 58) → çıktı (koşum, 301, 26) [pilotta 52 → 32]: model, deney koşullarından tüm kimyasal yörüngeyi üretiyor.
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
