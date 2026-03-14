# Kimya Doç. Açıklamaları (Özet)

## Ekip ve Veri Akışı
```
Kimya Doç. → MATLAB fit + PHREEQC model
Kimya asistanları → PHREEQC+Python, sensitivite analizi, sentetik veri üretimi
Hasan (CS MSc) → Sentetik verileri al, LSTM surrogate model eğit
CS Doç. → ML danışmanlığı
```

## İki Deney Seti
| | Muller 2024 | Mura 2024 |
|---|---|---|
| Basınç | ~2 bar | ~60 bar |
| Fizik | Henry yasası | Peng-Robinson EOS |
| Süre | ~19 gün | ~100 gün |
| Kayaç | 4 tür × 3 sıcaklık | Sandstone, 36°C |

## PHREEQC Notları
- USGS jeokimyasal modelleme programı
- MATLAB'dan farklı: fizik gömülü, müdahale edilemez
- Manuel 2 hafta uğraşılarak yazılmış

## Mura 2024 Belirsizlikleri
- 60 bar basıncın nasıl sağlandığı belirsiz
- Headspace volume belli değil
- Makaledeki bazı değerler tutarsız
