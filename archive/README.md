# Archive - Eski Çalışmalar

⚠️ **Bu klasördeki dosyalar ARŞİV amaçlıdır - aktif çalışmalarda KULLANMAYIN!**

## İçerik

### reactions_old/
Eski reactions/ klasörünün tam kopyası. İçerir:

- **Matlab codes/**: Eski MATLAB kodları (v1, v2, v3?)
- **Pytorch-ipynb/**: Eski PyTorch denemeleri
- **dataset_for_training_different_rocks_at_25C-34C-40C/**: Eski dataset organizasyonu
- **Reactive transport description/**: Eski açıklama dokümanları

## ⚠️ Neden Arşivlendi?

**3 Ocak 2026 akşam mailinden önce:** v1, v2, v3 gibi eski tek fazlı modeller (yanlış fizik)

**3 Ocak 2026 akşam mailinden sonra:** Profesör **v4 iki fazlı modeli** gönderdi:
- ✅ 14 state variables (v3'te 13'tü)
- ✅ 28 parameters (v3'te 13'tü)
- ✅ İki fazlı sistem (gaz + sıvı)
- ✅ Doğru Henry yasası
- ✅ pH türlenmesi
- ✅ 12 başarılı fit (Sandstone, Basalt, Calcite, Gypsum × 25°C, 34°C, 40°C)

## 🎯 Hangi Kodları Kullanmalısın?

❌ **KULLANMA** → `archive/reactions_old/`  
✅ **KULLAN** → [`CURRENT/code/v4_two_phase/`](../CURRENT/)

## Referans İçin Saklanma Nedenleri

1. **Tarihsel kayıt**: Ne yapıldığını görmek için
2. **Karşılaştırma**: v4 ile eski versiyonları karşılaştırmak için
3. **Veri kaybı önleme**: Belki yararlı bir şey vardır diye

## Not

Bu klasördeki kodlar:
- Yanlış fiziksel modeller içerebilir
- Eksik parametreler olabilir
- Güncellenmeyecek
- Sadece referans amaçlı

---

**Arşivlenme Tarihi**: 5 Ocak 2026  
**Neden**: Profesörün v4 two-phase model ile güncelleme yapması
