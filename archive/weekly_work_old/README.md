# Eski Haftalık Çalışmalar

⚠️ **Bu klasördeki dosyalar ARŞİV amaçlıdır - aktif çalışmalarda KULLANMAYIN!**

## İçerik

### 9_12_2025_calisma/
- **Tarih**: 9 Aralık 2025
- **İçerik**: İlk MATLAB baseline çalışması
- **Kod versiyonu**: v3 (tek fazlı - yanlış fizik)
- **Dosyalar**: rnn_transport_multiguild_uq_v3.m, trained_LSTM_multiguild.mat

### 16.12_2025_calisma/
- **Tarih**: 16 Aralık 2025
- **İçerik**: Keşif ve deney haftası
- **Kod versiyonu**: v3 (tek fazlı - yanlış fizik)
- **Alt klasörler**: 16_12_2025_icin_taskler/ (eski task'lar)

### 23_12_2025_calisma/
- **Tarih**: 23 Aralık 2025
- **İçerik**: PINN finalizasyonu
- **Kod versiyonu**: v3 (tek fazlı - yanlış fizik)
- **Alt klasörler**: İlk_deneme/, İkinci_deneme - Kopya/

---

## ⚠️ Neden Arşivlendi?

Bu çalışmalar **v3 tek fazlı model** ile yapıldı. Profesör **3 Ocak 2026**'da **v4 iki fazlı model** gönderdi:

| Özellik | v3 (Eski) | v4 (Yeni) |
|---------|-----------|-----------|
| Faz sayısı | 1 (sadece sıvı) | 2 (gaz + sıvı) |
| State variables | 13 | 14 |
| Parameters | 13 | 28 |
| Henry yasası | ❌ Yok | ✅ Var |
| Fizik | ❌ Yanlış | ✅ Doğru |

## 🎯 Şu An Kullanılması Gereken

❌ **KULLANMA** → Bu klasör (`archive/weekly_work_old/`)
✅ **KULLAN** → [`CURRENT/code/v4_two_phase/`](../../CURRENT/)

---

**Arşivlenme Tarihi**: 5 Ocak 2026
