# conditional_model_v3_8_rocks_wd — weight decay ile LORO deney alanı

Bu klasör bir **çalışma alanıdır**, kod kopyası değildir. Model kodu
`conditional_model_v1/` altında kalır (deney = config + commit ilkesi). Tasarım
notu ve önceden yazılmış başarı ölçütleri: `daily-sessions/2026-09-17_0112.md`.

## Soru

v2'de 8 kayaçlı LSTM gördüğü kayaçlarda iyi (normalize RMSE ≈ 0.07), hiç
görmediği kayaçta kötüydü (Calcite 1.36, Dolomite 3.90; 1.0 = "ortalamayı
söyleyen model"). Bu tur tek bir şeyi dener: **ağırlıkların büyümesini
engellemek** (weight decay) görülmemiş kayaç hatasını düşürüyor mu?

Weight decay, sıfırdan: her eğitim adımında hata itişinin yanına ikinci bir itiş
eklenir; her ağırlık büyüklüğüyle orantılı olarak sıfıra doğru çekilir (sıfıra
bağlı bir yay gibi). Ayar `training.weight_decay`; sayı yayın sertliği. Dropout,
öğrenme hızı ayarı, model boyutu bu turda **değişmez**, böylece fark tek nedene
bağlanır.

## Yöntem: yay sertliği eğrisi

Seçim yok, ölçüm var. Üç sertlikte 8 katlı LORO, kayaç başına görülmemiş hata
yan yana:

| Nokta | Config | weight_decay | Nereden |
|---|---|---|---|
| 0 | `configs/conditional_model_v1/loro_colab_eight_rocks.yaml` | 0.0 | v2, Drive'da hazır |
| 1 | `configs/conditional_model_v1/loro_colab_eight_rocks_wd_1e-4.yaml` | 0.0001 | bu defter |
| 2 | `configs/conditional_model_v1/loro_colab_eight_rocks_wd_1e-3.yaml` | 0.001 | bu defter |

Neden tarama değil: görülen kayaçların val'ına göre seçilen sertlik, görülmemiş
kayaç için en iyi sertlik olmayabilir (v2'de val'ı en iyi kat Dolomite, unseen'i
en kötüydü). Görülmemiş kayaca bakarak sertlik "seçmek" de yasak (test'e bakmak).
O yüzden iddia "en iyi sertlik şu" değil, "sertlik arttıkça unseen hata şöyle
değişiyor" olacak.

## Dosyalar

| Dosya | Ne işe yarar |
|---|---|
| `colab_eight_rocks_wd_loro.ipynb` | Colab defteri: repo klonu → Drive'daki `eight_rocks_v1` önbelleğini kopyala + `--deep` doğrula (**zip açma yok**) → her sertlik için `cli.loro` (8 kat) → her kat Drive'a → 8 kayaç × 3 sertlik tablosu + ölçüt kontrolü. |
| `README.md` | Bu dosya. |

Komut (defter bunu çağırır):

```
python -m conditional_model_v1.cli.loro \
  --config configs/conditional_model_v1/loro_colab_eight_rocks_wd_1e-4.yaml \
  --reference-run-dir <v2 8-kayaç referans koşu>
```

## Ölçütler (sonuç görülmeden yazıldı)

- **B açığı kapandı:** unseen normalize RMSE 8 kayacın en az 5'inde v2'ye göre
  düştü **ve** en az 4'ünde 1.0'ın altına indi **ve** seen normalize RMSE 0.10'u
  geçmedi.
- **Kapanmadı:** unseen çoğunlukla ≥ 1.0 kaldı → sorun ezber değil, kapsam;
  sonraki adım regularizasyon değil, veri/girdi tarafı.
- Aradaki her şey "kısmi"; sayılarla raporlanır, "düzeldi" denmez.

Defterin son hücresi bu sayımı otomatik yapar. Normalize sütunlar her katın
kendi cetveliyle (7 eğitim kayacından) hesaplanır; katlar arası yaklaşık
karşılaştırmadır. Tek sayı ilk bakış içindir; sonuç iddiası kayaç × çıktı ısı
haritası ve en kötü koşu eğrileriyle desteklenir.

## Kullanıcının yapacakları

1. Drive'da `chemical_thesis_data/processed/eight_rocks_v1/_SUCCESS` olduğunu
   kontrol et (v2 defteri kurdu). Yoksa önce v2 defterinin 6. hücresi.
2. Defterin 1. hücresinde `REFERENCE_RUN_DIR` (v2 8-kayaç referans koşu) ve
   `V2_LORO_DIR` (v2 LORO toplu klasörü) yollarını Drive'a göre doğrula.
3. GPU ile hücreleri sırayla çalıştır. Her sertlik ≈ 2 s 40 dk (T4); iki sertlik
   tek oturuma sığmazsa `WD_TO_RUN` ile böl, ikinci oturumda kalan.
4. Sonuç tablosunu ve ölçüt satırlarını `daily-sessions/` notuna ve
   `docs/gunluk/` satırına taşı.

## Nereye ne kaydedilir

v2 ile aynı düzen (`conditional_model_v2_8_rocks/README.md`). Koşu klasörleri
`<ts>_eight_rocks_loro_wd_1e-4_loro_<Kayaç>/`, toplu klasör
`<ts>_eight_rocks_loro_wd_1e-4_loro/` (1e-3 için aynı). Ortak `summary.csv` ve
`registry.sqlite` (`loro_folds` tablosunda `loro_name` sütunu sertliği ayırır).
