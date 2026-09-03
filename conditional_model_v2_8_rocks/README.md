# conditional_model_v2_8_rocks — sekiz kayaç deney alanı

Bu klasör bir **çalışma alanıdır**, ayrı bir kod kopyası değildir. Model kodu tek
paket olarak `conditional_model_v1/` altında kalır (deney = config + commit
ilkesi; kod kopyası yok). Burada:

| Dosya | Ne işe yarar |
|---|---|
| `colab_eight_rocks_loro.ipynb` | Colab defteri: 8 zip'i açar, `eight_rocks_v1` önbelleğini kurar/alır, **referans** 8-kayaç modelini eğitir, sonra **LORO** (8 kat) çalıştırır, her şeyi Drive'a kopyalar. Yollar defterin 1. hücresinde. |
| `last_three_rock_molkgw_conversion/` | Kimya hocasının elle mol/kgw'a çevirdiği 3 şist `.phr` şablonu + birer tek koşu. **Veri seti değil, cevap anahtarı.** Git'e girmez (`.gitignore`). |

## Hangi config, hangi komut

| İş | Config | Komut (Colab defteri bunları çağırır) |
|---|---|---|
| Referans (8 kayaç, tek model) | `configs/conditional_model_v1/full_colab_eight_rocks.yaml` | `python -m conditional_model_v1.cli.train --config ...` |
| LORO (kayaç başına 1 model) | `configs/conditional_model_v1/loro_colab_eight_rocks.yaml` | `python -m conditional_model_v1.cli.loro --config ... --reference-run-dir <referans koşu>` |
| Yerel duman testi | `configs/conditional_model_v1/smoke_loro_local.yaml` | `env312/bin/python -m conditional_model_v1.cli.loro --config ...` |

İki config aynı önbelleği (`eight_rocks_v1`, 55.122 koşu), aynı modeli
(hidden 128, 2 katman) ve aynı eğitimi (100 epoch, batch 64, lr 1e-3, seed 42)
kullanır; tek fark `split.strategy`.

## LORO ne demek

Bir kayacı eğitim **ve** validasyondan tamamen çıkar, sadece o kayaçta test et.
Bunun için her kayaca ayrı model gerekir: 8 kayaç → 8 model (+ referans).
Her katta oranlar (%80/%10/%10) kalan 7 kayaca uygulanır; böylece aynı model
hem görmediği kayaçta (`test_unseen`) hem de gördüğü kayaçların test payında
(`test_seen`) ölçülür.

## Nereye ne kaydedilir

Colab'da `/content/runs/`, sonra Drive'da `MyDrive/chemical_thesis_repo/runs/`:

- **Her koşu klasörü** (`<ts>_eight_rocks_condition_lstm_v1/` ve
  `<ts>_eight_rocks_loro_v1_loro_<Kayaç>/`): `history.csv` (epoch başına
  train/val loss + lr), `metrics.json` (global + kayaç bazlı RMSE/MAE, LORO
  katlarında `test_unseen`/`test_seen`), `feature_metrics.csv`,
  `rock_feature_metrics.csv`, `eval_predictions.npz`, `checkpoints/best.pt`,
  `plots/loss_curve.png`, `plots/rock_overviews/<Kayaç>_{best,worst,mean}_overview.png`.
- **Toplu LORO klasörü** (`<ts>_eight_rocks_loro_v1_loro/`): `loro_summary.csv`
  (kat başına 1 satır; her kattan sonra yeniden yazılır), `loro_rock_feature_metrics.csv`
  (kayaç × 26 hedef), `loro_metrics.json`, `loro_config.json`, `plots/` (çubuk
  grafikler orijinal + normalize, kayaç × hedef ısı haritası, kopyalanan overview'lar).
- **Ortak kayıt defteri**: `summary.csv` (satır ekler) ve `registry.sqlite`
  (`runs`, `rock_feature_metrics`, `loro_folds` tabloları). Drive'daki kopya
  her oturumda devam ettirilir, geçmiş silinmez.

## Kullanıcının yapacakları

1. Drive `chemical_thesis_data/` altına 8 zip: `Calcite_wat_sat_data_3.zip`,
   `Dolomite_wat_sat_data_2.zip`, `Halite_wat_sat_data_2.zip`,
   `Trona_par_sat_data_3.zip`, `Sandstone_data_2.zip`, `Quarzite_schist_low_t.zip`,
   `Mica_schist_low_t.zip`, `Mica_carbonate_schist_low_t.zip`.
2. Defteri Colab'da GPU ile aç, 1. hücredeki yolları kontrol et, hücreleri sırayla çalıştır.
3. Süre yetmezse `HELD_OUT_ROCKS` ile alt küme; ikinci oturumda kalanlar.
   Bitmiş referans koşu için `REFERENCE_RUN_DIR` ver, yeniden eğitilmez.

## Açık nokta: Si birimi

Hocanın çevrilmiş şablonu Si için 28.08 (element) kullanmış; PHREEQC `units mg/L`
altında Si'yi `phreeqc.dat`'taki formüle göre SiO2 (60.08) olarak okur ve eldeki
şist koşuları böyle üretildi. Bu yüzden `conditional_model_v1/units.py` 60.08'de
kalır; hoca teyit ederse yalnız o sayı ve önbellek değişir. Diğer 10 tür birebir uyuştu.

## Kutu grafikleri (box plot)

Her eğitim koşusu, test bölmesindeki **her PHREEQC koşusu için ayrı bir RMSE** hesaplar
(`run_rmse.csv`, sqlite `run_rmse` tablosu). Bu sayılardan `plots/boxplots/` altında
üç PNG üretilir:

| Dosya | Ne gösterir |
|---|---|
| `rmse_boxplot_by_rock.png` | Kayaç başına koşu-RMSE dağılımı (26 çıktı birlikte) |
| `rmse_boxplot_by_feature.png` | Çıktı (26 değişken) başına dağılım, tüm kayaçlar |
| `rmse_boxplot_by_rock_and_feature.png` | Kayaç × çıktı; her kayaç için ayrı panel |

LORO toplu klasöründe ek olarak `plots/loro_unseen_rmse_boxplot.png` bulunur: her
dışarıda bırakılan kayaç için, o kayacı hiç görmemiş modelin koşu-RMSE kutusu (mavi)
ile 8 kayaçlı referans modelin aynı kayaçtaki kutusu (turuncu) yan yana. Sayılar
`loro_run_rmse.csv` içinde. Not: normalize RMSE her modelin kendi ölçekleyicisiyle
hesaplanır; kutular yaklaşık karşılaştırma içindir.
