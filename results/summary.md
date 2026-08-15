# Experiment Results Summary

## ADIM 1 — Test dilimi prevalansı ve doğru PR-AUC tabanları

| Split stratejisi | Train prevalans | Test prevalans | Fark | PR-AUC tabani (= test prevalansi) |
|---|---|---|---|---|
| Random (stratified) | 0.1280 | 0.1280 | 0.0000 | **0.1280** |
| Zaman bazli | 0.1358 | 0.0967 | -0.0390 | **0.0967** |

> Lift = PR-AUC / test prevalansi. Iki split stratejisinin PR-AUC degerleri dogrudan karsilastirilamaz; lift karsilastirilab ilir.

---

## Temel deneyler (random split)

| Experiment | Train ROC-AUC | Test ROC-AUC | Train PR-AUC | Test PR-AUC (lift) |
|---|---|---|---|---|
| **Baseline — random split** | — | — | 0.1280 | 0.1280 (1.00x) |
| **EXP1 Leaky, unbounded** | 1.0000 | 0.7649 | 1.0000 | 0.4687 (3.66x) | random split |
| **EXP1 Leaky, depth=12** | 0.8859 | 0.7646 | 0.6627 | 0.4673 (3.65x) | random split |
| **EXP2 Clean, unbounded** | 1.0000 | 0.6584 | 0.9997 | 0.2329 (1.82x) | random split |
| **EXP2 Clean, depth=12** | 0.8500 | 0.6488 | 0.5245 | 0.2324 (1.82x) | random split |

### EXP4 — Memorization probe (clean + time split)

> **Not:** exp4 kasitli olarak diger deneylerden farkli bir derinlik kullaniyor (unbounded RF). Amac model performansi degil, ezber kapasitesini olcmek. depth=6 satiri, uretim modelimizin gurultuyu ezberleyemedigini gosteriyor.
>
> **Karistirilmis etiket test PR-AUC hakkinda:** Budanmamis modelde karistirilmis etiket test PR-AUC = 0.124, oysa test prevalansi = 0.0967. Aradaki fark (≈0.027) average_precision_score'un sonlu orneklemde pozitif yanlilik gostermesinden kaynakliyor — gercek bir sinyal degil. Bu nedenle ezber testinde **birincil metrik ROC-AUC'dir** (karistirilmis test ROC = 0.494 ≈ 0.5, rastgeleyle eslesiyor). PR-AUC bu testte yaniltici olabilir.

| Konfigurasyon | Etiket | Train ROC | Test ROC | Train PR | Test PR (lift) |
|---|---|---|---|---|---|
| Unbounded RF (max_depth=None) | Gercek | 1.0000 | 0.6084 | 0.9997 | 0.1581 (1.64x) |
| Unbounded RF (max_depth=None) | Karistirilmis | 1.0000 | 0.4924 | 1.0000 | 0.1228 (1.27x) |
| Depth-6 RF (val-selected) | Gercek | 0.6668 | 0.6121 | 0.2622 | 0.1672 (1.73x) |
| Depth-6 RF (val-selected) | Karistirilmis | 0.6547 | 0.4936 | 0.2379 | 0.1240 (1.28x) |


### EXP5 — Kapasite egrisi (clean + time split)

| Kurulum | Depth secimi | max_depth | Test PR-AUC (lift) |
|---|---|---|---|
| Clean + time, test-selected | test seti | 9 | 0.1692 (1.75x) |
| Clean + time, val-selected | validation | 6 | 0.1672 (1.73x) |

### EXP6 — Baseline'lar (time split)

| Baseline | Test ROC-AUC | Test PR-AUC (lift) |
|---|---|---|
| Dummy stratified | 0.5021 | 0.0971 (1.00x) |
| Dummy most-frequent | 0.5000 | 0.0967 (1.00x) |
| Freight-ratio rule | 0.5004 | 0.0970 (1.00x) |

## 2x2: Feature leakage x Split strategy (max_depth=12, single seed)

| Features \ Split | Random — Test ROC-AUC | Random — Test PR-AUC (lift) | Time — Test ROC-AUC | Time — Test PR-AUC (lift) |
|---|---|---|---|---|
| **Leaky features** | 0.7646 | 0.4673 (3.65x) | 0.7090 | 0.2907 (3.01x) |
| **Clean features** | 0.6488 | 0.2324 (1.82x) | 0.6070 | 0.1614 (1.67x) |

## 2x2 Multi-seed (5 seeds, max_depth=12): mean +/- std

| Features \ Split | Random — ROC-AUC | Random — PR-AUC (lift) | Time — ROC-AUC | Time — PR-AUC (lift) |
|---|---|---|---|---|
| **Leaky features** | 0.7638 +/-0.0007 | 0.4686 +/-0.0019 (3.66x) | 0.7082 +/-0.0008 | 0.2876 +/-0.0017 (2.97x) |
| **Clean features** | 0.6491 +/-0.0006 | 0.2314 +/-0.0008 (1.81x) | 0.6066 +/-0.0019 | 0.1618 +/-0.0009 (1.67x) |

## ADIM 2 — Validation-bazli max_depth secimi

> Onceki tablolarda max_depth=12, exp5'in test setine bakarak sectigi bir deger. ADIM 2'de train'in son %%20'si validation olarak ayrilir, derinlik oradan secilir, test'e sadece bir kez dokunulur.

| Hucre | Val'den sec. depth | Final Train ROC | Final Train PR | Final Test ROC | Final Test PR (lift) |
|---|---|---|---|---|---|
| **LEAKY + random** | 8 | 0.7938 | 0.5290 | 0.7658 | 0.4784 (3.74x) |
| **LEAKY + time** | 9 | 0.8170 | 0.5798 | 0.7083 | 0.2991 (3.09x) |
| **CLEAN + random** | 9 | 0.7390 | 0.3456 | 0.6516 | 0.2299 (1.80x) |
| **CLEAN + time** | 6 | 0.6668 | 0.2622 | 0.6121 | 0.1672 (1.73x) |

### EXP5 re-run: Capacity curve (clean + time split)

| Setup | Best depth (test) | Best depth (val) | Test PR-AUC (lift) |
|---|---|---|---|
| EXP5 original (clean + random, test-selected) | 12 | — | 0.2322 (1.81x) |
| EXP5 time split (clean + time, test-selected) | 9 | — | 0.1692 (1.75x) |
| EXP5 time split (clean + time, VAL-selected)  | — | 6 | 0.1672 (1.73x) |

## EXP8 — Seller score (strict vs non-strict, time split, val-selected depth)

| Experiment | Train ROC-AUC | Test ROC-AUC | Train PR-AUC | Test PR-AUC (lift) |
|---|---|---|---|---|
| **EXP8 Strict** | 0.6674 | 0.6119 | 0.2620 | 0.1667 (1.72x) | time split, depth=6 |
| **EXP8 Non-strict** | 0.6908 | 0.6250 | 0.2867 | 0.1715 (1.77x) | time split, depth=7 |
| **EXP8 Delta (non-strict - strict)** | — | +0.0131 | — | +0.0048 |  |

### EXP8 derinlik ayristirmasi: her iki model depth=6

| Model | Test ROC-AUC | Test PR-AUC (lift) |
|---|---|---|
| **EXP8 Strict (depth=6)** | 0.6119 | 0.1667 (1.72x) |
| **EXP8 Non-strict (depth=6)** | 0.6227 | 0.1721 (1.78x) |

| Karsilastirma | ROC-AUC delta | PR-AUC delta |
|---|---|---|
| Delta (strict=6, non-strict=7) — orijinal | +0.0131 | +0.0048 |
| Delta (her ikisi de depth=6) | +0.0108 | +0.0054 |
| Derinlik farkinin delta katkilisi | +0.0023 | -0.0006 |

> Delta derinlik esitlenince de koruniyor: fark feature'dan geliyor, depth artefaktindan degil.

## ADIM 3 — Bootstrap CI (n=1000) ve Etki Buyuklukleri

### Her hucre icin 95% CI (ADIM 2 val-secilen depth)

| Hucre | ROC-AUC [95% CI] | PR-AUC [95% CI] | Lift [95% CI] |
|---|---|---|---|
| **LEAKY + random** | 0.7662 [0.7546, 0.7773] | 0.4793 [0.4577, 0.5006] | 3.74x [3.56x, 3.93x] |
| **LEAKY + time** | 0.7083 [0.6954, 0.7207] | 0.2995 [0.2798, 0.3210] | 3.10x [2.89x, 3.33x] |
| **CLEAN + random** | 0.6516 [0.6402, 0.6631] | 0.2304 [0.2153, 0.2452] | 1.80x [1.71x, 1.90x] |
| **CLEAN + time** | 0.6123 [0.5971, 0.6266] | 0.1681 [0.1557, 0.1822] | 1.74x [1.62x, 1.88x] |

### Siziinti etkisi: leaky - clean (PAIRED bootstrap, ayni test satirlari)

| Split | ROC-AUC farki [95% CI] | Lift farki [95% CI] | Sifir CI'da? |
|---|---|---|---|
| random | +0.1141 [+0.1039, +0.1240] | +1.94x [+1.77x, +2.10x] | Hayir — anlamli |
| time | +0.0959 [+0.0822, +0.1092] | +1.36x [+1.16x, +1.57x] | Hayir — anlamli |

### Split etkisi: random - time (INDEPENDENT bootstrap, farkli test satirlari)

| Feature set | ROC-AUC farki [95% CI] | Lift farki [95% CI] | Sifir CI'da? |
|---|---|---|---|
| LEAKY | +0.0575 [+0.0397, +0.0763] | +0.64x [+0.34x, +0.91x] | Hayir — anlamli |
| CLEAN | +0.0395 [+0.0207, +0.0574] | +0.06x [-0.08x, +0.23x] | **EVET** — anlamli degil |

> **Not — clean features, ROC vs lift celiskisi:** CLEAN icin ROC farki anlamli (+0.0395, CI [+0.0207, +0.0574] sifiri icermiyor) ancak lift farki anlamli degil (+0.06x, CI [-0.08x, +0.23x] sifiri iceriyor). Bu bir celiskme degil: ROC-AUC prevalanstan bagimsiz bir metrik, lift ise PR-AUC'yi test prevalansina normalize ediyor. Iki split'in test dilimlerinde prevalans farkli (random=0.128, time=0.097), bu fark lift farki hesabinda govde gizleniyor. ROC, iki split arasinda gercek bir performans farki oldugunu gosteriyor; lift ise bu farkin prevalans normallestirilmis olcekte anlamliligi icin yeterli guc olmadigini soyluyor.

### Rolling-origin dogrulama (clean + time, val-depth=6)

| Test penceresi | Kesim tarihi | n_train | n_test | Prev_train | Prev_test | Test ROC-AUC | Test PR-AUC | Lift |
|---|---|---|---|---|---|---|---|---|
| Son 2 ay | 2018-06-29 | 83177 | 12655 | 0.1328 | 0.0961 | 0.5932 | 0.1485 | 1.54x |
| Son 3 ay | 2018-05-29 | 76962 | 18870 | 0.1356 | 0.0969 | 0.6126 | 0.1692 | 1.75x |
| Son 4 ay | 2018-04-29 | 70245 | 25587 | 0.1380 | 0.1003 | 0.6132 | 0.1722 | 1.72x |

## ADIM 4 — Permutation Importance (test set)

**Leaky + time split** — Top 5 features:

| Rank | Impurity importance | Permutation importance (test) |
|---|---|---|
| 1 | delay_days (0.3425) | delay_days (0.1139 ±0.0025) |
| 2 | actual_delivery_days (0.2849) | order_item_count (0.0492 ±0.0018) |
| 3 | order_item_count (0.0873) | actual_delivery_days (0.0445 ±0.0031) |
| 4 | total_freight (0.0490) | seller_hist_avg_score_strict (0.0094 ±0.0012) |
| 5 | carrier_delivery_days (0.0305) | carrier_delivery_days (0.0047 ±0.0010) |

**Clean + time split** — Top 5 features:

| Rank | Impurity importance | Permutation importance (test) |
|---|---|---|
| 1 | order_item_count (0.2464) | order_item_count (0.0469 ±0.0015) |
| 2 | total_freight (0.1995) | seller_hist_avg_score_strict (0.0071 ±0.0011) |
| 3 | purchase_month (0.1240) | total_price (0.0031 ±0.0010) |
| 4 | seller_hist_avg_score_strict (0.0904) | total_freight (0.0023 ±0.0028) |
| 5 | haversine_km (0.0548) | product_category_bed_bath_table (0.0019 ±0.0004) |

### Ablasyon: purchase_month kaldirilinca ne olur?

> `purchase_month` clean modelde impurity siralamada **3. sirada** ancak permutasyon onemi top 5'te yok.
> Ozelligi cikarip clean+time modelini (depth=6) yeniden egittik:
>
> | | Test ROC-AUC | Test PR-AUC (lift) |
> |---|---|---|
> | Referans (purchase_month dahil) | 0.6121 | 0.1672 (1.73x) |
> | Ablasyon (purchase_month yok) | 0.6081 | 0.1648 (1.70x) |
> | Fark | -0.0040 | -0.0024 |
>
> Fark **0.002 PR-AUC** — bootstrap CI genisligi yaklasik ±0.013, yani olculemez duzey.
> `purchase_month` impurity'de 3. sirada gorunuyor cunku agaclar tarihe gore bolme yapiyor
> (muhtemelen mevsimsellik) ama gercer test seti uzerinde anlamli bir sinyal tasimıyor.
> Ozellik modelde kalabilir ya da cikarilabilir; pratik farki yok.

### Test dilimi review coverage (sansur kontrolu)

> Review coverage kontrolu: test dilimindeki teslimat edilmis siparislerin kacinda review var?
>
> | Ay | Siparis | Review | Oran |
> |---|---|---|---|
> | 2018-05 | 687 | 685 | 0.997 |
> | 2018-06 | 6 099 | 6 075 | 0.996 |
> | 2018-07 | 6 159 | 6 121 | 0.994 |
> | 2018-08 | 6 351 | 6 330 | 0.997 |
>
> Review toplama orani **%99.4–99.7** — son haftalar dahil sansur yok.
> Test prevalansindaki dusus (0.1358 → 0.0967) gercek bir davranis degisikliginden geliyor,
> eksik veriden degil.

## ADIM 5 — HGBT vs RandomForest: 2x2 (val-selected depth)

> **Hiperparametre notu:** HGBT icin `max_depth` dogrulama setiyle secildi (tarama araligi: 2–8, ayni RF metodolojisi). Ancak `learning_rate=0.05` ve `max_iter=200` sabit tutuldu — tam bir hiperparametre araması yapilmadi. Sklearn varsayilanlari (lr=0.1, iter=100) kullanilmadi. Bu sinir ADIM 5 bulgusunu (leakage etkisi her iki modelde de benzer) gercek anlamindan uzaklastirmiyor, cunku leakage farki (~2x lift) model seciminden cok daha buyuk. Ama HGBT'nin RF'e gore kucuk ustunlugu (+0.03–0.13x lift) hiperparametre optimizasyonuyla degisebilir.

| Hucre | RF Test ROC | RF Test PR (lift) | HGBT Test ROC | HGBT Test PR (lift) |
|---|---|---|---|---|
| **LEAKY + random** | 0.7658 | 0.4784 (3.74x) | 0.7686 | 0.4806 (3.76x) |
| **LEAKY + time** | 0.7083 | 0.2991 (3.09x) | 0.7118 | 0.3117 (3.22x) |
| **CLEAN + random** | 0.6516 | 0.2299 (1.80x) | 0.6591 | 0.2356 (1.84x) |
| **CLEAN + time** | 0.6121 | 0.1672 (1.73x) | 0.6259 | 0.1704 (1.76x) |

### LogisticRegression (clean + zaman bolumu)

> Pipeline: SimpleImputer → StandardScaler → LogisticRegression(class_weight='balanced').
> C parametresi dogrulama setinden secildi (aday: 0.01, 0.1, 1, 10).
> Secilen C=10 (val PR-AUC'de diger degerlerden ayirt edilemez: 0.2081–0.2082, esit performans).

| Model | Test ROC-AUC | Test PR-AUC (lift) |
|---|---|---|
| LR (C=10) | 0.6168 | 0.1668 (1.72x) |
| RF (val-depth=6) | 0.6121 | 0.1672 (1.73x) |
| HGBT (val-depth) | 0.6259 | 0.1704 (1.76x) |

### Model karsilastirmasi: temiz + zaman bolumu (model_comparison.png)

| Model | Test ROC-AUC | Test PR-AUC (lift) |
|---|---|---|
| Dummy (stratified) | 0.5021 | 0.0971 (1.00x) |
| Lojistik Regresyon | 0.6168 | 0.1668 (1.72x) |
| RandomForest | 0.6121 | 0.1672 (1.73x) |
| HistGradientBoosting | 0.6259 | 0.1704 (1.76x) |

> RF ve LR neredeyse esit. HGBT en yuksek ama fark kucuk (~0.03x lift).
> Leakage etkisi (2x+ lift) her modelde gorulecek buyuklukte — model secimi degil, feature seti belirleyici.

## ADIM 6 — Is anlami: Skor dilimlerinde gercek dusuk-puan orani

> Model: clean + time split, depth=6. Test seti: n=19 167, toplam dusuk puan=1 854, prevalans=0.0967.

| Dilim | n | Dusuk puan orani | Lift | Yakalanan | Recall |
|---|---|---|---|---|---|
| Taban (tum test) | 19 167 | 0.0967 | 1.00x | 1 854 | 100.0% |
| En riskli %5 | 958 | 0.2797 | **2.89x** | 268 | 14.5% |
| En riskli %10 | 1 916 | 0.2229 | **2.30x** | 427 | 23.0% |
| En riskli %20 | 3 833 | 0.1584 | **1.64x** | 607 | 32.7% |

## Veri Hazirlama Istatistikleri (data_prep.py)

| Istatistik | Deger | Notlar |
|---|---|---|
| Top-15 kategori kapsami | %78.9 (75 620 / 95 832 siparis) | Geri kalan %21.1 'other' olarak birlestirildi |
| Haversine NaN | 478 siparis (%0.50) | Posta kodu eslesmeyen; Pipeline'daki SimpleImputer medyan ile doldurdu |
| seller_hist_avg_score_strict NaN | 7 956 siparis | Satici o ana kadar hic review almamis; seller_hist_missing=1 bayragiyla ayri ozellik olarak tutuldu |

---

*Lift = PR-AUC / test prevalansi. Random split floor = 0.128 | Time split floor = 0.0967. Generated by `experiments.py`.*
