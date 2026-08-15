# olist-leakage-analysis

Olist Brazilian E-Commerce veri setinde veri sızıntısı, ezber ve
metrik seçiminin model performansına etkisinin ölçülmesi.

**Medium yazısı:** *(link sonra eklenecek)*

---

## Ana bulgu

Aynı veri, aynı algoritma. Sızıntılı kurulum lift **3,74×**, temiz kurulum **1,73×**.
Fark iki metodolojik karardan ibaret: sipariş anında bilinmeyen kolonların
kullanılması ve zaman serisi verisinin rastgele bölünmesi.

---

## Problem

Sipariş verildiği anda müşterinin düşük puan (`review_score ≤ 2`) verme
riskini tahmin etmek. 95.832 teslim edilmiş sipariş, sınıf oranı %12,8.

---

## Kurulum

### 1. Veriyi indir

Kaggle'dan [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) veri setini indirip `data/raw/` altına çıkar:

```
data/raw/
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_customers_dataset.csv
├── olist_sellers_dataset.csv
├── olist_products_dataset.csv
├── product_category_name_translation.csv
└── olist_geolocation_dataset.csv
```

### 2. Bağımlılıkları kur

```bash
pip install -r requirements.txt
```

### 3. Çalıştır

```bash
# Veri hazırlama (features_clean.parquet + features_leaky.parquet)
python src/data_prep.py

# Deneylerin tamamı
python src/experiments.py

# Makale grafikleri
python src/make_figures.py

# Tablo PNG'leri
python src/make_tables.py
```

---

## Deneyler

| Deney | Açıklama |
|---|---|
| exp1 | Sızıntılı özellik setiyle RandomForest (rastgele bölme) — tavan referansı |
| exp2 | Temiz özellik setiyle RandomForest (rastgele bölme) — dürüst üst sınır |
| exp4 | Karıştırılmış etiketlerle model — unbounded ve depth-6 RF ile ezber kapasitesi testi |
| exp5 | max\_depth 1–30 kapasite eğrisi (clean + zaman bölmesi, validation bazlı derinlik seçimi) |
| exp6 | DummyClassifier ve freight-oranı kural tabanı — taban çizgisi |
| exp8 | Katı vs. gevşek satıcı geçmişi skoru — zamansal sızıntı testi (strict/non-strict) |

Ek analizler: 2×2 grid (sızıntı × bölme stratejisi), bootstrap CI, rolling-origin
doğrulaması, permütasyon önemi, LogisticRegression ve HistGradientBoosting karşılaştırması.

---

## Sonuçlar

| Kurulum | Test ROC-AUC | Test Lift | Not |
|---|---|---|---|
| Sızıntılı + rastgele bölme | 0,766 | 3,74× | Referans üst sınır |
| Temiz + rastgele bölme | 0,652 | 1,80× | Dürüst üst sınır |
| Temiz + zaman bölmesi | 0,612 | 1,73× [1,62×–1,88×] | Gerçekçi kurulum |

En riskli %5'lik dilimde gerçek düşük puan oranı: **%28,0** (genel oranın 2,9 katı).

Tam tablolar: [`results/summary.md`](results/summary.md)  
Grafikler: [`results/figures/`](results/figures/)  
Tablo PNG'leri: [`results/figures/tables/`](results/figures/tables/)

---

## Yöntem notları

- Tüm ön işleme sklearn `Pipeline` içinde; `fit` yalnızca eğitim fold'unda
- Hiperparametreler eğitim setinin son %20'si (zaman bazlı) üzerinden seçildi; test setine bir kez dokunuldu
- Güven aralıkları test seti üzerinde bootstrap (n = 1.000)
- Zaman bazlı bölme: eğitim %12,8, test %9,7 prevalans — PR-AUC'ler doğrudan karşılaştırılamaz
- Lift = PR-AUC / test dilimi prevalansı (0,097); ROC-AUC prevalanstan bağımsız

---

## Lisans

MIT
