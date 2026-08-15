import pandas as pd
import numpy as np

clean = pd.read_parquet("data/processed/features_clean.parquet")

META   = ["order_id", "order_purchase_timestamp"]
TARGET = "is_low_review"
prod_cat_cols = [c for c in clean.columns if c.startswith("product_category_")]
pay_type_cols = [c for c in clean.columns if c.startswith("payment_type_")]
one_hot_cols  = prod_cat_cols + pay_type_cols
numeric_cols  = [c for c in clean.columns
                 if c not in META + [TARGET] + one_hot_cols]

print("=== CHECK 1: Column inventory ===")
print(f"Meta        ({len(META)}): {META}")
print(f"Target       (1): [{TARGET}]")
print(f"Numeric     ({len(numeric_cols)}): {numeric_cols}")
print(f"payment_type one-hot ({len(pay_type_cols)}): {pay_type_cols}")
print(f"product_category one-hot: {len(prod_cat_cols)} kolon (15'ten fazla -> konsolidasyon gerekli)")

print()
print("=== CHECK 2: order_purchase_timestamp ===")
present = "order_purchase_timestamp" in clean.columns
print(f"Present : {present}")
if present:
    print(f"dtype   : {clean['order_purchase_timestamp'].dtype}")
    print(f"sample  : {clean['order_purchase_timestamp'].iloc[0]}")

print()
print("=== Top 20 product_category frekans ===")
cat_sums = clean[prod_cat_cols].sum().sort_values(ascending=False)
for name, val in cat_sums.head(20).items():
    short = name.replace("product_category_", "")
    print(f"  {short:<55s} {int(val):>6,}")
print(f"  ... ve {len(prod_cat_cols) - 20} kolon daha")

print()
print("=== seller_hist_avg_score NaN ===")
col = "seller_hist_avg_score"
n_nan = clean[col].isna().sum()
print(f"NaN: {n_nan:,} / {len(clean):,}  ({n_nan / len(clean) * 100:.2f}%)")
