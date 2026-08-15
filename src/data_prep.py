#!/usr/bin/env python3
"""
data_prep.py

Loads Olist Brazilian E-Commerce data, joins all tables, engineers features,
and writes two parquet files:

  data/processed/features_leaky.parquet  – includes post-purchase information
                                           that constitutes data leakage
  data/processed/features_clean.parquet  – only information available at the
                                           exact moment the order is placed

Run:
    python src/data_prep.py
"""

import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

TARGET = "is_low_review"

# Categorical columns (one-hot encoded identically in both sets)
CAT_COLS = ["product_category", "payment_type"]

# Keep only the N most frequent product categories; the rest collapse to 'other'
TOP_PRODUCT_CATEGORIES = 15

# ── Feature groups ────────────────────────────────────────────────────────────
# MODEL_NUMERIC  : features used in all standard experiments (exp1-exp6)
# COMPARISON_ONLY: kept in parquet only for exp8 side-by-side comparison;
#                  NOT fed to any model in other experiments

MODEL_NUMERIC = [
    "total_price",                  # sum of item prices in the order
    "total_freight",                # sum of freight costs
    "freight_ratio",                # total_freight / total_price
    "order_item_count",             # number of order lines
    "avg_weight_g",                 # average product weight
    "avg_volume_cm3",               # average product volume (l*h*w)
    "avg_photos_qty",               # average number of product photos
    "payment_installments",         # maximum installment count chosen by buyer
    "same_state",                   # 1 if buyer and seller are in the same state
    "haversine_km",                 # great-circle distance buyer ↔ seller zip (km)
    "estimated_delivery_days",      # (estimated_delivery_date - purchase_date).days
    "purchase_month",               # calendar month of purchase
    "purchase_dayofweek",           # 0=Monday ... 6=Sunday
    "purchase_hour",                # hour of purchase (local time stored in dataset)
    "seller_hist_avg_score_strict", # expanding mean keyed by review_creation_date
                                    # (only reviews written before this order's purchase)
    "seller_hist_missing",          # 1 when seller_hist_avg_score_strict is NaN
                                    # (seller had no prior reviewed orders)
]

# Non-strict seller score: sorting by order_purchase_timestamp instead of
# review_creation_date. Subtle leak — prior orders' reviews may not exist yet.
# Stored in the parquet for exp8 only.
COMPARISON_ONLY_NUMERIC = ["seller_hist_avg_score"]

# Combined lists used when writing parquet files
CLEAN_NUMERIC  = MODEL_NUMERIC + COMPARISON_ONLY_NUMERIC

# Columns derived from timestamps that are unknown at order time -> hard leakage
LEAKY_EXTRA = [
    "actual_delivery_days",   # (order_delivered_customer_date - purchase).days
    "carrier_delivery_days",  # (order_delivered_carrier_date  - purchase).days
    "delay_days",             # actual_delivery_days - estimated_delivery_days
    "approval_time_hours",    # (order_approved_at - purchase).total_seconds() / 3600
]

CLEAN_FEATURES = CLEAN_NUMERIC + CAT_COLS
LEAKY_FEATURES = CLEAN_NUMERIC + LEAKY_EXTRA + CAT_COLS


# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_km_vec(lat1: pd.Series, lon1: pd.Series,
                     lat2: pd.Series, lon2: pd.Series) -> np.ndarray:
    """Vectorized haversine distance (km); propagates NaN for missing coords."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = (np.radians(s.to_numpy()) for s in [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def series_mode(s: pd.Series):
    """Mode of a Series; returns NaN on all-missing input."""
    m = s.dropna().mode()
    return m.iloc[0] if not m.empty else np.nan


def consolidate_product_categories(
    df: pd.DataFrame,
    col: str = "product_category",
    top_n: int = TOP_PRODUCT_CATEGORIES,
) -> pd.DataFrame:
    """
    Keep the top_n most frequent categories; map everything else (including NaN)
    to the string 'other'. Reports NaN (truly missing) separately from rare
    categories so the two sources of missingness can be distinguished.
    """
    n_total  = len(df)
    n_nan    = df[col].isna().sum()           # rows with NO product data at all
    top      = df[col].value_counts().head(top_n).index.tolist()
    n_rare   = (~df[col].isin(top) & df[col].notna()).sum()  # non-top, non-NaN

    df[col]  = df[col].where(df[col].isin(top), other="other")

    print(f"  product_category breakdown BEFORE consolidation:")
    print(f"    truly missing (NaN)    : {n_nan:,}  ({n_nan/n_total*100:.2f}%) "
          f"-- orders with no item data")
    print(f"    rare categories (non-top-{top_n}): {n_rare:,}  ({n_rare/n_total*100:.2f}%) "
          f"-- valid but infrequent")
    print(f"    -> both collapsed to 'other' ({n_nan+n_rare:,} total)")
    print(f"    top-{top_n} categories kept: {top}")
    return df


def compute_strict_seller_score(
    seller_tl: pd.DataFrame,
    reviews_raw: pd.DataFrame,
) -> pd.Series:
    """
    Strict seller score: for each order X, compute the seller's mean score using
    only reviews whose review_creation_date < X.order_purchase_timestamp.

    This differs from seller_hist_avg_score (which sorts by order_purchase_timestamp)
    because reviews arrive days/weeks after delivery — the non-strict version
    includes scores that were NOT yet written when X was placed.

    Uses merge_asof (O(n log n)) to avoid a slow nested loop.
    Returns a Series indexed by order_id.
    """
    rev_dates = (
        reviews_raw
        .sort_values("review_creation_date")
        .drop_duplicates("order_id", keep="first")
        [["order_id", "review_creation_date"]]
    )
    tl = seller_tl.merge(rev_dates, on="order_id", how="left")

    # Cumulative stats per seller, sorted by review_creation_date
    review_events = (
        tl.dropna(subset=["review_creation_date"])
        .sort_values(["seller_id", "review_creation_date"])
        .reset_index(drop=True)
    )
    review_events["_cum_count"] = review_events.groupby("seller_id").cumcount() + 1
    review_events["_cum_sum"]   = (
        review_events.groupby("seller_id")["review_score"].cumsum()
    )

    # Only orders with a valid purchase timestamp
    orders_q = (
        seller_tl[["order_id", "seller_id", "order_purchase_timestamp"]]
        .dropna(subset=["order_purchase_timestamp"])
    )

    # Per-seller numpy searchsorted: for each order, count reviews with
    # creation_date < purchase_timestamp (side="left" = strictly less than)
    strict_results: dict = {}
    for seller_id, rev_group in review_events.groupby("seller_id"):
        ord_group = orders_q[orders_q["seller_id"] == seller_id]
        if ord_group.empty:
            continue
        rev_ts_arr   = rev_group["review_creation_date"].values   # already sorted asc
        cum_cnt      = rev_group["_cum_count"].values
        cum_sum      = rev_group["_cum_sum"].values
        purchase_arr = ord_group["order_purchase_timestamp"].values
        order_ids    = ord_group["order_id"].values

        idxs  = np.searchsorted(rev_ts_arr, purchase_arr, side="left")
        valid = idxs > 0
        safe  = np.clip(idxs - 1, 0, len(cum_sum) - 1)
        scores = np.where(valid, cum_sum[safe] / cum_cnt[safe], np.nan)
        strict_results.update(zip(order_ids, scores.tolist()))

    result = pd.Series(strict_results, name="seller_hist_avg_score_strict")
    result.index.name = "order_id"
    return result


# ── Step 1: Load raw CSVs ─────────────────────────────────────────────────────

def load_raw() -> dict[str, pd.DataFrame]:
    """Load every raw CSV and print shape + column list for sanity-checking."""
    manifest = {
        "orders":    "olist_orders_dataset.csv",
        "items":     "olist_order_items_dataset.csv",
        "reviews":   "olist_order_reviews_dataset.csv",
        "payments":  "olist_order_payments_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "sellers":   "olist_sellers_dataset.csv",
        "products":  "olist_products_dataset.csv",
        "cat_trans": "product_category_name_translation.csv",
        "geo":       "olist_geolocation_dataset.csv",
    }
    dfs: dict[str, pd.DataFrame] = {}
    for name, fname in manifest.items():
        df = pd.read_csv(os.path.join(RAW_DIR, fname))
        print(f"  [{name:10s}] shape={str(df.shape):20s} "
              f"cols={list(df.columns)}")
        dfs[name] = df
    return dfs


# ── Step 2: Merge all tables ──────────────────────────────────────────────────

def merge_all(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join tables on order_id / customer_id / seller_id.
    Keeps only delivered orders that have at least one review.
    Returns a wide DataFrame with raw columns intact (feature engineering later).
    """
    orders    = dfs["orders"].copy()
    items     = dfs["items"].copy()
    reviews   = dfs["reviews"].copy()
    payments  = dfs["payments"].copy()
    customers = dfs["customers"].copy()
    sellers   = dfs["sellers"].copy()
    products  = dfs["products"].copy()
    cat_trans = dfs["cat_trans"].copy()
    geo       = dfs["geo"].copy()

    # -- Parse timestamps --
    date_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col])
    reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"])

    # -- Filter: delivered orders only --
    orders = orders[orders["order_status"] == "delivered"].reset_index(drop=True)
    print(f"  Delivered orders           : {len(orders):,}")

    # -- Target variable --
    # Multiple reviews per order are rare; keep the earliest one by creation date.
    rev_first = (
        reviews
        .sort_values("review_creation_date")
        .drop_duplicates("order_id", keep="first")
        [["order_id", "review_score"]]
    )
    orders = orders.merge(rev_first, on="order_id", how="inner")
    orders[TARGET] = (orders["review_score"] <= 2).astype(int)
    print(f"  Orders with a review       : {len(orders):,}")
    print(f"  Low-review rate (score<=2) : {orders[TARGET].mean():.3%}")

    # -- Products: add English category name and volume --
    products = products.merge(cat_trans, on="product_category_name", how="left")
    products["product_volume_cm3"] = (
        products["product_length_cm"]
        * products["product_height_cm"]
        * products["product_width_cm"]
    )

    items_rich = items.merge(
        products[[
            "product_id", "product_category_name_english",
            "product_weight_g", "product_volume_cm3", "product_photos_qty",
        ]],
        on="product_id", how="left",
    )

    # -- Aggregate items to order level --
    items_agg = (
        items_rich
        .groupby("order_id", sort=False)
        .agg(
            total_price       =("price",                          "sum"),
            total_freight     =("freight_value",                  "sum"),
            order_item_count  =("order_item_id",                  "count"),
            product_category  =("product_category_name_english",  series_mode),
            avg_weight_g      =("product_weight_g",               "mean"),
            avg_volume_cm3    =("product_volume_cm3",             "mean"),
            avg_photos_qty    =("product_photos_qty",             "mean"),
        )
        .reset_index()
    )

    # Collapse rare categories to 'other' (prevents ~70 near-empty one-hot columns)
    items_agg = consolidate_product_categories(items_agg)

    # -- Primary seller per order (the seller with the most items) --
    primary_seller = (
        items.groupby(["order_id", "seller_id"])
        .size()
        .reset_index(name="_cnt")
        .sort_values(["order_id", "_cnt"], ascending=[True, False])
        .drop_duplicates("order_id")
        [["order_id", "seller_id"]]
    )

    # -- Payments: first payment type + max installment count per order --
    pay_agg = (
        payments[payments["payment_type"] != "not_defined"]
        .sort_values(["order_id", "payment_sequential"])
        .groupby("order_id")
        .agg(
            payment_type         =("payment_type",          "first"),
            payment_installments =("payment_installments",  "max"),
        )
        .reset_index()
    )

    # -- Geolocation: median lat/lng per zip prefix (zip codes have many duplicates) --
    geo_med = (
        geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .median()
        .reset_index()
        .rename(columns={
            "geolocation_zip_code_prefix": "zip",
            "geolocation_lat": "lat",
            "geolocation_lng": "lng",
        })
    )

    cust_geo = customers.merge(
        geo_med.rename(columns={"zip": "customer_zip_code_prefix",
                                "lat": "cust_lat", "lng": "cust_lng"}),
        on="customer_zip_code_prefix", how="left",
    )
    sell_geo = sellers.merge(
        geo_med.rename(columns={"zip": "seller_zip_code_prefix",
                                "lat": "sell_lat", "lng": "sell_lng"}),
        on="seller_zip_code_prefix", how="left",
    )

    # -- Seller historical average score (two variants) --
    #
    # Non-strict (seller_hist_avg_score):
    #   Sorted by order_purchase_timestamp; shift(1) = mean of all PRIOR orders.
    #   Subtle leak: a "prior" order's review may not yet be written at purchase time.
    #
    # Strict (seller_hist_avg_score_strict):
    #   Uses review_creation_date as the availability signal. Only includes scores
    #   whose review was written BEFORE the current order was placed.
    seller_tl = (
        primary_seller
        .merge(
            orders[["order_id", "order_purchase_timestamp", "review_score"]],
            on="order_id", how="left",
        )
        .sort_values(["seller_id", "order_purchase_timestamp"])
        .reset_index(drop=True)
    )
    seller_tl["seller_hist_avg_score"] = (
        seller_tl.groupby("seller_id")["review_score"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )

    strict_scores = compute_strict_seller_score(seller_tl, reviews)
    seller_tl = seller_tl.join(strict_scores, on="order_id")

    # -- Final merge --
    df = (
        orders
        .merge(items_agg,                                                   on="order_id",   how="left")
        .merge(primary_seller,                                              on="order_id",   how="left")
        .merge(pay_agg,                                                     on="order_id",   how="left")
        .merge(cust_geo[["customer_id", "customer_state", "cust_lat", "cust_lng"]],
               on="customer_id", how="left")
        .merge(sell_geo[["seller_id", "seller_state", "sell_lat", "sell_lng"]],
               on="seller_id",   how="left")
        .merge(seller_tl[["order_id", "seller_hist_avg_score",
                           "seller_hist_avg_score_strict"]],                on="order_id",   how="left")
    )

    print(f"  Merged DataFrame shape     : {df.shape}")
    return df


# ── Step 3: Feature engineering ───────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive clean and leaky features from the merged DataFrame.
    Clean features use only information available at purchase time.
    Leaky features are derived from timestamps that occur after the order.
    """
    ts = df["order_purchase_timestamp"]

    # ---- Clean features ----
    df["freight_ratio"] = (
        df["total_freight"] / df["total_price"].replace(0.0, np.nan)
    )

    # NaN-safe same_state: False only when both states are known
    both_known = df["customer_state"].notna() & df["seller_state"].notna()
    df["same_state"] = np.where(
        both_known,
        (df["customer_state"] == df["seller_state"]).astype(float),
        np.nan,
    )

    df["haversine_km"] = haversine_km_vec(
        df["cust_lat"], df["cust_lng"],
        df["sell_lat"], df["sell_lng"],
    )

    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - ts
    ).dt.days

    df["purchase_month"]      = ts.dt.month
    df["purchase_dayofweek"]  = ts.dt.dayofweek   # 0 = Monday
    df["purchase_hour"]       = ts.dt.hour

    # Binary flag: 1 when the strict seller score is NaN (no prior reviewed orders).
    # Letting the imputer fill NaN with a median would erase this signal, so we
    # preserve it explicitly as a feature before imputation happens in the Pipeline.
    df["seller_hist_missing"] = df["seller_hist_avg_score_strict"].isna().astype(float)

    # ---- Leaky features (unknown at order time) ----
    df["actual_delivery_days"]  = (df["order_delivered_customer_date"] - ts).dt.days
    df["carrier_delivery_days"] = (df["order_delivered_carrier_date"]  - ts).dt.days
    df["delay_days"]            = df["actual_delivery_days"] - df["estimated_delivery_days"]
    df["approval_time_hours"]   = (
        (df["order_approved_at"] - ts).dt.total_seconds() / 3600.0
    )

    print(f"  Shape after feature engineering: {df.shape}")
    return df


# ── Step 4: Save feature sets ─────────────────────────────────────────────────

def save_feature_sets(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode categorical columns and persist two parquet files:
      features_leaky.parquet and features_clean.parquet.
    Meta columns (order_id, order_purchase_timestamp) are kept for time-split
    experiments but are NOT used as model features.
    """
    df = df.dropna(subset=[TARGET]).copy()

    meta = ["order_id", "order_purchase_timestamp"]

    def _save(feature_cols: list[str], fname: str) -> pd.DataFrame:
        cols = meta + feature_cols + [TARGET]
        sub  = df[cols].copy()
        # Any residual NaN in categorical columns (orders with no items) -> 'other'
        for c in CAT_COLS:
            if c in sub.columns:
                sub[c] = sub[c].fillna("other")
        sub  = pd.get_dummies(
            sub,
            columns=CAT_COLS,
            dummy_na=False,
            dtype=float,
        )
        path = os.path.join(PROCESSED_DIR, fname)
        sub.to_parquet(path, index=False)
        n_feat = sub.shape[1] - len(meta) - 1
        print(f"  {fname}: {sub.shape[0]:,} rows × {sub.shape[1]} cols "
              f"({n_feat} features + {len(meta)} meta + 1 target)")
        return sub

    leaky = _save(LEAKY_FEATURES, "features_leaky.parquet")
    clean = _save(CLEAN_FEATURES, "features_clean.parquet")
    return leaky, clean


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sep = "=" * 62

    print(f"\n{sep}")
    print("STEP 1 — Load raw CSVs")
    print(sep)
    dfs = load_raw()

    print(f"\n{sep}")
    print("STEP 2 — Merge tables")
    print(sep)
    df = merge_all(dfs)

    print(f"\n{sep}")
    print("STEP 3 — Engineer features")
    print(sep)
    df = engineer_features(df)

    print(f"\n{sep}")
    print("STEP 4 — Save feature sets")
    print(sep)
    leaky, clean = save_feature_sets(df)

    # ── Verification output ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("VERIFICATION")
    print(sep)

    # Check 1: column inventory of clean set
    prod_cols  = [c for c in clean.columns if c.startswith("product_category_")]
    pay_cols   = [c for c in clean.columns if c.startswith("payment_type_")]
    meta_v     = ["order_id", "order_purchase_timestamp"]
    num_cols   = [c for c in clean.columns
                  if c not in meta_v + [TARGET] + prod_cols + pay_cols]
    print("\n[CHECK 1] Clean feature set column inventory:")
    print(f"  Meta              ({len(meta_v):2d}): {meta_v}")
    print(f"  Numeric/bool      ({len(num_cols):2d}): {num_cols}")
    print(f"  payment_type OHE  ({len(pay_cols):2d}): {pay_cols}")
    print(f"  product_category OHE ({len(prod_cols):2d}): "
          f"{[c.replace('product_category_','') for c in prod_cols]}")

    # Check 2: order_purchase_timestamp present
    print("\n[CHECK 2] order_purchase_timestamp in clean set:", "order_purchase_timestamp" in clean.columns)

    # Check 3: seller score comparison
    s1 = clean["seller_hist_avg_score"]
    s2 = clean["seller_hist_avg_score_strict"]
    corr = s1.corr(s2)
    nan1 = s1.isna().sum()
    nan2 = s2.isna().sum()
    extra_nan = (s2.isna() & s1.notna()).sum()
    missing_flag = clean["seller_hist_missing"].sum()
    print(f"\n[CHECK 3] seller_hist_avg_score vs seller_hist_avg_score_strict:")
    print(f"  Pearson correlation        : {corr:.6f}")
    print(f"  NaN in non-strict          : {nan1:,}  ({nan1/len(clean)*100:.2f}%) -- sellers with no prior orders")
    print(f"  NaN in strict              : {nan2:,}  ({nan2/len(clean)*100:.2f}%) -- sellers with no prior REVIEWS")
    print(f"  Extra NaN (strict only)    : {extra_nan:,}  -- prior orders existed but reviews not yet written")
    print(f"  seller_hist_missing=1      : {int(missing_flag):,}  (matches NaN in strict)")
    print(f"  Mean non-strict            : {s1.mean():.4f}")
    print(f"  Mean strict                : {s2.mean():.4f}")

    # Check 4: model vs comparison column split
    model_cols_ohe = [
        c for c in clean.columns
        if c not in ["order_id", "order_purchase_timestamp", TARGET,
                     "seller_hist_avg_score"]
    ]
    comparison_cols = ["seller_hist_avg_score"]
    print(f"\n[CHECK 4] Column split (post OHE):")
    print(f"  Standard model features ({len(model_cols_ohe)}): used in exp1-exp6")
    print(f"  Comparison-only         ({len(comparison_cols)}): {comparison_cols} -- exp8 only")

    print(f"\nDone. Files written to {PROCESSED_DIR}/")
