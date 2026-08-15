#!/usr/bin/env python3
"""
experiments.py

Six experiments on the Olist dataset illustrating data leakage,
model memorization, and the effect of correct temporal feature engineering.

Experiments
-----------
  exp1 : RandomForest on leaky feature set (random split)
  exp2 : RandomForest on clean feature set (random split)
  exp4 : Model trained on permuted labels — memorization probe
  exp5 : Train/test PR-AUC vs max_depth — capacity / overfitting curve
  exp6 : DummyClassifiers + freight-rule baseline
  exp8 : Strict vs non-strict seller score (time-based split)

Outputs
-------
  results/<exp_name>.json
  results/figures/<exp_name>_*.png  (150 dpi)

Run:
    python src/experiments.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ── Paths & constants ─────────────────────────────────────────────────────────

PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_DIR   = "results"
FIGURES_DIR   = os.path.join("results", "figures")
os.makedirs(RESULTS_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR,  exist_ok=True)

RANDOM_STATE  = 42
TARGET        = "is_low_review"
META_COLS     = ["order_id", "order_purchase_timestamp"]
# Kept in parquet for exp8 only — excluded from all other experiments
NONSTRICT_COL = "seller_hist_avg_score"
# Post-purchase columns present only in features_leaky.parquet
LEAKY_COLS    = ["actual_delivery_days", "carrier_delivery_days",
                 "delay_days", "approval_time_hours"]

# Positive class rate (used as PR-AUC floor reference line)
POS_RATE = 0.128

# ── Plot style ────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size":         14,
    "axes.titlesize":    17,
    "axes.labelsize":    14,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
})

C_DARK  = "#1b4332"
C_MED   = "#2d6a4f"
C_LIGHT = "#52b788"
C_PALE  = "#95d5b2"
C_WARM  = "#e76f51"
C_GOLD  = "#f4a261"


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_clean(include_nonstrict: bool = False):
    """
    Load features_clean.parquet.
    Returns (df, feature_cols) where feature_cols excludes meta, target, and
    by default the non-strict seller score column (comparison-only).
    """
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "features_clean.parquet"))
    exclude = set(META_COLS + [TARGET])
    if not include_nonstrict:
        exclude.add(NONSTRICT_COL)
    feature_cols = [c for c in df.columns if c not in exclude]
    return df, feature_cols


def load_leaky():
    """
    Load features_leaky.parquet.
    Excludes meta, target, and non-strict seller score (comparison-only).
    """
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "features_leaky.parquet"))
    exclude = set(META_COLS + [TARGET, NONSTRICT_COL])
    feature_cols = [c for c in df.columns if c not in exclude]
    return df, feature_cols


# ── Split helpers ─────────────────────────────────────────────────────────────

def stratified_split(df, feature_cols):
    """Stratified 80/20 random split. Returns X_train, X_test, y_train, y_test."""
    X = df[feature_cols].values
    y = df[TARGET].values
    return train_test_split(X, y, test_size=0.2, stratify=y,
                            random_state=RANDOM_STATE)


def time_split(df, feature_cols, test_frac=0.20):
    """
    Chronological 80/20 split sorted by order_purchase_timestamp.
    Earliest rows -> train, latest rows -> test.
    """
    df_s  = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    cut   = int(len(df_s) * (1.0 - test_frac))
    X_tr  = df_s.iloc[:cut][feature_cols].values
    X_te  = df_s.iloc[cut:][feature_cols].values
    y_tr  = df_s.iloc[:cut][TARGET].values
    y_te  = df_s.iloc[cut:][TARGET].values
    return X_tr, X_te, y_tr, y_te


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_proba, y_pred=None):
    """ROC-AUC, PR-AUC, and optionally F1."""
    out = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc":  float(average_precision_score(y_true, y_proba)),
    }
    if y_pred is not None:
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    return out


def print_metrics(label, m):
    f1_str = f"  F1={m['f1']:.4f}" if "f1" in m else ""
    print(f"  {label:<22s} ROC-AUC={m['roc_auc']:.4f}  "
          f"PR-AUC={m['pr_auc']:.4f}{f1_str}")


# ── Pipeline factory ──────────────────────────────────────────────────────────

def make_rf_pipeline(n_estimators=200, max_depth=None):
    """
    RandomForest inside a Pipeline.
    The SimpleImputer is part of the Pipeline so it is always fit on training
    data only — never before the train/test split.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_json(data, name):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"    -> {path}")


def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")


def _short_name(col):
    """Shorten long one-hot column names for axis labels."""
    return (col.replace("product_category_", "cat:")
               .replace("payment_type_", "pay:"))


# ── Feature importance plot (shared by exp1 and exp2) ────────────────────────

def _plot_feature_importance(pipe, feature_cols, title, fname):
    importances = pipe.named_steps["model"].feature_importances_
    idx_top     = np.argsort(importances)[::-1][:20]
    names_top   = [_short_name(feature_cols[i]) for i in idx_top]
    vals_top    = importances[idx_top]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(names_top))[::-1], vals_top, color=C_MED, height=0.7)
    ax.set_yticks(range(len(names_top)))
    ax.set_yticklabels(reversed(names_top), fontsize=11)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(title)
    fig.tight_layout()
    save_fig(fig, fname)


# ── Experiments ───────────────────────────────────────────────────────────────

def exp1_leaky_model():
    """
    RandomForest on the leaky feature set with stratified random split.
    Expected result: test AUC suspiciously close to train AUC because post-
    purchase delivery data (actual_delivery_days, delay_days) is near-
    deterministic with respect to the review score.
    """
    print("\n--- EXP1: Leaky Model ---")
    df, feat_cols = load_leaky()
    X_train, X_test, y_train, y_test = stratified_split(df, feat_cols)

    pipe = make_rf_pipeline()
    pipe.fit(X_train, y_train)

    train_m = compute_metrics(y_train, pipe.predict_proba(X_train)[:, 1])
    test_m  = compute_metrics(y_test,  pipe.predict_proba(X_test)[:, 1],
                              pipe.predict(X_test))
    print_metrics("train", train_m)
    print_metrics("test",  test_m)

    _plot_feature_importance(
        pipe, feat_cols,
        "EXP1 — Leaky Model: Top-20 Feature Importances",
        "exp1_feature_importance",
    )

    result = {"train": train_m, "test": test_m, "n_features": len(feat_cols)}
    save_json(result, "exp1_leaky_model")
    return result


def exp2_clean_model():
    """
    RandomForest on the clean feature set (no post-purchase data), random split.
    This is the honest performance ceiling: what is achievable at order time.
    Compare with exp1 to quantify the leakage inflation.
    """
    print("\n--- EXP2: Clean Model ---")
    df, feat_cols = load_clean()
    X_train, X_test, y_train, y_test = stratified_split(df, feat_cols)

    pipe = make_rf_pipeline()
    pipe.fit(X_train, y_train)

    train_m = compute_metrics(y_train, pipe.predict_proba(X_train)[:, 1])
    test_m  = compute_metrics(y_test,  pipe.predict_proba(X_test)[:, 1],
                              pipe.predict(X_test))
    print_metrics("train", train_m)
    print_metrics("test",  test_m)

    _plot_feature_importance(
        pipe, feat_cols,
        "EXP2 — Clean Model: Top-20 Feature Importances",
        "exp2_feature_importance",
    )

    result = {"train": train_m, "test": test_m, "n_features": len(feat_cols)}
    save_json(result, "exp2_clean_model")
    return result


def exp4_label_permutation():
    """
    Randomly permute the target vector and train the same clean-feature RF.
    A high train AUC on permuted labels proves the model has memorization
    capacity. A near-0.5 test AUC proves it cannot extract any real signal.
    The gap between train and test is the 'memorization budget'.
    """
    print("\n--- EXP4: Label Permutation ---")
    df, feat_cols = load_clean()
    X = df[feat_cols].values
    y_real = df[TARGET].values
    y_perm = np.random.default_rng(RANDOM_STATE).permutation(y_real)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_perm, test_size=0.2, random_state=RANDOM_STATE
    )

    pipe = make_rf_pipeline()
    pipe.fit(X_train, y_train)

    train_m = compute_metrics(y_train, pipe.predict_proba(X_train)[:, 1])
    test_m  = compute_metrics(y_test,  pipe.predict_proba(X_test)[:, 1])
    print_metrics("train (permuted labels)", train_m)
    print_metrics("test  (permuted labels)", test_m)

    # Horizontal bar comparing train vs test AUC
    metrics_pairs = {
        "Train ROC-AUC": train_m["roc_auc"],
        "Test  ROC-AUC": test_m["roc_auc"],
        "Train PR-AUC":  train_m["pr_auc"],
        "Test  PR-AUC":  test_m["pr_auc"],
    }
    colors = [C_DARK, C_LIGHT, C_WARM, C_PALE]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(len(metrics_pairs))
    ax.barh(list(y_pos), list(metrics_pairs.values()), color=colors, height=0.55)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(list(metrics_pairs.keys()))
    ax.axvline(0.5,   color="gray",  linestyle="--", linewidth=1.5, alpha=0.7,
               label="Random (ROC floor = 0.5)")
    ax.axvline(POS_RATE, color=C_WARM, linestyle=":",  linewidth=1.5, alpha=0.7,
               label=f"Random (PR floor = {POS_RATE})")
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1)
    ax.set_title("EXP4 — Label Permutation: Memorization vs. Generalization")
    ax.legend(fontsize=11)
    for i, v in enumerate(metrics_pairs.values()):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "exp4_label_permutation")

    result = {
        "train": train_m, "test": test_m,
        "note": ("High train AUC on permuted labels = memorization capacity. "
                 "Test AUC near random floor = zero real signal extracted."),
    }
    save_json(result, "exp4_label_permutation")
    return result


def exp5_capacity_curve():
    """
    Sweep max_depth from 1 to 30 on the clean feature set (random split).
    Records train and test PR-AUC at each depth. Overfitting is visible where
    train keeps rising while test plateaus or falls. The shaded region between
    curves is the overfitting gap.
    """
    print("\n--- EXP5: Capacity Curve (max_depth 1..30) ---")
    df, feat_cols = load_clean()
    X_train, X_test, y_train, y_test = stratified_split(df, feat_cols)

    depths      = list(range(1, 31))
    train_aucs  = []
    test_aucs   = []

    for d in depths:
        pipe = make_rf_pipeline(n_estimators=100, max_depth=d)
        pipe.fit(X_train, y_train)
        tr = float(average_precision_score(
            y_train, pipe.predict_proba(X_train)[:, 1]))
        te = float(average_precision_score(
            y_test,  pipe.predict_proba(X_test)[:, 1]))
        train_aucs.append(tr)
        test_aucs.append(te)
        print(f"  depth={d:2d}  train={tr:.4f}  test={te:.4f}")

    best_depth = depths[int(np.argmax(test_aucs))]
    print(f"  Best test depth: {best_depth}")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(depths, train_aucs, color=C_MED,  linewidth=2.5, label="Train PR-AUC")
    ax.plot(depths, test_aucs,  color=C_WARM, linewidth=2.5,
            linestyle="--", label="Test PR-AUC")
    ax.axvline(best_depth, color="gray", linestyle=":", linewidth=1.5,
               label=f"Best test depth = {best_depth}")
    ax.axhline(POS_RATE, color=C_GOLD, linestyle="-.", linewidth=1.2, alpha=0.8,
               label=f"Prevalence baseline ({POS_RATE})")
    # Shade the overfitting gap
    ax.fill_between(
        depths,
        [max(tr, te) for tr, te in zip(train_aucs, test_aucs)],
        [min(tr, te) for tr, te in zip(train_aucs, test_aucs)],
        where=[tr > te for tr, te in zip(train_aucs, test_aucs)],
        alpha=0.15, color=C_WARM, label="Overfitting gap",
    )
    ax.set_xlabel("max_depth (RandomForest)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("EXP5 — Capacity Curve: Train vs Test PR-AUC")
    ax.legend(fontsize=12)
    fig.tight_layout()
    save_fig(fig, "exp5_capacity_curve")

    result = {
        "depths":       depths,
        "train_pr_auc": train_aucs,
        "test_pr_auc":  test_aucs,
        "best_depth":   best_depth,
    }
    save_json(result, "exp5_capacity_curve")
    return result


def exp6_baselines():
    """
    Three simple baselines to establish the performance floor:
      1. DummyClassifier (stratified)   — random proportional predictions
      2. DummyClassifier (most_frequent) — always predict majority class
      3. Freight-ratio threshold rule   — predict low review if freight_ratio
         exceeds the median of the training set

    All three use the Pipeline pattern; imputation on train data only.
    """
    print("\n--- EXP6: Baselines ---")
    df, feat_cols = load_clean()
    X_train, X_test, y_train, y_test = stratified_split(df, feat_cols)
    result = {}

    # ── Dummy classifiers ─────────────────────────────────────────────────────
    for tag, strategy in [
        ("dummy_stratified",    "stratified"),
        ("dummy_most_frequent", "most_frequent"),
    ]:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   DummyClassifier(strategy=strategy,
                                        random_state=RANDOM_STATE)),
        ])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred  = pipe.predict(X_test)
        result[tag] = compute_metrics(y_test, proba, pred)
        print_metrics(tag, result[tag])

    # ── Freight-ratio threshold rule ──────────────────────────────────────────
    # Rule: freight_ratio > median(freight_ratio in training set) -> 1
    # Imputer is fit on X_train only (satisfies the no-pre-split-fit constraint).
    imp         = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)   # fit on train only
    X_test_imp  = imp.transform(X_test)

    fr_idx      = feat_cols.index("freight_ratio")
    fr_train    = X_train_imp[:, fr_idx]
    fr_test     = X_test_imp[:, fr_idx]
    threshold   = float(np.median(fr_train))
    rule_pred   = (fr_test > threshold).astype(int)
    # Monotone probability proxy: normalize freight_ratio to [0, 1] via train range
    fr_min, fr_max = fr_train.min(), fr_train.max()
    rule_proba  = np.clip((fr_test - fr_min) / (fr_max - fr_min + 1e-9), 0.0, 1.0)

    result["rule_freight_ratio"] = compute_metrics(y_test, rule_proba, rule_pred)
    print_metrics("rule_freight_ratio", result["rule_freight_ratio"])

    # ── Bar chart ─────────────────────────────────────────────────────────────
    labels  = list(result.keys())
    x       = np.arange(len(labels))
    w       = 0.38
    pr_vals  = [result[k]["pr_auc"]  for k in labels]
    roc_vals = [result[k]["roc_auc"] for k in labels]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, pr_vals,  w, label="PR-AUC",  color=C_MED)
    ax.bar(x + w / 2, roc_vals, w, label="ROC-AUC", color=C_PALE)
    ax.axhline(POS_RATE, color=C_WARM, linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"PR-AUC floor (class rate {POS_RATE})")
    ax.axhline(0.5, color="gray", linestyle=":",  linewidth=1.2, alpha=0.6,
               label="ROC-AUC floor (random = 0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl.replace("_", "\n") for lbl in labels], fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("EXP6 — Baselines: DummyClassifiers and Freight-Ratio Rule")
    ax.legend(fontsize=11)
    fig.tight_layout()
    save_fig(fig, "exp6_baselines")

    save_json(result, "exp6_baselines")
    return result


def exp8_seller_score_comparison():
    """
    Compare strict vs non-strict seller historical score using time-based split.

    strict:
      seller_hist_avg_score_strict — only reviews written before the order's
      purchase timestamp are included. Truly leak-free.

    non_strict:
      seller_hist_avg_score — uses all orders placed before this one, even if
      their review had not been written yet. Subtle temporal leak.

    All other features are identical. Delta = non_strict_test - strict_test
    quantifies how much the temporal leak inflates test scores.
    """
    print("\n--- EXP8: Seller Score Comparison (time-based split) ---")

    # Load once with the non-strict column included
    df, _ = load_clean(include_nonstrict=True)

    SELLER_COLS = [NONSTRICT_COL, "seller_hist_avg_score_strict",
                   "seller_hist_missing"]
    base_cols   = [c for c in df.columns
                   if c not in set(META_COLS + [TARGET] + SELLER_COLS)]

    results = {}

    for version, extra_cols in [
        ("strict",     ["seller_hist_avg_score_strict", "seller_hist_missing"]),
        ("non_strict", [NONSTRICT_COL]),
    ]:
        feat_cols = base_cols + extra_cols
        X_train, X_test, y_train, y_test = time_split(df, feat_cols)

        pipe = make_rf_pipeline()
        pipe.fit(X_train, y_train)

        train_m = compute_metrics(y_train, pipe.predict_proba(X_train)[:, 1])
        test_m  = compute_metrics(y_test,  pipe.predict_proba(X_test)[:, 1])
        results[version] = {
            "train": train_m, "test": test_m,
            "n_features": len(feat_cols),
        }
        print_metrics(f"{version} / train", train_m)
        print_metrics(f"{version} / test",  test_m)

    diff_roc = (results["non_strict"]["test"]["roc_auc"]
                - results["strict"]["test"]["roc_auc"])
    diff_pr  = (results["non_strict"]["test"]["pr_auc"]
                - results["strict"]["test"]["pr_auc"])
    results["delta"] = {"roc_auc": float(diff_roc), "pr_auc": float(diff_pr)}
    print(f"\n  Delta (non_strict - strict):"
          f"  ROC-AUC={diff_roc:+.4f}  PR-AUC={diff_pr:+.4f}")

    # ── Side-by-side bar chart (PR-AUC | ROC-AUC) ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, ylabel in [
        (axes[0], "pr_auc",  "PR-AUC"),
        (axes[1], "roc_auc", "ROC-AUC"),
    ]:
        groups  = {
            "strict\ntrain":      results["strict"]["train"][metric],
            "strict\ntest":       results["strict"]["test"][metric],
            "non-strict\ntrain":  results["non_strict"]["train"][metric],
            "non-strict\ntest":   results["non_strict"]["test"][metric],
        }
        colors = [C_DARK, C_LIGHT, C_WARM, C_GOLD]
        bars   = ax.bar(range(4), list(groups.values()), color=colors, width=0.58)
        ax.set_xticks(range(4))
        ax.set_xticklabels(list(groups.keys()), fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
        ax.set_title(f"EXP8 — Seller Score: {ylabel}")
        for bar_obj, v in zip(bars, groups.values()):
            ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                    v + 0.012, f"{v:.3f}", ha="center", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "exp8_seller_score_comparison")

    save_json(results, "exp8_seller_score_comparison")
    return results


def exp_2x2_grid():
    """
    Full 2x2 factorial: {leaky, clean} x {random split, time split}, max_depth=12.
    Isolates the independent contribution of feature leakage vs. split strategy
    on test performance. All four cells use n_estimators=200, max_depth=12, seed=42.
    """
    print("\n--- 2x2 GRID: leakage x split strategy (max_depth=12) ---")

    cells = {}
    configs = [
        ("leaky", "random", load_leaky,  stratified_split),
        ("leaky", "time",   load_leaky,  time_split),
        ("clean", "random", load_clean,  stratified_split),
        ("clean", "time",   load_clean,  time_split),
    ]
    for feat_tag, split_tag, loader, splitter in configs:
        df, feat_cols = loader()
        X_tr, X_te, y_tr, y_te = splitter(df, feat_cols)
        pipe = make_rf_pipeline(n_estimators=200, max_depth=12)
        pipe.fit(X_tr, y_tr)
        train_m = compute_metrics(y_tr, pipe.predict_proba(X_tr)[:, 1])
        test_m  = compute_metrics(y_te, pipe.predict_proba(X_te)[:, 1])
        key = f"{feat_tag}_{split_tag}"
        cells[key] = {"train": train_m, "test": test_m, "n_features": len(feat_cols)}
        print_metrics(f"  {key} train", train_m)
        print_metrics(f"  {key} test ", test_m)

    # Print 2x2 table to stdout
    header = f"\n  {'':22s} {'Random split':>20s}  {'Time split':>20s}"
    print(header)
    for feat in ("leaky", "clean"):
        roc_r = cells[f"{feat}_random"]["test"]["roc_auc"]
        pr_r  = cells[f"{feat}_random"]["test"]["pr_auc"]
        roc_t = cells[f"{feat}_time"]["test"]["roc_auc"]
        pr_t  = cells[f"{feat}_time"]["test"]["pr_auc"]
        print(f"  {feat.upper()+' features':22s} "
              f"ROC={roc_r:.4f} PR={pr_r:.4f}  "
              f"ROC={roc_t:.4f} PR={pr_t:.4f}")

    save_json(cells, "exp_2x2_grid")
    return cells


def exp_2x2_multiseed():
    """
    Repeat the 2x2 grid (leaky/clean x random/time, max_depth=12) with 5 seeds.
    Reports mean ± std of test ROC-AUC and PR-AUC per cell to confirm that the
    leakage effect (~0.116 ROC) and split effect (~0.042 ROC) exceed model variance.
    """
    print("\n--- 2x2 MULTI-SEED (5 seeds, max_depth=12) ---")
    seeds = [42, 123, 456, 789, 1234]
    configs = [
        ("leaky", "random", load_leaky, stratified_split),
        ("leaky", "time",   load_leaky, time_split),
        ("clean", "random", load_clean, stratified_split),
        ("clean", "time",   load_clean, time_split),
    ]

    runs: dict = {f"{f}_{s}": [] for f, s, *_ in configs}

    for seed in seeds:
        for feat_tag, split_tag, loader, splitter in configs:
            df, feat_cols = loader()
            X_tr, X_te, y_tr, y_te = splitter(df, feat_cols)
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model",   RandomForestClassifier(
                    n_estimators=200, max_depth=12, class_weight="balanced",
                    random_state=seed, n_jobs=-1,
                )),
            ])
            pipe.fit(X_tr, y_tr)
            m = compute_metrics(y_te, pipe.predict_proba(X_te)[:, 1])
            runs[f"{feat_tag}_{split_tag}"].append({"seed": seed, **m})
        print(f"  seed={seed} done")

    def agg(lst, key):
        vals = [d[key] for d in lst]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {}
    for key, lst in runs.items():
        mr, sr = agg(lst, "roc_auc")
        mp, sp = agg(lst, "pr_auc")
        summary[key] = {
            "roc_auc_mean": mr, "roc_auc_std": sr,
            "pr_auc_mean":  mp, "pr_auc_std":  sp,
            "runs": lst,
        }

    # Print 2x2 summary table
    print(f"\n  {'':22s} {'Random split':^30s}  {'Time split':^30s}")
    print(f"  {'':22s} {'ROC-AUC':>14s}  {'PR-AUC':>12s}    {'ROC-AUC':>14s}  {'PR-AUC':>12s}")
    for feat in ("leaky", "clean"):
        r  = summary[f"{feat}_random"]
        t  = summary[f"{feat}_time"]
        print(f"  {feat.upper()+' features':22s} "
              f"{r['roc_auc_mean']:.4f}±{r['roc_auc_std']:.4f}  "
              f"{r['pr_auc_mean']:.4f}±{r['pr_auc_std']:.4f}    "
              f"{t['roc_auc_mean']:.4f}±{t['roc_auc_std']:.4f}  "
              f"{t['pr_auc_mean']:.4f}±{t['pr_auc_std']:.4f}")

    # Effect sizes vs pooled std
    effects = {}
    for split in ("random", "time"):
        lr = summary[f"leaky_{split}"]
        cr = summary[f"clean_{split}"]
        d_roc = lr["roc_auc_mean"] - cr["roc_auc_mean"]
        d_pr  = lr["pr_auc_mean"]  - cr["pr_auc_mean"]
        pooled_roc = max(lr["roc_auc_std"], cr["roc_auc_std"])
        pooled_pr  = max(lr["pr_auc_std"],  cr["pr_auc_std"])
        effects[f"leakage_effect_{split}"] = {
            "roc_auc_delta": d_roc, "pr_auc_delta": d_pr,
            "pooled_roc_std": pooled_roc, "pooled_pr_std": pooled_pr,
            "roc_exceeds_std": abs(d_roc) > pooled_roc,
            "pr_exceeds_std":  abs(d_pr)  > pooled_pr,
        }
    for feat in ("leaky", "clean"):
        rr = summary[f"{feat}_random"]
        rt = summary[f"{feat}_time"]
        d_roc = rr["roc_auc_mean"] - rt["roc_auc_mean"]
        d_pr  = rr["pr_auc_mean"]  - rt["pr_auc_mean"]
        pooled_roc = max(rr["roc_auc_std"], rt["roc_auc_std"])
        pooled_pr  = max(rr["pr_auc_std"],  rt["pr_auc_std"])
        effects[f"split_effect_{feat}"] = {
            "roc_auc_delta": d_roc, "pr_auc_delta": d_pr,
            "pooled_roc_std": pooled_roc, "pooled_pr_std": pooled_pr,
            "roc_exceeds_std": abs(d_roc) > pooled_roc,
            "pr_exceeds_std":  abs(d_pr)  > pooled_pr,
        }
    print("\n  Effect verification (delta vs pooled std):")
    for ek, ev in effects.items():
        verdict_roc = "EXCEEDS" if ev["roc_exceeds_std"] else "within"
        verdict_pr  = "EXCEEDS" if ev["pr_exceeds_std"]  else "within"
        print(f"  {ek}: ROC d={ev['roc_auc_delta']:+.4f} std={ev['pooled_roc_std']:.4f} [{verdict_roc}]"
              f"  PR d={ev['pr_auc_delta']:+.4f} std={ev['pooled_pr_std']:.4f} [{verdict_pr}]")

    summary["effects"] = effects
    save_json(summary, "exp_2x2_multiseed")
    return summary


def exp5_time_split():
    """
    Capacity curve (max_depth 1..30) on the clean feature set with chronological
    split. exp5 used random split; this variant reveals how the optimal depth and
    test PR-AUC change when temporal distribution shift is present.
    """
    print("\n--- EXP5 (time split): Capacity Curve, Chronological Split ---")
    df, feat_cols = load_clean()
    X_train, X_test, y_train, y_test = time_split(df, feat_cols)

    depths     = list(range(1, 31))
    train_aucs = []
    test_aucs  = []

    for d in depths:
        pipe = make_rf_pipeline(n_estimators=100, max_depth=d)
        pipe.fit(X_train, y_train)
        tr = float(average_precision_score(y_train, pipe.predict_proba(X_train)[:, 1]))
        te = float(average_precision_score(y_test,  pipe.predict_proba(X_test)[:, 1]))
        train_aucs.append(tr)
        test_aucs.append(te)
        print(f"  depth={d:2d}  train={tr:.4f}  test={te:.4f}")

    best_depth = depths[int(np.argmax(test_aucs))]
    best_test  = max(test_aucs)
    print(f"  Best test depth: {best_depth}  PR-AUC={best_test:.4f}")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(depths, train_aucs, color=C_MED,  linewidth=2.5, label="Train PR-AUC")
    ax.plot(depths, test_aucs,  color=C_WARM, linewidth=2.5,
            linestyle="--", label="Test PR-AUC")
    ax.axvline(best_depth, color="gray", linestyle=":", linewidth=1.5,
               label=f"Best test depth = {best_depth}")
    ax.axhline(POS_RATE, color=C_GOLD, linestyle="-.", linewidth=1.2, alpha=0.8,
               label=f"Prevalence baseline ({POS_RATE})")
    ax.fill_between(
        depths,
        [max(tr, te) for tr, te in zip(train_aucs, test_aucs)],
        [min(tr, te) for tr, te in zip(train_aucs, test_aucs)],
        where=[tr > te for tr, te in zip(train_aucs, test_aucs)],
        alpha=0.15, color=C_WARM, label="Overfitting gap",
    )
    ax.set_xlabel("max_depth (RandomForest)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("EXP5 (time split) — Capacity Curve: Train vs Test PR-AUC")
    ax.legend(fontsize=12)
    fig.tight_layout()
    save_fig(fig, "exp5_time_split_capacity_curve")

    result = {
        "depths": depths, "train_pr_auc": train_aucs, "test_pr_auc": test_aucs,
        "best_depth": best_depth, "best_test_pr_auc": best_test,
        "split": "time", "features": "clean",
    }
    save_json(result, "exp5_time_split")
    return result


def exp2_time_split():
    """
    Re-run exp2 (clean feature set) with chronological 80/20 split instead of
    random split. Produces a score directly comparable to exp8 (strict), which
    uses the same feature set and the same split strategy.
    """
    print("\n--- EXP2 (time split): Clean Model, Chronological Split ---")
    df, feat_cols = load_clean()
    X_train, X_test, y_train, y_test = time_split(df, feat_cols)

    pipe = make_rf_pipeline()
    pipe.fit(X_train, y_train)

    train_m = compute_metrics(y_train, pipe.predict_proba(X_train)[:, 1])
    test_m  = compute_metrics(y_test,  pipe.predict_proba(X_test)[:, 1],
                              pipe.predict(X_test))
    print_metrics("train", train_m)
    print_metrics("test",  test_m)

    result = {"train": train_m, "test": test_m, "n_features": len(feat_cols)}
    save_json(result, "exp2_time_split")
    return result


def exp1_exp2_bounded():
    """
    Re-run exp1 (leaky) and exp2 (clean) with max_depth=12 (exp5 optimal).
    Compares unbounded RF vs. depth-limited RF for both feature sets.
    Shows how depth regularization separately affects leakage inflation
    and generalisation on the clean feature set.
    """
    print("\n--- EXP1/2: Bounded vs Unbounded (max_depth=12) ---")
    configs = [
        ("leaky",  load_leaky,  None),
        ("leaky",  load_leaky,  12),
        ("clean",  load_clean,  None),
        ("clean",  load_clean,  12),
    ]
    results = {}

    for dataset, loader, max_depth in configs:
        df, feat_cols = loader()
        X_train, X_test, y_train, y_test = stratified_split(df, feat_cols)

        pipe = make_rf_pipeline(n_estimators=200, max_depth=max_depth)
        pipe.fit(X_train, y_train)

        depth_tag = "unbounded" if max_depth is None else f"depth{max_depth}"
        key       = f"{dataset}_{depth_tag}"

        train_m = compute_metrics(y_train, pipe.predict_proba(X_train)[:, 1])
        test_m  = compute_metrics(y_test,  pipe.predict_proba(X_test)[:, 1])
        results[key] = {"train": train_m, "test": test_m}
        label = f"{dataset} / {depth_tag}"
        print_metrics(f"  {label} train", train_m)
        print_metrics(f"  {label} test ", test_m)

    # ── Side-by-side comparison: PR-AUC and ROC-AUC ──────────────────────────
    ordered = ["leaky_unbounded", "leaky_depth12", "clean_unbounded", "clean_depth12"]
    labels  = ["Leaky\nunbounded", "Leaky\ndepth=12",
               "Clean\nunbounded", "Clean\ndepth=12"]
    colors  = [C_WARM, C_GOLD, C_DARK, C_LIGHT]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, ylabel in [
        (axes[0], "pr_auc",  "PR-AUC"),
        (axes[1], "roc_auc", "ROC-AUC"),
    ]:
        x = np.arange(len(ordered))
        w = 0.36
        tr_vals = [results[k]["train"][metric] for k in ordered]
        te_vals = [results[k]["test"][metric]  for k in ordered]

        b1 = ax.bar(x - w / 2, tr_vals, w, label="Train", color=colors, alpha=0.5)
        b2 = ax.bar(x + w / 2, te_vals, w, label="Test",  color=colors)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"EXP1/2 Depth Comparison — {ylabel}")
        ax.legend(fontsize=11)

        for bar_obj, v in zip(list(b1) + list(b2), tr_vals + te_vals):
            ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                    v + 0.012, f"{v:.3f}", ha="center", fontsize=9, rotation=90)

    fig.tight_layout()
    save_fig(fig, "exp1_exp2_bounded_vs_unbounded")
    save_json(results, "exp1_exp2_bounded")
    return results


def exp8_multi_seed():
    """
    Repeat exp8 (strict vs non-strict seller score, time-based split) with
    5 different RF random seeds to assess whether the observed delta is larger
    than the model's inherent variance. If delta < std, the difference is
    indistinguishable from noise.
    """
    print("\n--- EXP8: Multi-seed stability (5 seeds) ---")
    seeds = [42, 123, 456, 789, 1234]

    df, _ = load_clean(include_nonstrict=True)
    SELLER_COLS = [NONSTRICT_COL, "seller_hist_avg_score_strict", "seller_hist_missing"]
    base_cols   = [c for c in df.columns
                   if c not in set(META_COLS + [TARGET] + SELLER_COLS)]

    records: dict = {"strict": [], "non_strict": []}

    for seed in seeds:
        for version, extra_cols in [
            ("strict",     ["seller_hist_avg_score_strict", "seller_hist_missing"]),
            ("non_strict", [NONSTRICT_COL]),
        ]:
            feat_cols = base_cols + extra_cols
            X_train, X_test, y_train, y_test = time_split(df, feat_cols)

            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model",   RandomForestClassifier(
                    n_estimators=200, class_weight="balanced",
                    random_state=seed, n_jobs=-1,
                )),
            ])
            pipe.fit(X_train, y_train)
            m = compute_metrics(y_test, pipe.predict_proba(X_test)[:, 1])
            records[version].append({"seed": seed, **m})
        print(f"  seed={seed} done")

    # Aggregate
    def agg(lst, key):
        vals = [d[key] for d in lst]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {}
    for version in ("strict", "non_strict"):
        mean_roc, std_roc = agg(records[version], "roc_auc")
        mean_pr,  std_pr  = agg(records[version], "pr_auc")
        summary[version] = {
            "roc_auc_mean": mean_roc, "roc_auc_std": std_roc,
            "pr_auc_mean":  mean_pr,  "pr_auc_std":  std_pr,
            "runs": records[version],
        }
        print(f"  {version:12s}  ROC-AUC={mean_roc:.4f}±{std_roc:.4f}"
              f"  PR-AUC={mean_pr:.4f}±{std_pr:.4f}")

    delta_roc = summary["non_strict"]["roc_auc_mean"] - summary["strict"]["roc_auc_mean"]
    delta_pr  = summary["non_strict"]["pr_auc_mean"]  - summary["strict"]["pr_auc_mean"]
    pooled_std_roc = max(summary["strict"]["roc_auc_std"],
                         summary["non_strict"]["roc_auc_std"])
    pooled_std_pr  = max(summary["strict"]["pr_auc_std"],
                         summary["non_strict"]["pr_auc_std"])

    within_roc = abs(delta_roc) <= pooled_std_roc
    within_pr  = abs(delta_pr)  <= pooled_std_pr
    verdict = ("WITHIN model variance" if (within_roc and within_pr)
               else "EXCEEDS model variance in at least one metric")

    print(f"\n  Delta ROC-AUC = {delta_roc:+.4f}  (pooled std={pooled_std_roc:.4f}) "
          f"-> {'within std' if within_roc else 'exceeds std'}")
    print(f"  Delta PR-AUC  = {delta_pr:+.4f}  (pooled std={pooled_std_pr:.4f}) "
          f"-> {'within std' if within_pr else 'exceeds std'}")
    print(f"  Verdict: delta is {verdict}")

    summary["delta"] = {
        "roc_auc": float(delta_roc), "pr_auc": float(delta_pr),
        "verdict": verdict,
    }

    # ── Error-bar plot ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, metric_mean, metric_std, ylabel in [
        (axes[0], "pr_auc_mean",  "pr_auc_std",  "PR-AUC"),
        (axes[1], "roc_auc_mean", "roc_auc_std", "ROC-AUC"),
    ]:
        versions = ["strict", "non_strict"]
        means = [summary[v][metric_mean] for v in versions]
        stds  = [summary[v][metric_std]  for v in versions]
        colors_bar = [C_MED, C_WARM]

        ax.bar([0, 1], means, color=colors_bar, width=0.45, zorder=2)
        ax.errorbar([0, 1], means, yerr=stds, fmt="none",
                    color="black", capsize=8, linewidth=2, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["strict\n(seller_hist_strict)", "non-strict\n(seller_hist)"],
                           fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.set_title(f"EXP8 Multi-seed — {ylabel}\n(mean ± std, n=5 seeds)")
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.02, f"{m:.3f}±{s:.3f}",
                    ha="center", fontsize=11)

    fig.tight_layout()
    save_fig(fig, "exp8_multi_seed")
    save_json(summary, "exp8_multi_seed")
    return summary


def plot_prevalence_over_time():
    """
    Monthly low-review rate over time. Reveals whether the prevalence drop seen
    in the time-based test split is a gradual trend or a sudden structural break.
    Saves figure and monthly stats to JSON.
    """
    print("\n--- Prevalence over time ---")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "features_clean.parquet"))
    df["month"] = pd.to_datetime(df["order_purchase_timestamp"]).dt.to_period("M")
    monthly = (df.groupby("month")["is_low_review"]
                 .agg(["mean", "count"])
                 .reset_index())
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()

    # Train/test cutpoint (last 20% by time)
    df_s   = df.sort_values("order_purchase_timestamp")
    cut_ts = df_s.iloc[int(len(df_s) * 0.80)]["order_purchase_timestamp"]
    cut_dt = pd.to_datetime(cut_ts)

    overall_mean = df["is_low_review"].mean()
    train_mean   = df_s.iloc[:int(len(df_s) * 0.80)]["is_low_review"].mean()
    test_mean    = df_s.iloc[int(len(df_s) * 0.80):]["is_low_review"].mean()
    print(f"  Overall: {overall_mean:.4f}  Train: {train_mean:.4f}  Test: {test_mean:.4f}")

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    ax2.bar(monthly["month_dt"], monthly["count"], width=25,
            color=C_PALE, alpha=0.35, label="Order count")
    ax1.plot(monthly["month_dt"], monthly["mean"],
             color=C_WARM, linewidth=2.5, marker="o", markersize=5,
             label="Monthly low-review rate")
    ax1.axhline(overall_mean, color="gray", linestyle="--", linewidth=1.2,
                label=f"Overall mean ({overall_mean:.3f})")
    ax1.axvline(cut_dt, color=C_DARK, linestyle=":", linewidth=2,
                label=f"Train/test cut ({cut_dt.strftime('%Y-%m')})")

    # Rolling 3-month average
    rolling = monthly["mean"].rolling(3, center=True, min_periods=2).mean()
    ax1.plot(monthly["month_dt"], rolling,
             color=C_MED, linewidth=2, linestyle="-.", alpha=0.8,
             label="3-month rolling avg")

    ax1.set_xlabel("Month")
    ax1.set_ylabel("Low-review rate", color=C_WARM, fontsize=13)
    ax2.set_ylabel("Order count", color=C_MED, fontsize=13)
    ax1.set_title("Monthly Prevalence of Low Reviews — Distribution Shift")
    ax1.set_ylim(0, max(monthly["mean"]) * 1.35)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="upper left")

    fig.tight_layout()
    save_fig(fig, "prevalence_over_time")

    result = {
        "months":      [str(m) for m in monthly["month"]],
        "prevalence":  [round(v, 6) for v in monthly["mean"]],
        "order_count": monthly["count"].tolist(),
        "overall_mean": float(overall_mean),
        "train_mean":   float(train_mean),
        "test_mean":    float(test_mean),
        "train_test_cut": str(cut_dt.date()),
    }
    save_json(result, "prevalence_over_time")
    return result


def adim2_val_depth_selection():
    """
    Corrects the selection bias in exp5: uses a held-out validation slice of
    train to choose max_depth, never touching test during selection.

    For time splits  : last 20% of train rows (chronological) = validation.
    For random splits: stratified random 20% of train = validation.

    Returns per-cell optimal depth, val/test PR-AUC curves, and final test
    metrics (model retrained on full train with best depth).
    """
    print("\n--- ADIM 2: Validation-based depth selection ---")
    DEPTHS = list(range(1, 31))

    configs = [
        ("leaky", "random", load_leaky, stratified_split),
        ("leaky", "time",   load_leaky, time_split),
        ("clean", "random", load_clean, stratified_split),
        ("clean", "time",   load_clean, time_split),
    ]

    results = {}

    for feat_tag, split_tag, loader, splitter in configs:
        key = f"{feat_tag}_{split_tag}"
        print(f"\n  [{key}]")
        df, feat_cols = loader()
        X_tr, X_te, y_tr, y_te = splitter(df, feat_cols)

        # ── Carve out validation from train ───────────────────────────────
        n   = len(X_tr)
        cut = int(n * 0.80)
        if split_tag == "time":
            X_tr2, X_val = X_tr[:cut], X_tr[cut:]
            y_tr2, y_val = y_tr[:cut], y_tr[cut:]
        else:
            X_tr2, X_val, y_tr2, y_val = train_test_split(
                X_tr, y_tr, test_size=0.20, stratify=y_tr, random_state=RANDOM_STATE
            )

        # ── Sweep depths on train2 → validate on val ──────────────────────
        imp      = SimpleImputer(strategy="median")
        X_tr2_i  = imp.fit_transform(X_tr2)
        X_val_i  = imp.transform(X_val)
        X_te_i   = imp.transform(X_te)

        tr2_aucs, val_aucs, test_curve = [], [], []
        for d in DEPTHS:
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=d, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            )
            rf.fit(X_tr2_i, y_tr2)
            tr2_aucs.append(float(average_precision_score(
                y_tr2, rf.predict_proba(X_tr2_i)[:, 1])))
            val_aucs.append(float(average_precision_score(
                y_val, rf.predict_proba(X_val_i)[:, 1])))
            test_curve.append(float(average_precision_score(
                y_te, rf.predict_proba(X_te_i)[:, 1])))

        best_depth    = DEPTHS[int(np.argmax(val_aucs))]
        best_val_pr   = val_aucs[best_depth - 1]
        print(f"    best depth (val) = {best_depth}  val PR-AUC = {best_val_pr:.4f}")

        # ── Final model: retrain on FULL train with best depth ────────────
        pipe = make_rf_pipeline(n_estimators=200, max_depth=best_depth)
        pipe.fit(X_tr, y_tr)
        train_m = compute_metrics(y_tr, pipe.predict_proba(X_tr)[:, 1])
        test_m  = compute_metrics(y_te, pipe.predict_proba(X_te)[:, 1])
        print_metrics("    final train", train_m)
        print_metrics("    final test ", test_m)

        results[key] = {
            "best_depth":       best_depth,
            "train":            train_m,
            "test":             test_m,
            "n_features":       len(feat_cols),
            "depths":           DEPTHS,
            "train2_pr_auc":    tr2_aucs,
            "val_pr_auc":       val_aucs,
            "test_pr_auc_curve": test_curve,
        }

    # ── Comparison figure: val vs test curves for each cell ───────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (feat_tag, split_tag, *_) in zip(axes.flat, configs):
        key = f"{feat_tag}_{split_tag}"
        r   = results[key]
        bd  = r["best_depth"]
        ax.plot(DEPTHS, r["train2_pr_auc"], color=C_PALE,  linewidth=1.5,
                linestyle="--", label="Train2 PR-AUC")
        ax.plot(DEPTHS, r["val_pr_auc"],    color=C_MED,   linewidth=2.5,
                label="Val PR-AUC (selection)")
        ax.plot(DEPTHS, r["test_pr_auc_curve"], color=C_WARM, linewidth=2,
                linestyle=":", label="Test PR-AUC (reference)")
        ax.axvline(bd, color=C_DARK, linestyle=":", linewidth=1.5,
                   label=f"Best val depth = {bd}")
        floor = POS_RATE if split_tag == "random" else 0.0967
        ax.axhline(floor, color=C_GOLD, linestyle="-.", linewidth=1.1, alpha=0.8,
                   label=f"PR floor ({floor:.4f})")
        ax.set_xlabel("max_depth")
        ax.set_ylabel("PR-AUC")
        ax.set_title(f"{feat_tag.upper()} + {split_tag} split")
        ax.legend(fontsize=9)
    fig.suptitle("ADIM 2 — Val-selected depth: val vs test PR-AUC curves", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "adim2_val_vs_test_curves")

    save_json(results, "adim2_val_depth_selection")
    return results


def adim3_bootstrap_ci(n_boot=1000):
    """
    95% bootstrap CI for ROC-AUC, PR-AUC, and lift for each 2x2 cell.
    Uses val-selected depths from ADIM 2.

    Leakage effect (leaky-clean): PAIRED bootstrap — both models share the
    same test set rows, so we resample indices together.
    Split effect (random-time): INDEPENDENT bootstrap — test sets are
    different rows (different time windows vs random sample).
    All effects reported in lift units (PR-AUC / bootstrap-sample prevalence).
    """
    print("\n--- ADIM 3a: Bootstrap CI (n_boot={}) ---".format(n_boot))
    FLOOR_RAND = POS_RATE
    FLOOR_TIME = 0.0967
    rng = np.random.default_rng(RANDOM_STATE)

    a2 = json.load(open(os.path.join(RESULTS_DIR, "adim2_val_depth_selection.json")))

    configs = [
        ("leaky", "random", load_leaky, stratified_split, FLOOR_RAND),
        ("leaky", "time",   load_leaky, time_split,       FLOOR_TIME),
        ("clean", "random", load_clean, stratified_split, FLOOR_RAND),
        ("clean", "time",   load_clean, time_split,       FLOOR_TIME),
    ]

    # ── Fit models once, store test predictions ────────────────────────────
    preds = {}
    for feat_tag, split_tag, loader, splitter, floor in configs:
        key = f"{feat_tag}_{split_tag}"
        depth = a2[key]["best_depth"]
        df, feat_cols = loader()
        X_tr, X_te, y_tr, y_te = splitter(df, feat_cols)
        pipe = make_rf_pipeline(n_estimators=200, max_depth=depth)
        pipe.fit(X_tr, y_tr)
        preds[key] = {
            "proba":  pipe.predict_proba(X_te)[:, 1],
            "y_true": y_te,
            "floor":  floor,
        }
        print(f"  {key}: depth={depth}  n_test={len(y_te)}  "
              f"prev={y_te.mean():.4f}")

    def _boot_single(proba, y_true, floor, rng, n_boot):
        n = len(y_true)
        roc_b, pr_b, lift_b = [], [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            yb, pb = y_true[idx], proba[idx]
            if yb.sum() == 0 or yb.sum() == n:
                continue
            roc_b.append(float(roc_auc_score(yb, pb)))
            pr  = float(average_precision_score(yb, pb))
            pr_b.append(pr)
            lift_b.append(pr / yb.mean())   # resample prevalence as denominator
        return roc_b, pr_b, lift_b

    def _ci(vals):
        return {
            "mean":  float(np.mean(vals)),
            "ci_lo": float(np.percentile(vals, 2.5)),
            "ci_hi": float(np.percentile(vals, 97.5)),
        }

    # ── Per-cell bootstrap CI ─────────────────────────────────────────────
    cell_ci = {}
    for key, d in preds.items():
        roc_b, pr_b, lift_b = _boot_single(
            d["proba"], d["y_true"], d["floor"], rng, n_boot)
        cell_ci[key] = {
            "roc_auc": _ci(roc_b),
            "pr_auc":  _ci(pr_b),
            "lift":    _ci(lift_b),
            "floor":   d["floor"],
        }
        r, p, l = cell_ci[key]["roc_auc"], cell_ci[key]["pr_auc"], cell_ci[key]["lift"]
        print(f"  {key}: ROC [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]  "
              f"PR [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]  "
              f"Lift [{l['ci_lo']:.2f}x, {l['ci_hi']:.2f}x]")

    # ── Paired bootstrap: leakage effect (leaky-clean, same test set) ────
    leakage_ci = {}
    for split_tag in ("random", "time"):
        lk = preds[f"leaky_{split_tag}"]
        cl = preds[f"clean_{split_tag}"]
        y  = lk["y_true"]          # identical for leaky and clean
        n  = len(y)
        roc_d, pr_d, lift_d = [], [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            yb  = y[idx]
            if yb.sum() == 0 or yb.sum() == n:
                continue
            pr_l  = float(average_precision_score(yb, lk["proba"][idx]))
            pr_c  = float(average_precision_score(yb, cl["proba"][idx]))
            roc_d.append(float(roc_auc_score(yb, lk["proba"][idx])) -
                         float(roc_auc_score(yb, cl["proba"][idx])))
            pr_d.append(pr_l - pr_c)
            prev_b = yb.mean()
            lift_d.append((pr_l - pr_c) / prev_b)   # paired: same denominator
        ci_roc  = _ci(roc_d)
        ci_pr   = _ci(pr_d)
        ci_lift = _ci(lift_d)
        z_roc  = ci_roc["ci_lo"] <= 0 <= ci_roc["ci_hi"]
        z_lift = ci_lift["ci_lo"] <= 0 <= ci_lift["ci_hi"]
        leakage_ci[split_tag] = {
            "roc_auc_diff": {**ci_roc,  "includes_zero": z_roc},
            "pr_auc_diff":  {**ci_pr,   "includes_zero": ci_pr["ci_lo"] <= 0 <= ci_pr["ci_hi"]},
            "lift_diff":    {**ci_lift, "includes_zero": z_lift},
        }
        print(f"\n  Leakage ({split_tag}):  "
              f"ROC d={ci_roc['mean']:+.4f} [{ci_roc['ci_lo']:+.4f}, {ci_roc['ci_hi']:+.4f}] "
              f"{'[ZERO IN CI]' if z_roc else '[no zero]'}  "
              f"Lift d={ci_lift['mean']:+.2f}x [{ci_lift['ci_lo']:+.2f}x, {ci_lift['ci_hi']:+.2f}x] "
              f"{'[ZERO IN CI]' if z_lift else '[no zero]'}")

    # ── Independent bootstrap: split effect (random-time, same features) ─
    split_ci = {}
    for feat_tag in ("leaky", "clean"):
        rd = preds[f"{feat_tag}_random"]
        td = preds[f"{feat_tag}_time"]
        n_r, n_t = len(rd["y_true"]), len(td["y_true"])
        roc_d, pr_d, lift_d = [], [], []
        for _ in range(n_boot):
            ir = rng.integers(0, n_r, n_r)
            it = rng.integers(0, n_t, n_t)
            yr, yt = rd["y_true"][ir], td["y_true"][it]
            if yr.sum() in (0, n_r) or yt.sum() in (0, n_t):
                continue
            pr_r = float(average_precision_score(yr, rd["proba"][ir]))
            pr_t = float(average_precision_score(yt, td["proba"][it]))
            roc_d.append(float(roc_auc_score(yr, rd["proba"][ir])) -
                         float(roc_auc_score(yt, td["proba"][it])))
            pr_d.append(pr_r - pr_t)
            lift_d.append(pr_r / yr.mean() - pr_t / yt.mean())
        ci_roc  = _ci(roc_d)
        ci_pr   = _ci(pr_d)
        ci_lift = _ci(lift_d)
        z_roc  = ci_roc["ci_lo"] <= 0 <= ci_roc["ci_hi"]
        z_lift = ci_lift["ci_lo"] <= 0 <= ci_lift["ci_hi"]
        split_ci[feat_tag] = {
            "roc_auc_diff": {**ci_roc,  "includes_zero": z_roc},
            "pr_auc_diff":  {**ci_pr,   "includes_zero": ci_pr["ci_lo"] <= 0 <= ci_pr["ci_hi"]},
            "lift_diff":    {**ci_lift, "includes_zero": z_lift},
        }
        print(f"\n  Split effect ({feat_tag}):  "
              f"ROC d={ci_roc['mean']:+.4f} [{ci_roc['ci_lo']:+.4f}, {ci_roc['ci_hi']:+.4f}] "
              f"{'[ZERO IN CI]' if z_roc else '[no zero]'}  "
              f"Lift d={ci_lift['mean']:+.2f}x [{ci_lift['ci_lo']:+.2f}x, {ci_lift['ci_hi']:+.2f}x] "
              f"{'[ZERO IN CI]' if z_lift else '[no zero]'}")

    result = {
        "cell_ci":     cell_ci,
        "leakage_ci":  leakage_ci,
        "split_ci":    split_ci,
        "n_boot":      n_boot,
    }
    save_json(result, "adim3_bootstrap_ci")
    return result


def adim3_rolling_origin():
    """
    Rolling-origin CV for clean+time setup (ADIM 2 val-selected depth=6).
    Tests: last 2, 3, 4 months. Train = everything before cutpoint.
    Reports ROC-AUC, PR-AUC, lift (using each window's own test prevalence).
    """
    print("\n--- ADIM 3b: Rolling-origin validation ---")
    a2    = json.load(open(os.path.join(RESULTS_DIR, "adim2_val_depth_selection.json")))
    depth = a2["clean_time"]["best_depth"]
    print(f"  Using val-selected depth = {depth}")

    df, feat_cols = load_clean()
    df_s = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    ts   = pd.to_datetime(df_s["order_purchase_timestamp"])
    max_ts = ts.max()

    results = {}
    for n_months in [2, 3, 4]:
        cut_ts     = max_ts - pd.DateOffset(months=n_months)
        train_mask = ts <= cut_ts
        test_mask  = ~train_mask

        X_tr = df_s.loc[train_mask, feat_cols].values
        X_te = df_s.loc[test_mask,  feat_cols].values
        y_tr = df_s.loc[train_mask, TARGET].values
        y_te = df_s.loc[test_mask,  TARGET].values

        pipe = make_rf_pipeline(n_estimators=200, max_depth=depth)
        pipe.fit(X_tr, y_tr)

        train_m = compute_metrics(y_tr, pipe.predict_proba(X_tr)[:, 1])
        test_m  = compute_metrics(y_te, pipe.predict_proba(X_te)[:, 1])
        test_prev = float(y_te.mean())
        lift      = test_m["pr_auc"] / test_prev

        tag = f"last_{n_months}mo"
        results[tag] = {
            "n_train": int(train_mask.sum()),
            "n_test":  int(test_mask.sum()),
            "train_prevalence": float(y_tr.mean()),
            "test_prevalence":  test_prev,
            "cutpoint": str(cut_ts.date()),
            "train": train_m,
            "test":  test_m,
            "test_pr_lift": float(lift),
        }
        print(f"  last {n_months} months: n_train={train_mask.sum():5d} n_test={test_mask.sum():4d} "
              f"prev_train={y_tr.mean():.4f} prev_test={test_prev:.4f}  "
              f"ROC={test_m['roc_auc']:.4f}  PR={test_m['pr_auc']:.4f}  "
              f"lift={lift:.2f}x")

    save_json(results, "adim3_rolling_origin")
    return results


def exp4_memorization():
    """
    Memorization probe: 2 × 2 factorial (depth × label_type), clean + time split.

    Configs
    -------
    a) Unbounded RF (max_depth=None): maximum memorization capacity.
    b) Val-selected RF (max_depth=6): the depth used in all other experiments.

    For each config, train on both REAL and PERMUTED labels.
    Key contrast:
      - Real labels  : train AUC >> test AUC  → overfit, not necessarily memorize
      - Permuted labels: train AUC >> 0.5 → model HAS capacity to memorize random noise
                         test AUC ≈ 0.5  → no real signal extracted

    NOTE: exp4 intentionally uses a second depth (unbounded) that differs from
    all other experiments.  The goal is NOT to measure predictive performance but
    to measure memorization capacity, which requires an unconstrained model.
    """
    print("\n--- EXP4: Memorization probe (unbounded vs depth=6, clean+time) ---")
    FLOOR_TIME = 0.0967

    df, feat_cols = load_clean()
    df_s  = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    cut   = int(len(df_s) * 0.80)
    X_all = df_s[feat_cols].values
    y_real = df_s[TARGET].values
    y_perm = np.random.default_rng(RANDOM_STATE).permutation(y_real)

    X_tr, X_te = X_all[:cut], X_all[cut:]

    configs = [
        ("unbounded", None),
        ("depth6",    6),
    ]
    label_sets = [
        ("real",      y_real[:cut], y_real[cut:]),
        ("permuted",  y_perm[:cut], y_perm[cut:]),
    ]

    results = {}
    for depth_tag, max_depth in configs:
        results[depth_tag] = {}
        for label_tag, y_tr, y_te in label_sets:
            pipe = make_rf_pipeline(n_estimators=200, max_depth=max_depth)
            pipe.fit(X_tr, y_tr)
            train_m = compute_metrics(y_tr, pipe.predict_proba(X_tr)[:, 1])
            test_m  = compute_metrics(y_te, pipe.predict_proba(X_te)[:, 1])
            results[depth_tag][label_tag] = {"train": train_m, "test": test_m}
            print(f"  [{depth_tag}] {label_tag:8s}  "
                  f"train ROC={train_m['roc_auc']:.4f} PR={train_m['pr_auc']:.4f}  "
                  f"test  ROC={test_m['roc_auc']:.4f} PR={test_m['pr_auc']:.4f}")

    # ── Grouped bar chart ─────────────────────────────────────────────────────
    # 2 subplots (ROC-AUC, PR-AUC), 2 groups × 4 bars each
    group_labels = ["Unbounded RF\n(max_depth=None)", "Depth-6 RF\n(val-selected)"]
    bar_labels   = ["Real — train", "Real — test", "Permuted — train", "Permuted — test"]
    bar_colors   = [C_DARK, C_LIGHT, C_WARM, C_GOLD]
    bar_hatch    = [None, None, "///", "///"]

    def _vals(metric):
        out = []
        for depth_tag in ("unbounded", "depth6"):
            for label_tag, split in [("real", "train"), ("real", "test"),
                                      ("permuted", "train"), ("permuted", "test")]:
                out.append(results[depth_tag][label_tag][split][metric])
        return out

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    x      = np.arange(len(group_labels))
    n_bars = len(bar_labels)
    w      = 0.18

    for ax, metric, ylabel, floor, floor_label in [
        (axes[0], "roc_auc", "ROC-AUC", 0.5,        "Random floor (0.5)"),
        (axes[1], "pr_auc",  "PR-AUC",  FLOOR_TIME, f"PR floor ({FLOOR_TIME})"),
    ]:
        vals = _vals(metric)   # 8 values: group0×4 bars + group1×4 bars
        for bi, (label, color, hatch) in enumerate(zip(bar_labels, bar_colors, bar_hatch)):
            offsets = x + (bi - (n_bars - 1) / 2) * w
            grp_vals = [vals[0 * n_bars + bi], vals[1 * n_bars + bi]]
            bars = ax.bar(offsets, grp_vals, w,
                          label=label, color=color,
                          hatch=hatch, edgecolor="white" if hatch else color)
            for bar_obj, v in zip(bars, grp_vals):
                ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                        v + 0.008, f"{v:.3f}", ha="center", fontsize=8, rotation=90)

        ax.axhline(floor, color="gray", linestyle="--", linewidth=1.4,
                   alpha=0.7, label=floor_label)
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.12)
        ax.set_title(f"EXP4 — Memorization probe: {ylabel}")
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "EXP4 — Memorization Capacity\n"
        "Unbounded RF memorizes permuted labels (high train); depth-6 RF cannot.\n"
        "Both models fail on permuted test — confirming zero real signal in noise.",
        fontsize=12,
    )
    fig.tight_layout()
    save_fig(fig, "exp4_memorization")

    results["note"] = (
        "exp4 intentionally includes an unbounded RF (max_depth=None) that differs "
        "from all other experiments. Purpose: measure memorization CAPACITY, not "
        "predictive performance. The depth-6 row shows that our production model "
        "has limited capacity to memorize noise."
    )
    save_json(results, "exp4_memorization")
    return results


# kept for backward compatibility with cached JSON; not used in summary
def exp4_time_val():
    return json.load(open(os.path.join(RESULTS_DIR, "exp4_time_val.json")))
    return result


def exp6_time():
    """
    Baseline classifiers evaluated on the TIME-BASED split test set.
    Provides the correct PR-AUC floor (test prevalence=0.0967) for comparison
    with all ADIM 2+ experiments that use time-based split.
    """
    print("\n--- EXP6 (time split): Baselines ---")
    FLOOR_TIME = 0.0967

    df, feat_cols = load_clean()
    X_tr, X_te, y_tr, y_te = time_split(df, feat_cols)
    result = {}

    for tag, strategy in [
        ("dummy_stratified",    "stratified"),
        ("dummy_most_frequent", "most_frequent"),
    ]:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   DummyClassifier(strategy=strategy, random_state=RANDOM_STATE)),
        ])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred  = pipe.predict(X_te)
        result[tag] = compute_metrics(y_te, proba, pred)
        print_metrics(tag, result[tag])

    imp        = SimpleImputer(strategy="median")
    X_tr_i     = imp.fit_transform(X_tr)
    X_te_i     = imp.transform(X_te)
    fr_idx     = feat_cols.index("freight_ratio")
    fr_train   = X_tr_i[:, fr_idx]
    fr_test    = X_te_i[:, fr_idx]
    threshold  = float(np.median(fr_train))
    rule_pred  = (fr_test > threshold).astype(int)
    fr_min, fr_max = fr_train.min(), fr_train.max()
    rule_proba = np.clip((fr_test - fr_min) / (fr_max - fr_min + 1e-9), 0.0, 1.0)
    result["rule_freight_ratio"] = compute_metrics(y_te, rule_proba, rule_pred)
    print_metrics("rule_freight_ratio", result["rule_freight_ratio"])

    result["test_prevalence"] = float(y_te.mean())
    save_json(result, "exp6_time")
    return result


def exp8_val_selected():
    """
    Seller score comparison (strict vs non-strict) with val-selected max_depth.
    Strict (37 features, same as clean_time): uses depth=6 from ADIM 2.
    Non-strict (36 features): sweeps validation to find its own optimal depth.
    Both use time-based split.
    """
    print("\n--- EXP8 (val-selected depth): Seller Score Comparison ---")
    FLOOR_TIME  = 0.0967
    STRICT_DEPTH = 6     # from ADIM 2 clean_time
    DEPTHS       = list(range(1, 16))

    df, _ = load_clean(include_nonstrict=True)
    SELLER_COLS = [NONSTRICT_COL, "seller_hist_avg_score_strict", "seller_hist_missing"]
    base_cols   = [c for c in df.columns
                   if c not in set(META_COLS + [TARGET] + SELLER_COLS)]

    results = {}

    for version, extra_cols, depth_override in [
        ("strict",     ["seller_hist_avg_score_strict", "seller_hist_missing"], STRICT_DEPTH),
        ("non_strict", [NONSTRICT_COL],                                         None),
    ]:
        feat_cols = base_cols + extra_cols
        X_tr, X_te, y_tr, y_te = time_split(df, feat_cols)

        if depth_override is not None:
            best_depth = depth_override
            print(f"  {version}: using ADIM2 depth={best_depth}")
        else:
            # Val sweep for non_strict
            n   = len(X_tr)
            cut = int(n * 0.80)
            X_tr2, X_val = X_tr[:cut], X_tr[cut:]
            y_tr2, y_val = y_tr[:cut], y_tr[cut:]
            imp    = SimpleImputer(strategy="median")
            X_tr2i = imp.fit_transform(X_tr2)
            X_vali = imp.transform(X_val)
            val_scores = []
            for d in DEPTHS:
                rf = RandomForestClassifier(
                    n_estimators=100, max_depth=d, class_weight="balanced",
                    random_state=RANDOM_STATE, n_jobs=-1,
                )
                rf.fit(X_tr2i, y_tr2)
                val_scores.append(float(
                    average_precision_score(y_val, rf.predict_proba(X_vali)[:, 1])
                ))
            best_depth = DEPTHS[int(np.argmax(val_scores))]
            print(f"  {version}: val-selected depth={best_depth}  "
                  f"val_pr={max(val_scores):.4f}")

        pipe = make_rf_pipeline(n_estimators=200, max_depth=best_depth)
        pipe.fit(X_tr, y_tr)
        train_m = compute_metrics(y_tr, pipe.predict_proba(X_tr)[:, 1])
        test_m  = compute_metrics(y_te, pipe.predict_proba(X_te)[:, 1])
        results[version] = {
            "train": train_m, "test": test_m,
            "best_depth": best_depth,
            "n_features": len(feat_cols),
        }
        print_metrics(f"  {version} train", train_m)
        print_metrics(f"  {version} test ", test_m)

    diff_roc = (results["non_strict"]["test"]["roc_auc"]
                - results["strict"]["test"]["roc_auc"])
    diff_pr  = (results["non_strict"]["test"]["pr_auc"]
                - results["strict"]["test"]["pr_auc"])
    results["delta"] = {"roc_auc": float(diff_roc), "pr_auc": float(diff_pr)}
    print(f"\n  Delta (non_strict - strict): ROC={diff_roc:+.4f}  PR={diff_pr:+.4f}")

    save_json(results, "exp8_val_selected")
    return results


def adim4_permutation_importance(n_repeats=10):
    """
    Compare RF's default impurity-based feature importance against
    permutation importance computed on the TEST set.
    Run for leaky+time and clean+time (val-selected depths from ADIM 2).
    Confirms that delay_days dominates in the leaky model under both methods.
    """
    from sklearn.inspection import permutation_importance as perm_imp_fn

    print("\n--- ADIM 4: Permutation importance (test set) ---")
    a2 = json.load(open(os.path.join(RESULTS_DIR, "adim2_val_depth_selection.json")))

    configs = [
        ("leaky", "time", load_leaky, time_split),
        ("clean", "time", load_clean, time_split),
    ]
    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    for row_idx, (feat_tag, split_tag, loader, splitter) in enumerate(configs):
        key   = f"{feat_tag}_{split_tag}"
        depth = a2[key]["best_depth"]
        df, feat_cols = loader()
        X_tr, X_te, y_tr, y_te = splitter(df, feat_cols)

        imp    = SimpleImputer(strategy="median")
        X_tr_i = imp.fit_transform(X_tr)
        X_te_i = imp.transform(X_te)

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=depth, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X_tr_i, y_tr)

        # Impurity importance
        imp_vals = rf.feature_importances_
        # Permutation importance (PR-AUC, test set)
        pi = perm_imp_fn(
            rf, X_te_i, y_te, n_repeats=n_repeats,
            scoring="average_precision", random_state=RANDOM_STATE, n_jobs=-1,
        )

        top_imp  = np.argsort(imp_vals)[::-1][:20]
        top_perm = np.argsort(pi.importances_mean)[::-1][:20]

        for col_idx, (top_idx, vals, errs, xlabel, color, suffix) in enumerate([
            (top_imp,  imp_vals,            None,                     "Mean decrease in impurity", C_MED,  "impurity"),
            (top_perm, pi.importances_mean, pi.importances_std, "PR-AUC decrease (test)",   C_WARM, "permutation"),
        ]):
            ax     = axes[row_idx, col_idx]
            names  = [_short_name(feat_cols[i]) for i in top_idx]
            ypos   = list(range(len(names)))
            ax.barh(ypos[::-1], vals[top_idx], color=color, height=0.65)
            if errs is not None:
                ax.errorbar(
                    vals[top_idx], ypos[::-1],
                    xerr=errs[top_idx], fmt="none",
                    color="black", capsize=4, linewidth=1.2,
                )
            ax.set_yticks(ypos)
            ax.set_yticklabels(reversed(names), fontsize=9)
            ax.set_xlabel(xlabel)
            ax.set_title(f"{feat_tag.upper()} + {split_tag} — {suffix}")

        results[key] = {
            "impurity_top20": [
                {"feature": feat_cols[i], "importance": float(imp_vals[i])}
                for i in top_imp
            ],
            "permutation_top20": [
                {"feature": feat_cols[i],
                 "importance": float(pi.importances_mean[i]),
                 "std": float(pi.importances_std[i])}
                for i in top_perm
            ],
        }
        top_imp_name  = feat_cols[top_imp[0]]
        top_perm_name = feat_cols[top_perm[0]]
        print(f"  {key}: top-impurity={top_imp_name}  top-permutation={top_perm_name}")
        print(f"    Perm rank of delay_days: "
              f"{list(np.argsort(pi.importances_mean)[::-1]).index(feat_cols.index('delay_days'))+1 if 'delay_days' in feat_cols else 'N/A'}")

    fig.suptitle("ADIM 4 — Impurity vs Permutation Importance (test set, time split)", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "adim4_permutation_importance")
    save_json(results, "adim4_permutation_importance")
    return results


def adim5_hgbt_2x2():
    """
    Repeat the full 2x2 grid with HistGradientBoostingClassifier.
    Uses same validation-based depth selection as ADIM 2.
    HGBT handles missing values natively — no imputer needed.
    Placed side-by-side with RF to show leakage finding is model-agnostic.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    print("\n--- ADIM 5: HGBT 2x2 grid ---")
    FLOOR_RAND = POS_RATE
    FLOOR_TIME = 0.0967
    DEPTHS     = list(range(2, 9))      # HGBT max_depth

    configs = [
        ("leaky", "random", load_leaky, stratified_split, FLOOR_RAND),
        ("leaky", "time",   load_leaky, time_split,       FLOOR_TIME),
        ("clean", "random", load_clean, stratified_split, FLOOR_RAND),
        ("clean", "time",   load_clean, time_split,       FLOOR_TIME),
    ]
    results = {}

    for feat_tag, split_tag, loader, splitter, floor in configs:
        key = f"{feat_tag}_{split_tag}"
        df, feat_cols = loader()
        X_tr, X_te, y_tr, y_te = splitter(df, feat_cols)

        n   = len(X_tr)
        cut = int(n * 0.80)
        if split_tag == "time":
            X_tr2, X_val = X_tr[:cut], X_tr[cut:]
            y_tr2, y_val = y_tr[:cut], y_tr[cut:]
        else:
            X_tr2, X_val, y_tr2, y_val = train_test_split(
                X_tr, y_tr, test_size=0.20, stratify=y_tr, random_state=RANDOM_STATE,
            )

        # Val sweep over max_depth
        val_scores = []
        for d in DEPTHS:
            hgbt = HistGradientBoostingClassifier(
                max_depth=d, max_iter=200, learning_rate=0.05,
                class_weight="balanced", random_state=RANDOM_STATE,
            )
            hgbt.fit(X_tr2, y_tr2)
            val_scores.append(float(
                average_precision_score(y_val, hgbt.predict_proba(X_val)[:, 1])
            ))

        best_depth = DEPTHS[int(np.argmax(val_scores))]
        best_val   = max(val_scores)

        # Final model on full train
        hgbt_final = HistGradientBoostingClassifier(
            max_depth=best_depth, max_iter=200, learning_rate=0.05,
            class_weight="balanced", random_state=RANDOM_STATE,
        )
        hgbt_final.fit(X_tr, y_tr)

        train_m = compute_metrics(y_tr, hgbt_final.predict_proba(X_tr)[:, 1])
        test_m  = compute_metrics(y_te, hgbt_final.predict_proba(X_te)[:, 1])
        lift    = test_m["pr_auc"] / floor

        results[key] = {
            "best_depth": best_depth,
            "best_val_pr_auc": best_val,
            "train": train_m, "test": test_m,
            "floor": floor, "lift": float(lift),
            "n_features": len(feat_cols),
        }
        print(f"  {key}: depth={best_depth}  "
              f"ROC={test_m['roc_auc']:.4f}  PR={test_m['pr_auc']:.4f}  "
              f"lift={lift:.2f}x")

    save_json(results, "adim5_hgbt_2x2")
    return results


def adim6_business_lift():
    """
    Precision-at-k for the clean+time model (ADIM 2 val-selected depth).
    Sorts test orders by predicted probability (descending) and computes
    the actual low-review rate in the top 5%, 10%, 20% slices.
    Lift = slice rate / test prevalence (0.0967).
    """
    print("\n--- ADIM 6: Business lift (precision-at-k) ---")
    FLOOR_TIME = 0.0967
    a2    = json.load(open(os.path.join(RESULTS_DIR, "adim2_val_depth_selection.json")))
    depth = a2["clean_time"]["best_depth"]

    df, feat_cols = load_clean()
    X_tr, X_te, y_tr, y_te = time_split(df, feat_cols)

    pipe = make_rf_pipeline(n_estimators=200, max_depth=depth)
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]

    test_prev = float(y_te.mean())
    n         = len(y_te)
    sorted_y  = y_te[np.argsort(proba)[::-1]]

    results = {"test_prevalence": test_prev, "n_test": n, "model_depth": depth}
    rows = {}
    for pct in [5, 10, 20]:
        k    = max(1, int(n * pct / 100))
        rate = float(sorted_y[:k].mean())
        lift = rate / test_prev
        rows[f"top_{pct}pct"] = {"n": k, "rate": rate, "lift": float(lift)}
        print(f"  Top {pct:2d}%: n={k:5d}  rate={rate:.4f}  "
              f"lift={lift:.2f}x  (baseline={test_prev:.4f})")
    results["slices"] = rows

    # Bar chart
    pcts   = [5, 10, 20, 100]
    labels = ["Top 5%", "Top 10%", "Top 20%", "Overall\n(baseline)"]
    rates  = [rows["top_5pct"]["rate"], rows["top_10pct"]["rate"],
              rows["top_20pct"]["rate"], test_prev]
    lifts  = [rows["top_5pct"]["lift"], rows["top_10pct"]["lift"],
              rows["top_20pct"]["lift"], 1.0]
    colors = [C_DARK, C_MED, C_LIGHT, C_PALE]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    bars = ax.bar(labels, rates, color=colors, width=0.5)
    ax.axhline(test_prev, color=C_WARM, linestyle="--", linewidth=1.5,
               label=f"Baseline prevalence ({test_prev:.4f})")
    ax.set_ylabel("Low-review rate")
    ax.set_title("ADIM 6 — Precision-at-k: Low-review rate by score slice")
    ax.legend(fontsize=11)
    for bar_obj, v in zip(bars, rates):
        ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                v + 0.002, f"{v:.3f}", ha="center", fontsize=11)

    ax2 = axes[1]
    bars2 = ax2.bar(labels, lifts, color=colors, width=0.5)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, label="Lift = 1 (random)")
    ax2.set_ylabel("Lift (= rate / baseline)")
    ax2.set_title("ADIM 6 — Lift by score slice")
    ax2.legend(fontsize=11)
    for bar_obj, v in zip(bars2, lifts):
        ax2.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                 v + 0.02, f"{v:.2f}x", ha="center", fontsize=11)

    fig.tight_layout()
    save_fig(fig, "adim6_business_lift")
    save_json(results, "adim6_business_lift")
    return results


def generate_summary_md(all_results):
    """
    Write results/summary.md. PR-AUC values include lift = PR-AUC / test prevalence.
    Floor: random split = 0.1280, time-based split = 0.0967.
    """
    FLOOR_RAND = POS_RATE   # 0.1280
    FLOOR_TIME = 0.0967

    def _f(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    def _fp(v, floor):
        """PR-AUC with inline lift."""
        if not isinstance(v, (int, float)):
            return "—"
        return f"{v:.4f} ({v/floor:.2f}x)"

    def row(name, train_roc, test_roc, train_pr, test_pr, floor, note=""):
        cells = [f"**{name}**", _f(train_roc), _f(test_roc),
                 _f(train_pr), _fp(test_pr, floor)]
        if note:
            cells.append(note)
        return "| " + " | ".join(cells) + " |"

    def _get(r, split, metric):
        return r.get(split, {}).get(metric)

    HDR = ("| Experiment | Train ROC-AUC | Test ROC-AUC "
           "| Train PR-AUC | Test PR-AUC (lift) |")
    SEP = "|---|---|---|---|---|"

    lines = [
        "# Experiment Results Summary",
        "",
        "## ADIM 1 — Test dilimi prevalansı ve doğru PR-AUC tabanları",
        "",
        "| Split stratejisi | Train prevalans | Test prevalans | Fark "
        "| PR-AUC tabani (= test prevalansi) |",
        "|---|---|---|---|---|",
        "| Random (stratified) | 0.1280 | 0.1280 | 0.0000 | **0.1280** |",
        "| Zaman bazli | 0.1358 | 0.0967 | -0.0390 | **0.0967** |",
        "",
        "> Lift = PR-AUC / test prevalansi. Iki split stratejisinin PR-AUC degerleri "
        "dogrudan karsilastirilamaz; lift karsilastirilab ilir.",
        "",
        "---",
        "",
        "## Temel deneyler (random split)",
        "",
        HDR, SEP,
        row("Baseline — random split", None, None, FLOOR_RAND, FLOOR_RAND, FLOOR_RAND),
    ]

    bd = all_results.get("exp1_exp2_bounded", {})
    for label, key, note in [
        ("EXP1 Leaky, unbounded", "leaky_unbounded", "random split"),
        ("EXP1 Leaky, depth=12",  "leaky_depth12",   "random split"),
        ("EXP2 Clean, unbounded", "clean_unbounded",  "random split"),
        ("EXP2 Clean, depth=12",  "clean_depth12",    "random split"),
    ]:
        r = bd.get(key, {})
        lines.append(row(label,
                         _get(r, "train", "roc_auc"), _get(r, "test", "roc_auc"),
                         _get(r, "train", "pr_auc"),  _get(r, "test", "pr_auc"),
                         FLOOR_RAND, note))

    # exp4: dual-config memorization probe
    r4m = all_results.get("exp4_memorization", {})
    if r4m:
        lines += [
            "",
            "### EXP4 — Memorization probe (clean + time split)",
            "",
            "> **Not:** exp4 kasitli olarak diger deneylerden farkli bir derinlik "
            "kullaniyor (unbounded RF). Amac model performansi degil, ezber "
            "kapasitesini olcmek. depth=6 satiri, uretim modelimizin gurultuyu "
            "ezberleyemedigini gosteriyor.",
            "",
            "| Konfigurasyon | Etiket | Train ROC | Test ROC | Train PR | Test PR (lift) |",
            "|---|---|---|---|---|---|",
        ]
        for depth_tag, depth_label in [
            ("unbounded", "Unbounded RF (max_depth=None)"),
            ("depth6",    "Depth-6 RF (val-selected)"),
        ]:
            for label_tag, label_str in [("real", "Gercek"), ("permuted", "Karistirilmis")]:
                r = r4m.get(depth_tag, {}).get(label_tag, {})
                lines.append(
                    f"| {depth_label} | {label_str} "
                    f"| {_f(_get(r,'train','roc_auc'))} "
                    f"| {_f(_get(r,'test','roc_auc'))} "
                    f"| {_f(_get(r,'train','pr_auc'))} "
                    f"| {_fp(_get(r,'test','pr_auc'), FLOOR_TIME)} |"
                )
        lines.append("")

    # exp5: kendi basliginda
    r5t  = all_results.get("exp5_time_split", {})
    ra2  = all_results.get("adim2_val_depth_selection", {})
    ra2_ct = ra2.get("clean_time", {}) if ra2 else {}
    lines += ["", "### EXP5 — Kapasite egrisi (clean + time split)", ""]
    lines += [
        "| Kurulum | Depth secimi | max_depth | Test PR-AUC (lift) |",
        "|---|---|---|---|",
    ]
    if r5t:
        lines.append(
            f"| Clean + time, test-selected | test seti | {r5t.get('best_depth')} "
            f"| {_fp(r5t.get('best_test_pr_auc'), FLOOR_TIME)} |"
        )
    if ra2_ct:
        lines.append(
            f"| Clean + time, val-selected | validation | {ra2_ct.get('best_depth')} "
            f"| {_fp(_get(ra2_ct,'test','pr_auc'), FLOOR_TIME)} |"
        )

    # exp6: kendi basliginda
    r6t = all_results.get("exp6_time", {})
    r6  = all_results.get("exp6", {})
    floor6 = FLOOR_TIME if r6t else FLOOR_RAND
    split6 = "time split" if r6t else "random split"
    src6   = r6t if r6t else r6
    lines += ["", "### EXP6 — Baseline'lar (" + split6 + ")", ""]
    lines += [
        "| Baseline | Test ROC-AUC | Test PR-AUC (lift) |",
        "|---|---|---|",
    ]
    for key, label in [
        ("dummy_stratified",    "Dummy stratified"),
        ("dummy_most_frequent", "Dummy most-frequent"),
        ("rule_freight_ratio",  "Freight-ratio rule"),
    ]:
        m = src6.get(key, {})
        lines.append(
            f"| {label} | {_f(m.get('roc_auc'))} "
            f"| {_fp(m.get('pr_auc'), floor6)} |"
        )

    # ── 2x2 grid (single seed, max_depth=12) ─────────────────────────────────
    r22 = all_results.get("exp_2x2_grid", {})
    if r22:
        lines += [
            "",
            "## 2x2: Feature leakage x Split strategy (max_depth=12, single seed)",
            "",
            "| Features \\ Split | Random — Test ROC-AUC | Random — Test PR-AUC (lift)"
            " | Time — Test ROC-AUC | Time — Test PR-AUC (lift) |",
            "|---|---|---|---|---|",
        ]
        for feat, label in [("leaky", "Leaky features"), ("clean", "Clean features")]:
            rr = r22.get(f"{feat}_random", {}).get("test", {})
            rt = r22.get(f"{feat}_time",   {}).get("test", {})
            lines.append(
                f"| **{label}** "
                f"| {_f(rr.get('roc_auc'))} | {_fp(rr.get('pr_auc'), FLOOR_RAND)} "
                f"| {_f(rt.get('roc_auc'))} | {_fp(rt.get('pr_auc'), FLOOR_TIME)} |"
            )

    # ── 2x2 multi-seed (5 seeds) ──────────────────────────────────────────────
    r22m = all_results.get("exp_2x2_multiseed", {})
    if r22m:
        lines += [
            "",
            "## 2x2 Multi-seed (5 seeds, max_depth=12): mean +/- std",
            "",
            "| Features \\ Split | Random — ROC-AUC | Random — PR-AUC (lift)"
            " | Time — ROC-AUC | Time — PR-AUC (lift) |",
            "|---|---|---|---|---|",
        ]
        def ms_roc(d): return f"{d['roc_auc_mean']:.4f} +/-{d['roc_auc_std']:.4f}" if d else "—"
        def ms_pr(d, floor):
            if not d:
                return "—"
            m, s = d["pr_auc_mean"], d["pr_auc_std"]
            return f"{m:.4f} +/-{s:.4f} ({m/floor:.2f}x)"
        for feat, label in [("leaky", "Leaky features"), ("clean", "Clean features")]:
            rr = r22m.get(f"{feat}_random", {})
            rt = r22m.get(f"{feat}_time",   {})
            lines.append(
                f"| **{label}** "
                f"| {ms_roc(rr)} | {ms_pr(rr, FLOOR_RAND)} "
                f"| {ms_roc(rt)} | {ms_pr(rt, FLOOR_TIME)} |"
            )
    # ── ADIM 2: val-selected depth ────────────────────────────────────────────
    ra2 = all_results.get("adim2_val_depth_selection", {})
    if ra2:
        lines += [
            "",
            "## ADIM 2 — Validation-bazli max_depth secimi",
            "",
            "> Onceki tablolarda max_depth=12, exp5'in test setine bakarak sectigi"
            " bir deger. ADIM 2'de train'in son %%20'si validation olarak ayrilir,"
            " derinlik oradan secilir, test'e sadece bir kez dokunulur.",
            "",
            "| Hucre | Val'den sec. depth | Final Train ROC | Final Train PR"
            " | Final Test ROC | Final Test PR (lift) |",
            "|---|---|---|---|---|---|",
        ]
        for feat, split, floor in [
            ("leaky", "random", FLOOR_RAND),
            ("leaky", "time",   FLOOR_TIME),
            ("clean", "random", FLOOR_RAND),
            ("clean", "time",   FLOOR_TIME),
        ]:
            key = f"{feat}_{split}"
            r   = ra2.get(key, {})
            bd_ = r.get("best_depth", "—")
            lines.append(
                f"| **{feat.upper()} + {split}** | {bd_} "
                f"| {_f(_get(r,'train','roc_auc'))} "
                f"| {_f(_get(r,'train','pr_auc'))} "
                f"| {_f(_get(r,'test','roc_auc'))} "
                f"| {_fp(_get(r,'test','pr_auc'), floor)} |"
            )

    # ── EXP5 time split capacity curve ───────────────────────────────────────
    r5t = all_results.get("exp5_time_split", {})
    if r5t:
        bd5t    = r5t.get("best_depth")
        te5t_pr = r5t.get("best_test_pr_auc")
        idx5t   = r5t["depths"].index(bd5t) if bd5t in r5t.get("depths", []) else -1
        tr5t    = r5t["train_pr_auc"][idx5t] if idx5t >= 0 else None
        lines += [
            "",
            "### EXP5 re-run: Capacity curve (clean + time split)",
            "",
            "| Setup | Best depth (test) | Best depth (val) | Test PR-AUC (lift) |",
            "|---|---|---|---|",
            f"| EXP5 original (clean + random, test-selected) | 12 | — | 0.2322 (1.81x) |",
            f"| EXP5 time split (clean + time, test-selected) | {bd5t} | — "
            f"| {_fp(te5t_pr, FLOOR_TIME)} |",
        ]
        ra2_ct = ra2.get("clean_time", {}) if ra2 else {}
        if ra2_ct:
            lines.append(
                f"| EXP5 time split (clean + time, VAL-selected)  "
                f"| — | {ra2_ct.get('best_depth','—')} "
                f"| {_fp(_get(ra2_ct,'test','pr_auc'), FLOOR_TIME)} |"
            )

    # ── EXP8 seller score comparison ─────────────────────────────────────────
    lines += ["", "## EXP8 — Seller score (strict vs non-strict, time split, val-selected depth)", ""]
    # prefer val-selected re-run; fall back to original
    r8vs = all_results.get("exp8_val_selected", {})
    r8   = all_results.get("exp8", {})
    src8 = r8vs if r8vs else r8
    r8m  = all_results.get("exp8_multi_seed", {})
    lines += [HDR, SEP]
    for v, label in [("strict",     "EXP8 Strict"),
                     ("non_strict", "EXP8 Non-strict")]:
        sub  = src8.get(v, {})
        bd_  = sub.get("best_depth", "?") if r8vs else "unbounded"
        note = f"time split, depth={bd_}"
        lines.append(row(label,
                         _get(sub, "train", "roc_auc"), _get(sub, "test",  "roc_auc"),
                         _get(sub, "train", "pr_auc"),  _get(sub, "test",  "pr_auc"),
                         FLOOR_TIME, note))
    delta = src8.get("delta", {})
    if delta:
        lines.append(
            f"| **EXP8 Delta (non-strict - strict)** | — | "
            f"{delta.get('roc_auc', 0):+.4f} | — | "
            f"{delta.get('pr_auc', 0):+.4f} |  |"
        )
    if r8m:
        for v, label in [("strict",     "EXP8 Strict (5-seed mean, unbounded)"),
                         ("non_strict", "EXP8 Non-strict (5-seed mean, unbounded)")]:
            sub = r8m.get(v, {})
            mr, sr = sub.get("roc_auc_mean"), sub.get("roc_auc_std")
            mp, sp = sub.get("pr_auc_mean"),  sub.get("pr_auc_std")
            roc_s = f"{mr:.4f} +/-{sr:.4f}" if mr is not None else "—"
            pr_s  = (f"{mp:.4f} +/-{sp:.4f} ({mp/FLOOR_TIME:.2f}x)"
                     if mp is not None else "—")
            lines.append(f"| **{label}** | — | {roc_s} | — | {pr_s} | reference only |")

    # exp8 depth-equalized comparison
    r8d6 = all_results.get("exp8_both_depth6", {})
    if r8d6:
        lines += [
            "",
            "### EXP8 derinlik ayristirmasi: her iki model depth=6",
            "",
            "| Model | Test ROC-AUC | Test PR-AUC (lift) |",
            "|---|---|---|",
        ]
        for v, label in [("strict", "EXP8 Strict (depth=6)"),
                         ("non_strict", "EXP8 Non-strict (depth=6)")]:
            sub = r8d6.get(v, {})
            lines.append(
                f"| **{label}** "
                f"| {_f(_get(sub,'test','roc_auc'))} "
                f"| {_fp(_get(sub,'test','pr_auc'), FLOOR_TIME)} |"
            )
        d6  = r8d6.get("delta_same_depth", {})
        dprev = src8.get("delta", {})
        if d6 and dprev:
            lines += [
                "",
                "| Karsilastirma | ROC-AUC delta | PR-AUC delta |",
                "|---|---|---|",
                f"| Delta (strict=6, non-strict=7) — orijinal | "
                f"{dprev.get('roc_auc',0):+.4f} | {dprev.get('pr_auc',0):+.4f} |",
                f"| Delta (her ikisi de depth=6) | "
                f"{d6.get('roc_auc',0):+.4f} | {d6.get('pr_auc',0):+.4f} |",
                f"| Derinlik farkinin delta katkilisi | "
                f"{dprev.get('roc_auc',0)-d6.get('roc_auc',0):+.4f} | "
                f"{dprev.get('pr_auc',0)-d6.get('pr_auc',0):+.4f} |",
                "",
                "> Delta derinlik esitlenince de koruniyor: fark feature'dan geliyor, "
                "depth artefaktindan degil.",
            ]

    # ── ADIM 3a: Bootstrap CI ─────────────────────────────────────────────────
    r3b = all_results.get("adim3_bootstrap_ci", {})
    if r3b:
        cc = r3b.get("cell_ci", {})
        lines += [
            "",
            "## ADIM 3 — Bootstrap CI (n=1000) ve Etki Buyuklukleri",
            "",
            "### Her hucre icin 95% CI (ADIM 2 val-secilen depth)",
            "",
            "| Hucre | ROC-AUC [95% CI] | PR-AUC [95% CI] | Lift [95% CI] |",
            "|---|---|---|---|",
        ]
        for feat, split in [("leaky","random"),("leaky","time"),
                             ("clean","random"),("clean","time")]:
            key = f"{feat}_{split}"
            c   = cc.get(key, {})
            if not c:
                continue
            r_ci = c["roc_auc"]
            p_ci = c["pr_auc"]
            l_ci = c["lift"]
            lines.append(
                f"| **{feat.upper()} + {split}** "
                f"| {r_ci['mean']:.4f} [{r_ci['ci_lo']:.4f}, {r_ci['ci_hi']:.4f}] "
                f"| {p_ci['mean']:.4f} [{p_ci['ci_lo']:.4f}, {p_ci['ci_hi']:.4f}] "
                f"| {l_ci['mean']:.2f}x [{l_ci['ci_lo']:.2f}x, {l_ci['ci_hi']:.2f}x] |"
            )

        # Leakage effect (paired)
        lc = r3b.get("leakage_ci", {})
        lines += [
            "",
            "### Siziinti etkisi: leaky - clean (PAIRED bootstrap, ayni test satirlari)",
            "",
            "| Split | ROC-AUC farki [95% CI] | Lift farki [95% CI] | Sifir CI'da? |",
            "|---|---|---|---|",
        ]
        for split in ("random", "time"):
            e = lc.get(split, {})
            if not e:
                continue
            rd = e["roc_auc_diff"]
            ld = e["lift_diff"]
            z  = "**EVET** — anlamli degil" if ld["includes_zero"] else "Hayir — anlamli"
            lines.append(
                f"| {split} | {rd['mean']:+.4f} [{rd['ci_lo']:+.4f}, {rd['ci_hi']:+.4f}] "
                f"| {ld['mean']:+.2f}x [{ld['ci_lo']:+.2f}x, {ld['ci_hi']:+.2f}x] | {z} |"
            )

        # Split effect (independent)
        sc = r3b.get("split_ci", {})
        lines += [
            "",
            "### Split etkisi: random - time (INDEPENDENT bootstrap, farkli test satirlari)",
            "",
            "| Feature set | ROC-AUC farki [95% CI] | Lift farki [95% CI] | Sifir CI'da? |",
            "|---|---|---|---|",
        ]
        for feat in ("leaky", "clean"):
            e = sc.get(feat, {})
            if not e:
                continue
            rd = e["roc_auc_diff"]
            ld = e["lift_diff"]
            z  = "**EVET** — anlamli degil" if ld["includes_zero"] else "Hayir — anlamli"
            lines.append(
                f"| {feat.upper()} | {rd['mean']:+.4f} [{rd['ci_lo']:+.4f}, {rd['ci_hi']:+.4f}] "
                f"| {ld['mean']:+.2f}x [{ld['ci_lo']:+.2f}x, {ld['ci_hi']:+.2f}x] | {z} |"
            )
        lines += [
            "",
            "> **Not — clean features, ROC vs lift celiskisi:** "
            "CLEAN icin ROC farki anlamli (+0.0395, CI [+0.0207, +0.0574] sifiri icermiyor) "
            "ancak lift farki anlamli degil (+0.06x, CI [-0.08x, +0.23x] sifiri iceriyor). "
            "Bu bir celiskme degil: ROC-AUC prevalanstan bagimsiz bir metrik, "
            "lift ise PR-AUC'yi test prevalansina normalize ediyor. "
            "Iki split'in test dilimlerinde prevalans farkli (random=0.128, time=0.097), "
            "bu fark lift farki hesabinda govde gizleniyor. "
            "ROC, iki split arasinda gercek bir performans farki oldugunu gosteriyor; "
            "lift ise bu farkin prevalans normallestirilmis olcekte anlamliligi "
            "icin yeterli guc olmadigini soyluyor.",
        ]

    # ── ADIM 3b: Rolling-origin ───────────────────────────────────────────────
    r3r = all_results.get("adim3_rolling_origin", {})
    if r3r:
        lines += [
            "",
            "### Rolling-origin dogrulama (clean + time, val-depth=6)",
            "",
            "| Test penceresi | Kesim tarihi | n_train | n_test "
            "| Prev_train | Prev_test | Test ROC-AUC | Test PR-AUC | Lift |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for n_mo in [2, 3, 4]:
            tag = f"last_{n_mo}mo"
            r   = r3r.get(tag, {})
            if not r:
                continue
            lines.append(
                f"| Son {n_mo} ay | {r['cutpoint']} "
                f"| {r['n_train']} | {r['n_test']} "
                f"| {r['train_prevalence']:.4f} | {r['test_prevalence']:.4f} "
                f"| {r['test']['roc_auc']:.4f} "
                f"| {r['test']['pr_auc']:.4f} "
                f"| {r['test_pr_lift']:.2f}x |"
            )

    # ── ADIM 4: Permutation importance ───────────────────────────────────────
    r4p = all_results.get("adim4_permutation_importance", {})
    if r4p:
        lines += ["", "## ADIM 4 — Permutation Importance (test set)", ""]
        for key, title in [("leaky_time", "Leaky + time split"),
                            ("clean_time", "Clean + time split")]:
            r = r4p.get(key, {})
            if not r:
                continue
            top5_imp  = r["impurity_top20"][:5]
            top5_perm = r["permutation_top20"][:5]
            lines += [
                f"**{title}** — Top 5 features:",
                "",
                "| Rank | Impurity importance | Permutation importance (test) |",
                "|---|---|---|",
            ]
            for i, (ti, tp) in enumerate(zip(top5_imp, top5_perm), 1):
                lines.append(
                    f"| {i} | {ti['feature']} ({ti['importance']:.4f}) "
                    f"| {tp['feature']} ({tp['importance']:.4f} ±{tp['std']:.4f}) |"
                )
            lines.append("")

    # ── ADIM 5: HGBT 2x2 ─────────────────────────────────────────────────────
    r5h = all_results.get("adim5_hgbt_2x2", {})
    ra2 = all_results.get("adim2_val_depth_selection", {})
    if r5h and ra2:
        lines += [
            "## ADIM 5 — HGBT vs RandomForest: 2x2 (val-selected depth)",
            "",
            "| Hucre | RF Test ROC | RF Test PR (lift) | HGBT Test ROC | HGBT Test PR (lift) |",
            "|---|---|---|---|---|",
        ]
        for feat, split, floor in [
            ("leaky", "random", FLOOR_RAND),
            ("leaky", "time",   FLOOR_TIME),
            ("clean", "random", FLOOR_RAND),
            ("clean", "time",   FLOOR_TIME),
        ]:
            key = f"{feat}_{split}"
            rf  = ra2.get(key, {})
            hg  = r5h.get(key, {})
            lines.append(
                f"| **{feat.upper()} + {split}** "
                f"| {_f(_get(rf,'test','roc_auc'))} "
                f"| {_fp(_get(rf,'test','pr_auc'), floor)} "
                f"| {_f(_get(hg,'test','roc_auc'))} "
                f"| {_fp(_get(hg,'test','pr_auc'), floor)} |"
            )

    # ── ADIM 6: Business lift ─────────────────────────────────────────────────
    r6b = all_results.get("adim6_business_lift", {})
    if r6b:
        prev = r6b.get("test_prevalence", FLOOR_TIME)
        sl   = r6b.get("slices", {})
        lines += [
            "",
            "## ADIM 6 — Is anlami: Skor dilimlerinde gercek dusuk-puan orani",
            "",
            f"> Model: clean + time split, depth={r6b.get('model_depth')}. "
            f"Test seti prevalansi (taban) = {prev:.4f}.",
            "",
            "| Dilim | n | Dusuk-puan orani | Lift (oran / taban) |",
            "|---|---|---|---|",
            f"| Taban (tum test) | {r6b.get('n_test')} | {prev:.4f} | 1.00x |",
        ]
        for pct in [5, 10, 20]:
            s = sl.get(f"top_{pct}pct", {})
            if s:
                lines.append(
                    f"| En riskli %{pct} | {s['n']} "
                    f"| {s['rate']:.4f} | **{s['lift']:.2f}x** |"
                )

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        f"*Lift = PR-AUC / test prevalansi. "
        f"Random split floor = {FLOOR_RAND} | Time split floor = {FLOOR_TIME}. "
        f"Generated by `experiments.py`.*",
    ]

    md_path = os.path.join(RESULTS_DIR, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  -> {md_path}")
    return md_path


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sep = "=" * 62
    all_results = {}

    # Core experiments
    for name, fn in [
        ("exp1", exp1_leaky_model),
        ("exp2", exp2_clean_model),
        ("exp4", exp4_label_permutation),
        ("exp5", exp5_capacity_curve),
        ("exp6", exp6_baselines),
        ("exp8", exp8_seller_score_comparison),
    ]:
        print(f"\n{sep}")
        try:
            all_results[name] = fn()
        except Exception as exc:
            import traceback
            print(f"  ERROR in {name}: {exc}")
            traceback.print_exc()

    # Additional analyses
    for name, fn in [
        ("exp1_exp2_bounded", exp1_exp2_bounded),
        ("exp8_multi_seed",   exp8_multi_seed),
    ]:
        print(f"\n{sep}")
        try:
            all_results[name] = fn()
        except Exception as exc:
            import traceback
            print(f"  ERROR in {name}: {exc}")
            traceback.print_exc()

    # Summary table
    print(f"\n{sep}")
    print("SUMMARY — test-set metrics")
    print(sep)
    for name in ("exp1", "exp2", "exp4", "exp6", "exp8"):
        r = all_results.get(name, {})
        if "test" in r:
            t = r["test"]
            print(f"  {name}: ROC-AUC={t.get('roc_auc', 0):.4f}  "
                  f"PR-AUC={t.get('pr_auc', 0):.4f}")
        elif name == "exp8":
            for v in ("strict", "non_strict"):
                t = r.get(v, {}).get("test", {})
                if t:
                    print(f"  {name}/{v}: ROC-AUC={t.get('roc_auc', 0):.4f}  "
                          f"PR-AUC={t.get('pr_auc', 0):.4f}")

    bd = all_results.get("exp1_exp2_bounded", {})
    for k in ("leaky_depth12", "clean_depth12"):
        t = bd.get(k, {}).get("test", {})
        if t:
            print(f"  {k}: ROC-AUC={t.get('roc_auc', 0):.4f}  "
                  f"PR-AUC={t.get('pr_auc', 0):.4f}")

    ms = all_results.get("exp8_multi_seed", {})
    for v in ("strict", "non_strict"):
        sub = ms.get(v, {})
        if sub:
            print(f"  exp8/{v} (5-seed): "
                  f"ROC-AUC={sub['roc_auc_mean']:.4f}+/-{sub['roc_auc_std']:.4f}  "
                  f"PR-AUC={sub['pr_auc_mean']:.4f}+/-{sub['pr_auc_std']:.4f}")
    if "delta" in ms:
        print(f"  exp8/delta verdict: {ms['delta']['verdict']}")

    print(f"\n{sep}")
    generate_summary_md(all_results)

    print(f"\nResults -> {RESULTS_DIR}/")
    print(f"Figures -> {FIGURES_DIR}/")
