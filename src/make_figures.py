"""
make_figures.py — Makale için güncel grafikler
Türkçe başlık/eksen, İngilizce sütun adları, 150 dpi
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_DARK  = "#1b4332"
C_MED   = "#2d6a4f"
C_LIGHT = "#52b788"
C_PALE  = "#95d5b2"
C_WARM  = "#e76f51"
C_GOLD  = "#f4a261"

plt.rcParams.update({
    "font.size":         13,
    "axes.titlesize":    16,
    "axes.labelsize":    13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

FIG        = "results/figures"
FLOOR_RAND = 0.1280
FLOOR_TIME = 0.0967
FLOOR_TR   = 0.1358   # eğitim seti prevalansı (zaman bölümü)


# ── (a) exp4_memorization.png — Ezber kapasitesi, ROC-AUC ─────────────────────
def fig_exp4():
    d4 = json.load(open("results/exp4_memorization.json"))

    groups  = ["Budanmamış RF\n(max_depth=None)", "Depth-6 RF\n(val-seçimli)"]
    blabels = ["Gerçek — Eğitim", "Gerçek — Test",
               "Karıştırılmış — Eğitim", "Karıştırılmış — Test"]
    colors  = [C_DARK, C_LIGHT, C_WARM, C_GOLD]
    hatches = [None, None, "///", "///"]

    vals = []
    for dtag in ("unbounded", "depth6"):
        row = []
        for ltag, spl in [("real","train"), ("real","test"),
                          ("permuted","train"), ("permuted","test")]:
            row.append(d4[dtag][ltag][spl]["roc_auc"])
        vals.append(row)

    fig, ax = plt.subplots(figsize=(12, 6))
    x, n, w = np.arange(2), 4, 0.18
    for bi, (lbl, col, hat) in enumerate(zip(blabels, colors, hatches)):
        offs  = x + (bi - (n - 1) / 2) * w
        bvals = [vals[0][bi], vals[1][bi]]
        bars  = ax.bar(offs, bvals, w, label=lbl, color=col,
                       hatch=hat, edgecolor="white" if hat else col, linewidth=0)
        for bo, v in zip(bars, bvals):
            ax.text(bo.get_x() + bo.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", fontsize=9, rotation=90)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.8,
               label="Rastgele sınır (ROC = 0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.18)
    ax.set_title("EXP4 — Ezber Kapasitesi (clean + zaman bölümü)")
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{FIG}/exp4_memorization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("(a) exp4_memorization.png")


# ── (b) permutation_importance_leaky.png — orijinal sütun adları ──────────────
def fig_perm_importance():
    dpi_data = json.load(open("results/adim4_permutation_importance.json"))
    top10    = dpi_data["leaky_time"]["permutation_top20"][:10]

    names = [t["feature"] for t in top10]    # orijinal İngilizce adlar
    vals  = [t["importance"] for t in top10]
    errs  = [t["std"] for t in top10]

    fig, ax = plt.subplots(figsize=(12, 6))
    ypos = list(range(len(names)))
    ax.barh(ypos[::-1], vals, color=C_MED, height=0.6,
            xerr=errs, error_kw={"capsize": 5, "color": "black", "linewidth": 1.2})
    ax.set_yticks(ypos)
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Ortalama PR-AUC düşüşü (permütasyon)")
    ax.set_title("ADIM 4 — Permütasyon Önemi\n(leaky + zaman bölümü, test seti, top 10)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/permutation_importance_leaky.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("(b) permutation_importance_leaky.png")


# ── (c) capacity_curve_clean_time.png — train, val, test + ezber makası ────────
def fig_capacity():
    a2     = json.load(open("results/adim2_val_depth_selection.json"))
    ct     = a2["clean_time"]
    depths = ct["depths"]
    tr_pr  = ct["train2_pr_auc"]
    val_pr = ct["val_pr_auc"]
    te_pr  = ct["test_pr_auc_curve"]
    best_d = ct["best_depth"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(depths, tr_pr,  color=C_PALE,  linewidth=2.0, linestyle="-",
            label="Eğitim PR-AUC")
    ax.plot(depths, val_pr, color=C_MED,   linewidth=2.5, linestyle="-",
            label="Doğrulama PR-AUC (derinlik seçimi için kullanıldı)")
    ax.plot(depths, te_pr,  color=C_WARM,  linewidth=2.5, linestyle="--",
            label="Test PR-AUC (yalnızca referans)")

    # Ezber makası: eğitim ile test arasındaki boşluk
    tr_arr = np.array(tr_pr)
    te_arr = np.array(te_pr)
    ax.fill_between(depths, tr_arr, te_arr,
                    where=(tr_arr > te_arr),
                    alpha=0.15, color=C_WARM, label="Ezber makası (eğitim > test)")

    ax.axvline(best_d, color=C_DARK, linestyle=":", linewidth=2,
               label=f"Seçilen derinlik = {best_d}")
    ax.axhline(FLOOR_TIME, color=C_GOLD, linestyle="-.", linewidth=1.4, alpha=0.85,
               label=f"Test tabanı (prevalans = {FLOOR_TIME})")

    ax.set_xlabel("max_depth (RandomForest)")
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0.05, None)
    ax.set_title("EXP5 — Kapasite Eğrisi (clean + zaman bölümü, val-seçimli derinlik)")
    ax.legend(fontsize=11, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{FIG}/capacity_curve_clean_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("(c) capacity_curve_clean_time.png")


# ── (d) prevalence_over_time.png — iki referans çizgisi, ≥100 sipariş ─────────
def fig_prevalence():
    dp        = json.load(open("results/prevalence_over_time.json"))
    months_dt = pd.to_datetime([m + "-01" for m in dp["months"]])
    prevs     = np.array(dp["prevalence"])
    counts    = np.array(dp["order_count"])
    cut_dt    = pd.to_datetime(dp["train_test_cut"])

    # 100'den az sipariş olan ayları çıkar
    mask     = counts >= 100
    m_filt   = months_dt[mask]
    p_filt   = prevs[mask]
    c_filt   = counts[mask]
    rolling  = pd.Series(p_filt).rolling(3, center=True, min_periods=2).mean().values

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()

    ax2.bar(m_filt, c_filt, width=25, color=C_PALE, alpha=0.35, label="Sipariş sayısı")
    ax1.plot(m_filt, p_filt, color=C_WARM, linewidth=2.5,
             marker="o", markersize=5, label="Aylık düşük puan oranı")
    ax1.plot(m_filt, rolling, color=C_MED, linewidth=2.0,
             linestyle="-.", alpha=0.85, label="3 aylık hareketli ortalama")

    # İki referans çizgisi: train ve test prevalansı
    ax1.axhline(FLOOR_TR, color=C_DARK, linestyle="--", linewidth=1.5,
                label=f"Eğitim prevalansı ({FLOOR_TR:.4f})")
    ax1.axhline(FLOOR_TIME, color=C_WARM, linestyle=":", linewidth=1.8,
                alpha=0.8, label=f"Test prevalansı ({FLOOR_TIME:.4f})")

    ax1.axvline(cut_dt, color="black", linestyle=":", linewidth=2.0,
                label=f"Eğitim/test kesimi ({cut_dt.strftime('%Y-%m')})")

    ax1.set_xlabel("Ay")
    ax1.set_ylabel("Düşük puan oranı (puan ≤ 2)", color=C_WARM)
    ax2.set_ylabel("Sipariş sayısı", color=C_MED)
    ax1.set_ylim(0, max(p_filt) * 1.45)
    ax1.set_title("ADIM 1 — Aylık Düşük Puan Oranı (Dağılım Kayması)")

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/prevalence_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("(d) prevalence_over_time.png")


# ── (e) risk_deciles.png — skor dilimi lift ───────────────────────────────────
def fig_risk():
    d6b      = json.load(open("results/adim6_business_lift.json"))
    sl       = d6b["slices"]
    baseline = d6b["test_prevalence"]

    labels = ["En riskli %5", "En riskli %10", "En riskli %20", "Tüm test\n(taban)"]
    rates  = [sl["top_5pct"]["rate"], sl["top_10pct"]["rate"],
              sl["top_20pct"]["rate"], baseline]
    lifts  = [sl["top_5pct"]["lift"], sl["top_10pct"]["lift"],
              sl["top_20pct"]["lift"], 1.0]
    col_d  = [C_DARK, C_MED, C_LIGHT, C_PALE]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    bars = ax.bar(labels, rates, color=col_d, width=0.5)
    ax.axhline(baseline, color=C_WARM, linestyle="--", linewidth=1.8,
               label=f"Prevalans tabanı ({baseline:.4f})")
    ax.set_ylabel("Düşük puan oranı")
    ax.set_title("Skor Dilimine Göre Düşük Puan Oranı")
    ax.legend(fontsize=11)
    for bo, v in zip(bars, rates):
        ax.text(bo.get_x() + bo.get_width() / 2, v + 0.003,
                f"{v:.3f}", ha="center", fontsize=11)

    ax2 = axes[1]
    bars2 = ax2.bar(labels, lifts, color=col_d, width=0.5)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Lift = 1 (rastgele)")
    ax2.set_ylabel("Lift (oran / prevalans)")
    ax2.set_title("Skor Dilimine Göre Lift")
    ax2.legend(fontsize=11)
    for bo, v in zip(bars2, lifts):
        ax2.text(bo.get_x() + bo.get_width() / 2, v + 0.04,
                 f"{v:.2f}×", ha="center", fontsize=11)

    fig.suptitle(
        "ADIM 6 — İş Anlamı: Modelin En Riskli Dilimlerdeki Hassasiyeti",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(f"{FIG}/risk_deciles.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("(e) risk_deciles.png")


# ── (f) adim2_val_vs_test_curves.png — lift'e dönüştürülmüş ──────────────────
def fig_adim2_lift():
    a2 = json.load(open("results/adim2_val_depth_selection.json"))

    configs = [
        ("leaky_random", "Sızıntılı + rastgele bölüm", FLOOR_RAND, FLOOR_RAND, FLOOR_RAND),
        ("leaky_time",   "Sızıntılı + zaman bölümü",   FLOOR_TR,   FLOOR_TR,   FLOOR_TIME),
        ("clean_random", "Temiz + rastgele bölüm",      FLOOR_RAND, FLOOR_RAND, FLOOR_RAND),
        ("clean_time",   "Temiz + zaman bölümü",        FLOOR_TR,   FLOOR_TR,   FLOOR_TIME),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (key, title, fl_tr, fl_val, fl_te) in zip(axes.flat, configs):
        ct     = a2[key]
        depths = ct["depths"]
        bd     = ct["best_depth"]

        tr_lift  = [v / fl_tr  for v in ct["train2_pr_auc"]]
        val_lift = [v / fl_val for v in ct["val_pr_auc"]]
        te_lift  = [v / fl_te  for v in ct["test_pr_auc_curve"]]

        ax.plot(depths, tr_lift,  color=C_PALE, linewidth=1.8, linestyle="-",
                label="Eğitim (lift)")
        ax.plot(depths, val_lift, color=C_MED,  linewidth=2.5, linestyle="-",
                label="Doğrulama (lift, seçim için)")
        ax.plot(depths, te_lift,  color=C_WARM, linewidth=2.0, linestyle="--",
                label="Test (lift, referans)")
        ax.axvline(bd, color=C_DARK, linestyle=":", linewidth=1.5,
                   label=f"Seçilen derinlik = {bd}")
        ax.axhline(1.0, color="gray", linestyle="-.", linewidth=1.0,
                   alpha=0.6, label="Lift = 1 (taban)")

        ax.set_xlabel("max_depth")
        ax.set_ylabel("Lift (PR-AUC / prevalans)")
        ax.set_title(title)
        ax.legend(fontsize=9)

    fig.suptitle(
        "ADIM 2 — Doğrulama ile Seçilen Derinlik: Eğitim / Doğrulama / Test Lift Eğrileri",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(f"{FIG}/adim2_val_vs_test_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("(f) adim2_val_vs_test_curves.png (lift ekseni)")


# ── (g) importance_comparison_leaky.png — 2 panel: impurity + permutation ────
def fig_importance_comparison():
    dpi_data = json.load(open("results/adim4_permutation_importance.json"))
    lt       = dpi_data["leaky_time"]

    top_imp  = lt["impurity_top20"][:10]
    top_perm = lt["permutation_top20"][:10]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Sol: impurity importance
    ax = axes[0]
    names_i = [t["feature"] for t in top_imp]
    vals_i  = [t["importance"] for t in top_imp]
    ypos    = list(range(len(names_i)))
    ax.barh(ypos[::-1], vals_i, color=C_MED, height=0.65)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names_i[::-1])
    ax.set_xlabel("Ortalama safsızlık azalması")
    ax.set_title("Safsızlık Tabanlı Önem\n(Eğitim seti, varsayılan)")

    # Sağ: permutation importance
    ax2 = axes[1]
    names_p = [t["feature"] for t in top_perm]
    vals_p  = [t["importance"] for t in top_perm]
    errs_p  = [t["std"] for t in top_perm]
    ypos2   = list(range(len(names_p)))
    ax2.barh(ypos2[::-1], vals_p, color=C_WARM, height=0.65,
             xerr=errs_p, error_kw={"capsize": 5, "color": "black", "linewidth": 1.2})
    ax2.set_yticks(ypos2)
    ax2.set_yticklabels(names_p[::-1])
    ax2.set_xlabel("Ortalama PR-AUC düşüşü")
    ax2.set_title("Permütasyon Önemi\n(Test seti, gerçek etki)")

    fig.suptitle(
        "ADIM 4 — Önem Karşılaştırması: Leaky + Zaman Bölümü (top 10)",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(f"{FIG}/importance_comparison_leaky.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("(g) importance_comparison_leaky.png")


# ── Çalıştır ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig_exp4()
    fig_perm_importance()
    fig_capacity()
    fig_prevalence()
    fig_risk()
    fig_adim2_lift()
    fig_importance_comparison()
    print("\nTüm grafikler tamamlandı.")
