"""
make_tables.py — Makale için dört tablo PNG (150 dpi, Türkçe)
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/figures/tables", exist_ok=True)

HEADER_BG = "#1b4332"   # koyu yeşil
ROW_ALT   = "#f2f7f4"   # açık yeşilimsi (çift satırlar)
GRID      = "#cccccc"
BLACK     = "#111111"

plt.rcParams.update({"font.family": "DejaVu Sans"})


def make_table(fname, title, headers, rows, col_widths=None, figsize=(11, None)):
    """
    Sade Matplotlib tablo.
    col_widths: her sütunun 0–1 aralığında göreli genişliği (toplamı 1 olmalı).
    figsize[1] None ise satır sayısına göre otomatik hesaplanır.
    """
    n_cols = len(headers)
    n_rows = len(rows)

    if col_widths is None:
        col_widths = [1 / n_cols] * n_cols

    row_h  = 0.58          # inç cinsinden satır yüksekliği
    head_h = 0.72
    pad    = 0.55          # başlık için üst boşluk
    height = figsize[1] if figsize[1] else head_h + n_rows * row_h + pad + 0.2

    fig, ax = plt.subplots(figsize=(figsize[0], height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, height)
    ax.axis("off")

    # ── Başlık ────────────────────────────────────────────────────────────────
    ax.text(0.5, height - 0.20, title,
            ha="center", va="top", fontsize=14, fontweight="bold", color=BLACK)

    # ── Header satırı ─────────────────────────────────────────────────────────
    y_top  = height - pad
    x = 0.0
    for j, (h, w) in enumerate(zip(headers, col_widths)):
        rect = plt.Rectangle((x, y_top - head_h), w, head_h,
                              fc=HEADER_BG, ec="white", lw=0.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y_top - head_h / 2, h,
                ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="white",
                multialignment="center")
        x += w

    # ── Veri satırları ────────────────────────────────────────────────────────
    for i, row in enumerate(rows):
        y_row = y_top - head_h - (i + 1) * row_h
        bg    = ROW_ALT if i % 2 == 0 else "white"
        x = 0.0
        for j, (cell, w) in enumerate(zip(row, col_widths)):
            rect = plt.Rectangle((x, y_row), w, row_h,
                                  fc=bg, ec=GRID, lw=0.5)
            ax.add_patch(rect)
            ax.text(x + w / 2, y_row + row_h / 2, str(cell),
                    ha="center", va="center", fontsize=10.5, color=BLACK,
                    multialignment="center")
            x += w

    # Alt çizgi
    ax.plot([0, 1], [y_top - head_h - n_rows * row_h] * 2,
            color=GRID, lw=0.8)

    fig.tight_layout(pad=0.1)
    path = f"results/figures/tables/{fname}"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  {fname}")


# ── Veri kaynakları ───────────────────────────────────────────────────────────
a2   = json.load(open("results/adim2_val_depth_selection.json"))
boot = json.load(open("results/adim3_bootstrap_ci.json"))
r6t  = json.load(open("results/exp6_time.json"))
r_lr = json.load(open("results/lr_clean_time.json"))
r5h  = json.load(open("results/adim5_hgbt_2x2.json"))
r8d6 = json.load(open("results/exp8_both_depth6.json"))
rb   = json.load(open("results/adim3_bootstrap_ci.json"))

FR = 0.1280
FT = 0.0967


# ── (a) tablo_2x2.png ─────────────────────────────────────────────────────────
def t_2x2():
    def fmt(key, floor):
        r   = a2[key]["test"]
        pr  = r["pr_auc"]
        roc = r["roc_auc"]
        lci = boot["cell_ci"][key]["lift"]
        return (
            f"{pr/floor:.2f}×\n"
            f"[{lci['ci_lo']:.2f}×, {lci['ci_hi']:.2f}×]\n"
            f"ROC {roc:.3f}"
        )

    make_table(
        "tablo_2x2.png",
        "2×2 — Özellik Sızıntısı × Bölüm Stratejisi  (val-seçimli derinlik, n=1 000 bootstrap)",
        ["", "Rastgele Bölme", "Zaman Bölmesi"],
        [
            ["Sızıntılı\nÖzellikler",
             fmt("leaky_random", FR),
             fmt("leaky_time",   FT)],
            ["Temiz\nÖzellikler",
             fmt("clean_random", FR),
             fmt("clean_time",   FT)],
        ],
        col_widths=[0.20, 0.40, 0.40],
        figsize=(11, None),
    )


# ── (b) tablo_modeller.png ────────────────────────────────────────────────────
def t_modeller():
    rows = [
        ["Dummy (stratified)",
         f"{r6t['dummy_stratified']['roc_auc']:.3f}",
         f"{r6t['dummy_stratified']['pr_auc']:.3f}",
         f"{r6t['dummy_stratified']['pr_auc']/FT:.2f}×"],
        ["Lojistik Regresyon",
         f"{r_lr['test']['roc_auc']:.3f}",
         f"{r_lr['test']['pr_auc']:.3f}",
         f"{r_lr['test']['lift']:.2f}×"],
        ["RandomForest\n(depth=6, val-seçimli)",
         f"{a2['clean_time']['test']['roc_auc']:.3f}",
         f"{a2['clean_time']['test']['pr_auc']:.3f}",
         f"{a2['clean_time']['test']['pr_auc']/FT:.2f}×"],
        ["HistGradientBoosting\n(depth val-seçimli)",
         f"{r5h['clean_time']['test']['roc_auc']:.3f}",
         f"{r5h['clean_time']['test']['pr_auc']:.3f}",
         f"{r5h['clean_time']['test']['pr_auc']/FT:.2f}×"],
    ]
    make_table(
        "tablo_modeller.png",
        "Model Karşılaştırması — Temiz + Zaman Bölümü\n(taban: prevalans = 0.0967)",
        ["Model", "Test ROC-AUC", "Test PR-AUC", "Lift"],
        rows,
        col_widths=[0.38, 0.20, 0.20, 0.22],
        figsize=(11, None),
    )


# ── (c) tablo_desiller.png ────────────────────────────────────────────────────
def t_desiller():
    rows = [
        ["Taban (tüm test)", "19 167", "0.097", "1.00×", "1 854", "%100.0"],
        ["En riskli %5",     "958",    "0.280", "2.89×", "268",   "%14.5"],
        ["En riskli %10",    "1 916",  "0.223", "2.30×", "427",   "%23.0"],
        ["En riskli %20",    "3 833",  "0.158", "1.64×", "607",   "%32.7"],
    ]
    make_table(
        "tablo_desiller.png",
        "Risk Dilimleri — Temiz + Zaman Bölümü (depth=6)\n"
        "Toplam test: 19 167 sipariş · 1 854 düşük puan · prevalans 0.097",
        ["Dilim", "n", "Düşük Puan\nOranı", "Lift", "Yakalanan\n(n)", "Recall"],
        rows,
        col_widths=[0.22, 0.10, 0.14, 0.12, 0.16, 0.12],
        figsize=(12, None),
    )


# ── (d) tablo_sizintilar.png ──────────────────────────────────────────────────
def t_sizintilar():
    lc  = rb["leakage_ci"]
    sc  = rb["split_ci"]

    rows = [
        ["Özellik sızıntısı\n(post-purchase veri)",
         "Sızıntılı − Temiz\n(zaman bölümü)",
         f"+{lc['time']['lift_diff']['mean']:.2f}×",
         "Lift",
         f"[{lc['time']['lift_diff']['ci_lo']:+.2f}×, {lc['time']['lift_diff']['ci_hi']:+.2f}×]  Anlamlı"],

        ["Özellik sızıntısı\n(post-purchase veri)",
         "Sızıntılı − Temiz\n(zaman bölümü)",
         f"+{lc['time']['roc_auc_diff']['mean']:.3f}",
         "ROC-AUC",
         f"[{lc['time']['roc_auc_diff']['ci_lo']:+.3f}, {lc['time']['roc_auc_diff']['ci_hi']:+.3f}]  Anlamlı"],

        ["Bölüm stratejisi\n(rastgele − zaman)",
         "Sızıntılı özellikler",
         f"+{sc['leaky']['lift_diff']['mean']:.2f}×",
         "Lift",
         f"[{sc['leaky']['lift_diff']['ci_lo']:+.2f}×, {sc['leaky']['lift_diff']['ci_hi']:+.2f}×]  Anlamlı"],

        ["Bölüm stratejisi\n(rastgele − zaman)",
         "Temiz özellikler",
         f"+{sc['clean']['lift_diff']['mean']:.2f}×",
         "Lift",
         f"[{sc['clean']['lift_diff']['ci_lo']:+.2f}×, {sc['clean']['lift_diff']['ci_hi']:+.2f}×]  Anlamsız"],

        ["Gecikmeli satıcı\nskoru (temporal)",
         "Non-strict − Strict\n(her ikisi depth=6)",
         f"+{r8d6['delta_same_depth']['roc_auc']:+.4f}",
         "ROC-AUC",
         "Küçük, feature farkından\n(depth artefaktı değil)"],
    ]

    make_table(
        "tablo_sizintilar.png",
        "Sızıntı Türleri ve Etkileri  (%95 bootstrap CI, n=1 000)",
        ["Sızıntı Türü", "Kıyaslama", "Etki", "Birim", "%95 CI / Not"],
        rows,
        col_widths=[0.22, 0.22, 0.10, 0.12, 0.34],
        figsize=(14, None),
    )


if __name__ == "__main__":
    t_2x2()
    t_modeller()
    t_desiller()
    t_sizintilar()
    print("\nTüm tablolar: results/figures/tables/")
