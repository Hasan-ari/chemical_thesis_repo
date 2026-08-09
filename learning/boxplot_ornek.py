"""Ogretici ornek: box plot nasil okunur + log eksen ne yapar.

Senaryo (basit): 3 arkadas 100 gun boyunca hava sicakligi tahmin ediyor.
Her gunun sonunda tek sayi var: o gun kac derece sasirdigi (= "gunluk hata").
Kisi basina 100 sayi -> her kisi icin BIR kutu. Ana RMSE grafigindeki
"koşu basina RMSE" ile ayni mantik: 1 kosu -> 1 sayi -> kutuya girer.

Sol panel: normal (lineer) eksen. Sag panel: AYNI veri, log eksen.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(7)

# Uc arkadas: Ali (tipikte iyi ama 3 felaket gunu var),
# Ayse (tipikte en iyi, 1 felaket), Mert (tipikte en kotu ama felaketsiz).
ali = np.concatenate([rng.lognormal(np.log(2.0), 0.35, 97), [22.0, 27.0, 31.0]])
ayse = np.concatenate([rng.lognormal(np.log(1.7), 0.30, 99), [18.0]])
mert = rng.lognormal(np.log(4.5), 0.25, 100)

people = ["Ali", "Ayşe", "Mert"]
data = [ali, ayse, mert]
colors = ["#2563EB", "#D97706", "#65752A"]

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
fig.suptitle(
    "Günlük tahmin hatası (°C) — aynı veri, iki eksen",
    fontsize=14,
)

for ax, log_scale in zip(axes, (False, True)):
    bp = ax.boxplot(
        data,
        tick_labels=people,
        patch_artist=True,
        widths=0.5,
        whis=1.5,
        flierprops=dict(marker="o", markersize=5, alpha=0.6),
        medianprops=dict(color="white", linewidth=2),
    )
    for patch, flier, color in zip(bp["boxes"], bp["fliers"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor("none")
    for part in ("whiskers", "caps"):
        for line, color in zip(bp[part], np.repeat(colors, 2)):
            line.set_color(color)

    if log_scale:
        ax.set_yscale("log")
        ticks = [1, 2, 3, 5, 10, 20, 30]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
        ax.set_title("Log eksen: sayılar AYNI, aralıklar 'kaç kat'")
    else:
        ax.set_title("Normal eksen: kutular dipte eziliyor")
    ax.set_ylabel("günlük hata (°C)")
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

# Sol panel: sorunlari isaretle
ax0 = axes[0]
ax0.annotate(
    "Ali'nin 3 felaket günü\n(outlier noktaları)",
    xy=(1.05, 27), xytext=(1.45, 24),
    arrowprops=dict(arrowstyle="->", color="0.3"), fontsize=9, color="0.25",
)
ax0.annotate(
    "kutular burada ezik,\nokunmuyor",
    xy=(2.75, 5.5), xytext=(2.1, 12),
    arrowprops=dict(arrowstyle="->", color="0.3"), fontsize=9, color="0.25",
)

# Sag panel: nasil okunacagini isaretle
ax1 = axes[1]
med_ali = float(np.median(ali))
ax1.annotate(
    f"beyaz çizgi = medyan\nAli tipik günde ~{med_ali:.1f}°C şaşıyor\n(soldaki sayıyı DİREKT oku)",
    xy=(1.26, med_ali), xytext=(1.5, 1.05),
    arrowprops=dict(arrowstyle="->", color="0.3"), fontsize=9, color="0.25",
)
ax1.annotate(
    "kutu = günlerin ortadaki %50'si",
    xy=(2.26, float(np.median(ayse))), xytext=(2.4, 0.85),
    arrowprops=dict(arrowstyle="->", color="0.3"), fontsize=9, color="0.25",
)
ax1.annotate(
    "Mert: tipikte EN KÖTÜ (kutu yukarıda)\nama felaketi yok (üstte nokta yok)\n= Halite'nin hikâyesi",
    xy=(3.0, 8.2), xytext=(2.35, 17),
    arrowprops=dict(arrowstyle="->", color="0.3"), fontsize=9, color="0.25",
)

fig.tight_layout()
out = "learning/boxplot_ornek.png"
fig.savefig(out, dpi=150, facecolor="#fcfcfb")
print("yazildi:", out)
print("medyanlar:", {p: round(float(np.median(d)), 2) for p, d in zip(people, data)})
