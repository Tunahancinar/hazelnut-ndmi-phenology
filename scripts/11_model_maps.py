"""Model haritaları: 2025 için on modelin mekânsal CV tahminleri ve gözlenen ΔNDMI; 2023-2026 DSI haritaları."""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import TwoSlopeNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harita_taban import taban

V = "/home/claude/data/csv"; R, O = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
W = dict(x=(29.65, 32.05), y=(40.45, 41.45)); E = dict(x=(38.55, 40.45), y=(40.45, 41.25))
oz = pd.read_csv(f"{V}/parsel_oznitelik.csv")[["PID", "LON", "LAT"]]
t = pd.read_csv(f"{R}/model_tahminleri.csv").merge(oz, on="PID")
dv = pd.read_csv(f"{R}/model_veri_seti.csv")[["PID", "YIL", "dNDMI", "DSI"]]
t = t.merge(dv, on=["PID", "YIL"])
MOD = ["MLR", "SVR", "kNN", "MLP", "RF", "ET", "GBM", "XGBoost", "LightGBM", "CatBoost"]


def ciz(cell, df, col, cmap, vmin, vmax, title, s=1.6, first_col=True, last_row=True, wlab=False):
    g = GridSpecFromSubplotSpec(1, 2, subplot_spec=cell, width_ratios=[2.4, 1.9], wspace=0.05)
    sc = None
    for k, w in enumerate((W, E)):
        a = plt.subplot(g[k]); taban(a, lw=0.25)
        q = df[df.LON.between(*w["x"]) & df.LAT.between(*w["y"])].sort_values(col, key=lambda x: -np.abs(x))
        nrm = TwoSlopeNorm(0, vmin, vmax) if vmin < 0 < vmax else None
        sc = a.scatter(q.LON, q.LAT, c=q[col], cmap=cmap, norm=nrm, vmin=None if nrm else vmin, vmax=None if nrm else vmax, s=s, lw=0, zorder=3)
        a.set_xlim(*w["x"]); a.set_ylim(*w["y"]); a.set_aspect(1.3); a.set_anchor("S")
        a.set_xlabel(""); a.set_ylabel("")
        a.tick_params(labelsize=5.5, length=2, pad=1)
        a.set_xticks([30, 31, 32] if k == 0 else [39, 40]); a.set_yticks([40.6, 41.0, 41.4] if k == 0 else [40.6, 41.0])
        if not last_row: a.set_xticklabels([])
        if k == 1 or not first_col: a.set_yticklabels([])
        for sp in a.spines.values(): sp.set_lw(0.5)
        if k == 0: a.set_title(title, fontsize=7.5, loc="left", pad=2)
        if last_row: a.set_xlabel("Western Black Sea (°E)" if k == 0 else "Eastern Black Sea (°E)", fontsize=6, labelpad=1)
    return sc


# Şekil A: 2025 tahmin haritaları
y = t[t.YIL == 2025].copy()
y["SD"] = y[[f"OOF_{m}" for m in MOD]].std(axis=1)
pan = [("dNDMI", "Observed ΔNDMI")] + [(f"OOF_{m}", m) for m in MOD]
fig = plt.figure(figsize=(7.2, 8.0))
gs = GridSpec(6, 2, figure=fig, left=0.06, right=0.99, top=0.975, bottom=0.10, hspace=0.30, wspace=0.06)
lab = "abcdefghijkl"
for i, (c, ttl) in enumerate(pan):
    sc = ciz(gs[i // 2, i % 2], y, c, "BrBG", -0.30, 0.10, f"({lab[i]}) {ttl}", first_col=i % 2 == 0, last_row=i // 2 == 5, wlab=i < 2)
sc2 = ciz(gs[5, 1], y, "SD", "magma_r", 0, 0.05, "(l) Inter-model standard deviation", first_col=False)
c1 = fig.add_axes([0.08, 0.045, 0.40, 0.013]); cb = fig.colorbar(sc, cax=c1, orientation="horizontal"); cb.set_ticks([-0.3, -0.2, -0.1, 0, 0.1])
cb.set_label("ΔNDMI, 2025 (Jul–Aug minus Apr–May)", fontsize=7); cb.ax.tick_params(labelsize=6)
c2 = fig.add_axes([0.56, 0.045, 0.40, 0.013]); cb = fig.colorbar(sc2, cax=c2, orientation="horizontal")
cb.set_label("Standard deviation across ten models", fontsize=7); cb.ax.tick_params(labelsize=6)
fig.savefig(f"{O}/Fig_model_maps_2025.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# Şekil B: temporal hold-out 2025 (model 2025'i hiç görmedi) en iyi dört model ve gözlenen
best4 = ["kNN", "CatBoost", "RF", "LightGBM"]
fig = plt.figure(figsize=(7.2, 4.5))
gs = GridSpec(3, 2, figure=fig, left=0.06, right=0.99, top=0.96, bottom=0.15, hspace=0.2, wspace=0.06)
pan = [("dNDMI", "Observed ΔNDMI"), ("OOF_RF", "RF, spatial block CV")] + [(f"T2025_{m}", f"{m}, 2025 held out") for m in best4]
for i, (c, ttl) in enumerate(pan):
    sc = ciz(gs[i // 2, i % 2], y, c, "BrBG", -0.30, 0.10, f"({lab[i]}) {ttl}", first_col=i % 2 == 0, last_row=i // 2 == 2, wlab=i < 2)
c1 = fig.add_axes([0.3, 0.06, 0.4, 0.018]); cb = fig.colorbar(sc, cax=c1, orientation="horizontal"); cb.set_ticks([-0.3, -0.2, -0.1, 0, 0.1])
cb.set_label("ΔNDMI, 2025", fontsize=7); cb.ax.tick_params(labelsize=6)
fig.savefig(f"{O}/Fig_model_maps_holdout2025.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# Şekil C: DSI haritaları 2023-2026
fig = plt.figure(figsize=(7.2, 3.4))
gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.99, top=0.95, bottom=0.2, hspace=0.2, wspace=0.06)
for i, yr in enumerate([2023, 2024, 2025, 2026]):
    sc = ciz(gs[i // 2, i % 2], t[t.YIL == yr], "DSI", "BrBG", -2.5, 2.5, f"({lab[i]}) {yr}", first_col=i % 2 == 0, last_row=i // 2 == 1, wlab=i < 2)
c1 = fig.add_axes([0.3, 0.07, 0.4, 0.025]); cb = fig.colorbar(sc, cax=c1, orientation="horizontal"); cb.set_ticks([-2, -1, 0, 1, 2])
cb.set_label("Standardized decline index (SDI)", fontsize=7); cb.ax.tick_params(labelsize=6)
fig.savefig(f"{O}/Fig_SDI_map.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# bölge özetleri (metin için)
ozb = pd.read_csv(f"{V}/parsel_oznitelik.csv")[["PID", "BOLGE_ADI"]]
q = t.merge(ozb, on="PID")
rg = q[q.YIL == 2025].groupby("BOLGE_ADI").agg(n=("PID", "size"), obs=("dNDMI", "mean"), rf=("OOF_RF", "mean"),
                                                 cat=("OOF_CatBoost", "mean"), stres=("DSI", lambda v: (v <= -1).mean()))
rg.to_csv(f"{R}/bolge_2025.csv"); print(rg.round(3))
from sklearn.metrics import r2_score
rr = {}
for b, g in q.groupby("BOLGE_ADI"):
    if len(g) > 500:
        rr[b] = {m: round(r2_score(g.dNDMI, g[f"OOF_{m}"]), 3) for m in MOD}
pd.DataFrame(rr).to_csv(f"{R}/bolge_r2.csv"); print(pd.DataFrame(rr))
