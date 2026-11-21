"""Korelasyon analizi, mekânsal ve zamansal otokorelasyon, VIF ile değişken eleme."""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

V, R, O = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_model_findik.py")).read().split('if __name__ == "__main__":')[0]
ns = {"__file__": "x"}; exec(compile(src, "m03", "exec"), ns)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
df, _ = ns["oku"](V); d = ns["hedef_hazirla"](df).dropna(subset=["dNDMI"]).reset_index(drop=True)
grp = {v: "Climate" for v in ns["IKLIM"]} | {v: "Terrain" for v in ns["TOPO"]} | {v: "Soil" for v in ns["TOPRAK"]} | {v: "Stand" for v in ns["MESCERE"]}
feats = [c for c in ns["IKLIM"] + ns["TOPO"] + ns["TOPRAK"] + ns["MESCERE"] if c in d.columns]
X = d[feats].fillna(d[feats].median())
OUT = {}

# 1 Korelasyon
P = X.corr(method="pearson"); S = X.corr(method="spearman")
P.round(3).to_csv(f"{R}/korelasyon_pearson.csv"); S.round(3).to_csv(f"{R}/korelasyon_spearman.csv")
pairs = []
for i, a in enumerate(feats):
    for b in feats[i + 1:]:
        if abs(P.loc[a, b]) > 0.80 or abs(S.loc[a, b]) > 0.80:
            pairs.append((a, b, round(P.loc[a, b], 3), round(S.loc[a, b], 3)))
pairs = pd.DataFrame(pairs, columns=["Var1", "Var2", "Pearson", "Spearman"]).sort_values("Pearson", key=abs, ascending=False)
pairs.to_csv(f"{R}/korelasyon_080_cifler.csv", index=False); OUT["n_pairs_080"] = len(pairs)
print(pairs.to_string())
# hedef ile korelasyon
OUT["target_pearson"] = d[feats + ["dNDMI"]].corr()["dNDMI"].drop("dNDMI").round(3).to_dict()

# 2 VIF (korelasyon matrisinin tersinin köşegeni) ve adımsal eleme
def vif(cols):
    c = np.corrcoef(X[cols].values, rowvar=False)
    return pd.Series(np.diag(np.linalg.pinv(c)), index=cols)

v0 = vif(feats); v0.round(2).to_csv(f"{R}/vif_baslangic.csv")
keep = list(feats); steps = []
while True:
    v = vif(keep)
    if v.max() <= 10: break
    w = v.idxmax(); steps.append((w, round(float(v.max()), 2))); keep.remove(w)
# ikinci adım: |r| > 0.80 kalan çiftlerde hedefle ilişkisi zayıf olanı çıkar
tr = d[feats + ["dNDMI"]].corr(method="spearman")["dNDMI"].abs()
pair_steps = []
while True:
    C = X[keep].corr().abs().values.copy(); np.fill_diagonal(C, 0)
    if C.max() <= 0.80: break
    i, j = np.unravel_index(C.argmax(), C.shape); a, b = keep[i], keep[j]
    w = a if tr[a] < tr[b] else b; pair_steps.append((w, a if w == b else b, round(float(C.max()), 3))); keep.remove(w)
OUT["pair_removed"] = pair_steps
v1 = vif(keep); v1.round(2).to_csv(f"{R}/vif_son.csv")
OUT["vif_initial_max"] = round(float(v0.max()), 1); OUT["vif_initial_n_gt10"] = int((v0 > 10).sum()); OUT["vif_initial_n_gt5"] = int((v0 > 5).sum())
OUT["vif_removed"] = steps; OUT["vif_final_max"] = round(float(v1.max()), 2); OUT["kept"] = keep
OUT["kept_groups"] = pd.Series([grp[k] for k in keep]).value_counts().to_dict()
P2 = X[keep].corr().values.copy(); np.fill_diagonal(P2, 0); OUT["kept_max_abs_r"] = round(float(np.abs(P2).max()), 3); OUT["kept_pairs_080"] = int((np.abs(P2) > 0.8).sum() // 2)
json.dump(keep, open(f"{R}/secili_degiskenler.json", "w"))
print("çıkarılan:", steps); print("kalan", len(keep), "max VIF", v1.max().round(2), "max |r|", OUT["kept_max_abs_r"])

# 3 Mekânsal otokorelasyon (Moran's I, parsel ortalamaları, k=8 komşu)
def moran(xy, z, k=8, perm=199, seed=1):
    z = np.asarray(z, float); z = z - z.mean()
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy); _, idx = nn.kneighbors(xy); idx = idx[:, 1:]
    def I(zz): return (len(zz) / (len(zz) * k)) * np.sum(zz[:, None] * zz[idx]) / np.sum(zz ** 2)
    i0 = I(z); rng = np.random.default_rng(seed)
    sim = np.array([I(rng.permutation(z)) for _ in range(perm)])
    return float(i0), float((np.sum(sim >= i0) + 1) / (perm + 1)), float(-1 / (len(z) - 1))

xy = np.c_[(d.LON - 27) * 111.32 * np.cos(np.radians(41)), (d.LAT - 38) * 110.57]
d["X"], d["Y"] = xy[:, 0], xy[:, 1]
pm = d.groupby("PID").agg(X=("X", "first"), Y=("Y", "first"), dNDMI=("dNDMI", "mean"))
OUT["moran_dNDMI_parcelmean"] = moran(pm[["X", "Y"]].values, pm.dNDMI.values)
mor_y = {}
for yr in sorted(d.YIL.unique()):
    q = d[d.YIL == yr]; mor_y[int(yr)] = round(moran(q[["X", "Y"]].values, q.dNDMI.values, perm=99)[0], 3)
OUT["moran_by_year"] = mor_y
# korelogram (mesafe bandı), parsel ortalaması
from sklearn.neighbors import BallTree
bt = BallTree(pm[["X", "Y"]].values); z = pm.dNDMI.values - pm.dNDMI.mean(); bands = [0, 1, 2, 5, 10, 20, 50, 100]
cg = []
for lo, hi in zip(bands[:-1], bands[1:]):
    ind_hi = bt.query_radius(pm[["X", "Y"]].values, hi); ind_lo = bt.query_radius(pm[["X", "Y"]].values, lo) if lo > 0 else [np.array([i]) for i in range(len(z))]
    num = 0.0; W = 0
    for i in range(len(z)):
        nb = np.setdiff1d(ind_hi[i], ind_lo[i]); nb = nb[nb != i]
        num += z[i] * z[nb].sum(); W += len(nb)
    cg.append((f"{lo}-{hi}", round(len(z) / W * num / np.sum(z ** 2), 3) if W else np.nan))
OUT["moran_correlogram_km"] = cg
# RF artıklarında Moran's I (mekânsal CV tahminleri)
t = pd.read_csv(f"{R}/model_tahminleri.csv").merge(d[["PID", "YIL", "dNDMI", "X", "Y"]], on=["PID", "YIL"])
t["res"] = t.dNDMI - t.OOF_RF
rm = t.groupby("PID").agg(X=("X", "first"), Y=("Y", "first"), res=("res", "mean"))
OUT["moran_rf_residual_parcelmean"] = moran(rm[["X", "Y"]].values, rm.res.values)
# zamansal otokorelasyon: parsel içinde ardışık yıllar arası (gecikme 1) korelasyon
dd = d.sort_values(["PID", "YIL"]); dd["lag"] = dd.groupby("PID").dNDMI.shift(1)
OUT["temporal_lag1_r_raw"] = round(float(dd[["dNDMI", "lag"]].corr().iloc[0, 1]), 3)
dd["an"] = dd.dNDMI - dd.groupby("YIL").dNDMI.transform("mean"); dd["lag_an"] = dd.groupby("PID").an.shift(1)
OUT["temporal_lag1_r_yearremoved"] = round(float(dd[["an", "lag_an"]].corr().iloc[0, 1]), 3)
json.dump(OUT, open(f"{R}/kollinearite_ozet.json", "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps({k: v for k, v in OUT.items() if k not in ("target_pearson", "kept")}, ensure_ascii=False, default=str))

# Şekil: korelasyon matrisi (hücre içi sayı yok)
order = sorted(feats, key=lambda c: ["Climate", "Terrain", "Soil", "Stand"].index(grp[c]))
fig, ax = plt.subplots(figsize=(7.2, 6.6))
im = ax.imshow(P.loc[order, order].values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(order)), order, rotation=90, fontsize=6.5); ax.set_yticks(range(len(order)), order, fontsize=6.5)
gl = [grp[c] for c in order]
for k_ in range(1, len(order)):
    if gl[k_] != gl[k_ - 1]:
        ax.axhline(k_ - 0.5, color="k", lw=0.9); ax.axvline(k_ - 0.5, color="k", lw=0.9)
for k_, c in enumerate(order):
    if c not in keep:
        ax.get_xticklabels()[k_].set_color("#b4443a"); ax.get_yticklabels()[k_].set_color("#b4443a")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Pearson correlation coefficient")
fig.tight_layout(); fig.savefig(f"{O}/Fig_correlation.png", dpi=300, bbox_inches="tight"); plt.close(fig)
# VIF şekli
fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.6), gridspec_kw=dict(height_ratios=[1, 1], hspace=0.9))
ax = axs[0]
vv = v0.loc[order].clip(upper=1000)
ax.bar(range(len(order)), vv, color=["#2f6f4e" if c in keep else "#b4443a" for c in order])
ax.axhline(10, color="0.2", lw=0.8, ls="--")
ax.set_yscale("log"); ax.set_ylabel("VIF, 41 predictors")
ax.set_xticks(range(len(order)), order, rotation=90, fontsize=5.5); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("(a) Before selection", fontsize=8, loc="left")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2f6f4e", label="Retained"), Patch(color="#b4443a", label="Removed")],
          frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.25), ncol=2)
ax = axs[1]; ko = [c for c in order if c in keep]
ax.bar(range(len(ko)), v1.loc[ko], color="#2f6f4e"); ax.axhline(10, color="0.2", lw=0.8, ls="--")
ax.set_ylabel("VIF, retained predictors"); ax.set_xticks(range(len(ko)), ko, rotation=90, fontsize=5.5)
ax.spines[["top", "right"]].set_visible(False); ax.set_title("(b) After selection", fontsize=8, loc="left")
fig.savefig(f"{O}/Fig_vif.png", dpi=300, bbox_inches="tight"); plt.close(fig)
