"""Hakem yorumları için ek analizler (temel model, blok boyutu, LOYO, belirsizlik, duyarlılıklar)."""
import sys, os, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors

V, R3, O = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(O, exist_ok=True)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_model_findik.py")).read().split('if __name__ == "__main__":')[0]
ns = {"__file__": "x"}; exec(compile(src, "m03", "exec"), ns)
df, _ = ns["oku"](V); d = ns["hedef_hazirla"](df).dropna(subset=["dNDMI"]).reset_index(drop=True)
feats = json.load(open(f"{R3}/secili_degiskenler.json"))
y = d.dNDMI.values; OUT = {}
MOD = ["MLR", "SVR", "kNN", "MLP", "RF", "ET", "GBM", "XGBoost", "LightGBM", "CatBoost"]
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
def save():
    json.dump(OUT, open(f"{O}/review_checks.json", "w"), indent=1, default=float)

def base_pred(tr, te, keys, fallback_keys=None):
    m = d.iloc[tr].groupby(keys).dNDMI.mean().rename("p").reset_index()
    p = d.iloc[te][keys].merge(m, on=keys, how="left")["p"]
    if fallback_keys:
        m2 = d.iloc[tr].groupby(fallback_keys).dNDMI.mean().rename("p2").reset_index()
        p = p.fillna(d.iloc[te][fallback_keys].merge(m2, on=fallback_keys, how="left")["p2"])
    return p.fillna(d.iloc[tr].dNDMI.mean()).values

xy = np.c_[(d.LON - 27) * 111.32 * np.cos(np.radians(41)), (d.LAT - 38) * 110.57]
def blocks(km):
    return (np.floor(xy[:, 0] / km)).astype(int) * 10000 + (np.floor(xy[:, 1] / km)).astype(int)

# (a) temel modeller, mekânsal 10 km ve zamansal
base = {}
for keys, lab in [(["YIL"], "Year mean"), (["YIL", "BOLGE_ADI"], "Year x RD mean")]:
    o = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(d, y, d.BLOK):
        o[te] = base_pred(tr, te, keys, ["YIL"] if len(keys) > 1 else None)
    base[lab] = o
    OUT[f"baseline_spatial_{lab}"] = dict(R2=r2_score(y, o), RMSE=rmse(y, o))
for yr in [2026, 2025]:
    tr = np.where(d.YIL != yr)[0]; te = np.where(d.YIL == yr)[0]
    p = base_pred(tr, te, ["BOLGE_ADI"])
    OUT[f"baseline_temporal_{yr}_RD_climatology"] = dict(R2=r2_score(y[te], p), RMSE=rmse(y[te], p))
log("baseline", {k: v for k, v in OUT.items()}); save()

# (b) blok bootstrap güven aralıkları (10 km OOF)
t = pd.read_csv(f"{R3}/model_tahminleri.csv").merge(d[["PID", "YIL", "dNDMI", "BLOK"]], on=["PID", "YIL", "BLOK"])
dd = d[["PID", "YIL"]].copy(); dd["B_YEAR"] = base["Year mean"]; dd["B_YRD"] = base["Year x RD mean"]
t = t.merge(dd, on=["PID", "YIL"])
cols = {m: f"OOF_{m}" for m in MOD} | {"Year x RD mean": "B_YRD", "Year mean": "B_YEAR"}
bl = t.BLOK.unique(); g = {b: np.where(t.BLOK.values == b)[0] for b in bl}; rng = np.random.default_rng(1)
boot = {k: [] for k in cols}
for _ in range(500):
    s = rng.choice(bl, len(bl), replace=True); idx = np.concatenate([g[b] for b in s])
    yy = t.dNDMI.values[idx]
    for k, c in cols.items():
        boot[k].append((r2_score(yy, t[c].values[idx]), rmse(yy, t[c].values[idx])))
ci = {}
for k, c in cols.items():
    a = np.array(boot[k]); ci[k] = dict(R2=r2_score(t.dNDMI, t[c]), R2_lo=np.percentile(a[:, 0], 2.5), R2_hi=np.percentile(a[:, 0], 97.5),
                                       RMSE=rmse(t.dNDMI, t[c]), RMSE_lo=np.percentile(a[:, 1], 2.5), RMSE_hi=np.percentile(a[:, 1], 97.5))
OUT["bootstrap_ci"] = ci; pd.DataFrame(ci).T.round(4).to_csv(f"{O}/bootstrap_ci.csv"); log("bootstrap done"); save()

# (c) blok boyutu duyarlılığı
bs = {}
for km in [10, 20, 50]:
    b = blocks(km); row = {"n_blocks": int(len(set(b)))}
    o = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(d, y, b): o[te] = base_pred(tr, te, ["YIL", "BOLGE_ADI"], ["YIL"])
    row["Year x RD mean"] = r2_score(y, o)
    for mname in ["MLR", "LightGBM", "CatBoost", "RF"]:
        o = np.zeros(len(d))
        for tr, te in GroupKFold(5).split(d, y, b):
            m = ns["modeller"]()[mname]; m.fit(d.loc[tr, feats], y[tr]); o[te] = m.predict(d.loc[te, feats])
        row[mname] = r2_score(y, o); log("block", km, mname, round(row[mname], 3))
    bs[km] = row
OUT["block_size"] = bs; pd.DataFrame(bs).T.round(3).to_csv(f"{O}/block_size.csv"); save()

# (d) ERA5 hücre düzeyinde toplulaştırma
cell = (np.round(d.LON / 0.1)).astype(int) * 1000 + (np.round(d.LAT / 0.1)).astype(int)
d["CELL"] = cell; OUT["n_era5_cells"] = int(len(set(cell)))
OUT["parcels_per_cell_median"] = float(d.drop_duplicates("PID").groupby("CELL").size().median())
ca = d.groupby(["CELL", "YIL"]).agg({**{f: "mean" for f in feats}, "dNDMI": "mean", "BLOK": "first"}).reset_index()
o = np.zeros(len(ca))
for tr, te in GroupKFold(5).split(ca, ca.dNDMI, ca.BLOK):
    m = ns["modeller"]()["LightGBM"]; m.fit(ca.loc[tr, feats], ca.dNDMI[tr]); o[te] = m.predict(ca.loc[te, feats])
OUT["cell_level"] = dict(n=int(len(ca)), R2=r2_score(ca.dNDMI, o))
o = np.zeros(len(ca))
for tr, te in GroupKFold(5).split(ca, ca.dNDMI, ca.BLOK):
    m = ca.iloc[tr].groupby("YIL").dNDMI.mean(); o[te] = ca.iloc[te].YIL.map(m).values
OUT["cell_level_year_baseline_R2"] = r2_score(ca.dNDMI, o)
log("cell", OUT["cell_level"]); save()

# (e) kıyı hücreleri: en yakın geçerli parsel ile doldurma (LightGBM, 10 km)
clim = [c for c in ns["IKLIM"] if c in feats]
d2 = d.copy(); miss = d2[clim].isna().any(axis=1); OUT["climate_missing_share"] = float(miss.mean())
for yr in d2.YIL.unique():
    q = d2.YIL == yr; ok = q & ~miss; bad = q & miss
    if bad.sum() == 0: continue
    nn = NearestNeighbors(n_neighbors=1).fit(xy[ok.values]); _, ix = nn.kneighbors(xy[bad.values])
    d2.loc[bad, clim] = d2.loc[ok, clim].values[ix[:, 0]]
o1 = np.zeros(len(d)); o2 = np.zeros(len(d))
for tr, te in GroupKFold(5).split(d, y, d.BLOK):
    m = ns["modeller"]()["LightGBM"]; m.fit(d.loc[tr, feats], y[tr]); o1[te] = m.predict(d.loc[te, feats])
    m = ns["modeller"]()["LightGBM"]; m.fit(d2.loc[tr, feats], y[tr]); o2[te] = m.predict(d2.loc[te, feats])
OUT["imputation"] = dict(median_R2=r2_score(y, o1), nearest_R2=r2_score(y, o2)); log("imp", OUT["imputation"]); save()

# (f) parsel ortalamalarında VIF
allf = [c for c in ns["IKLIM"] + ns["TOPO"] + ns["TOPRAK"] + ns["MESCERE"] if c in d.columns]
pm = d.groupby("PID")[allf].mean(); pm = pm.fillna(pm.median())
def vif(cols):
    c = np.corrcoef(pm[cols].values, rowvar=False); return pd.Series(np.diag(np.linalg.pinv(c)), index=cols)
v41 = vif(allf); v29 = vif(feats)
OUT["vif_parcelmeans"] = dict(n_gt10_all=int((v41 > 10).sum()), max_retained=float(v29.max()), argmax_retained=v29.idxmax(),
                              n_gt10_retained=int((v29 > 10).sum()), retained_gt10=list(v29[v29 > 10].index))
log("vif", OUT["vif_parcelmeans"]); save()

# (g) stres tipi duyarlılığı ve boş model
s = d[["PID", "YIL", "BLOK", "DSI", "NDRE_JA", "NDMI_JA"]].copy(); gr = s.groupby("PID")
s["NDRE_z"] = (s.NDRE_JA - gr.NDRE_JA.transform("mean")) / gr.NDRE_JA.transform("std")
s["NDMI_z"] = (s.NDMI_JA - gr.NDMI_JA.transform("mean")) / gr.NDMI_JA.transform("std")
s["ST"] = s.DSI <= -1; st = s[s.ST]
sens = {}
for th in [-0.5, -1.0, -1.5]:
    obs = float(((st.NDRE_z - st.NDMI_z) <= th).mean())
    nulls = []
    for k in range(50):
        sh = s.groupby("PID").NDRE_z.transform(lambda v: v.sample(frac=1, random_state=k).values)
        nulls.append(float(((sh[s.ST] - st.NDMI_z) <= th).mean()))
    sens[th] = dict(observed=obs, null_mean=float(np.mean(nulls)), null_p95=float(np.percentile(nulls, 95)))
OUT["biotic_threshold"] = sens
s["BS"] = s.groupby(["BLOK", "YIL"]).ST.transform("mean")
OUT["isolation_threshold"] = {th: float((st.index.isin(s.index[s.ST & (s.BS < th)])).mean()) for th in [0.1, 0.2, 0.3]}
OUT["corr_NDREz_NDMIz_stressed"] = float(st[["NDRE_z", "NDMI_z"]].corr().iloc[0, 1])
log("stress", sens, OUT["isolation_threshold"]); save()

# (h) DSI: bir yılı dışarıda bırakarak standartlaştırma
g2 = d.groupby("PID").dNDMI; n = g2.transform("count"); sm = g2.transform("sum"); ss = g2.transform(lambda v: (v ** 2).sum())
mu = (sm - d.dNDMI) / (n - 1); var = ((ss - d.dNDMI ** 2) - (n - 1) * mu ** 2) / (n - 2)
d["DSI_LOYO"] = (d.dNDMI - mu) / np.sqrt(var)
OUT["dsi_share_by_year"] = {"standard": d.groupby("YIL").DSI.apply(lambda v: (v <= -1).mean()).round(3).to_dict(),
                            "loyo": d.groupby("YIL").DSI_LOYO.apply(lambda v: (v <= -1).mean()).round(3).to_dict()}
OUT["dsi_share_all"] = dict(standard=float((d.DSI <= -1).mean()), loyo=float((d.DSI_LOYO <= -1).mean()))
log("dsi", OUT["dsi_share_by_year"]); save()

# (i) yılı dışarıda bırakan zamansal doğrulama (LOYO), tüm modeller
loyo = {}
for yr in sorted(d.YIL.unique()):
    tr = np.where(d.YIL != yr)[0]; te = np.where(d.YIL == yr)[0]
    row = {"Climatology (RD mean)": r2_score(y[te], base_pred(tr, te, ["BOLGE_ADI"]))}
    for mname in MOD:
        p = ns["fit_pred"](mname, ns["modeller"]()[mname], d.loc[tr, feats], d.dNDMI.iloc[tr], d.loc[te, feats])
        row[mname] = r2_score(y[te], p); row[mname + "_bias"] = float(np.mean(p) - np.mean(y[te]))
    loyo[int(yr)] = row; log("loyo", yr, {k: round(v, 3) for k, v in row.items() if not k.endswith("bias")})
    OUT["loyo"] = loyo; save(); pd.DataFrame(loyo).T.round(3).to_csv(f"{O}/loyo.csv")
log("DONE")
