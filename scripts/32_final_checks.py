"""Final reviewer checks: buffered and ERA5 cell grouped spatial CV for two references, temporal tests with past only training and in-sample-free variable selection, 2025 spring and summer components, peak date diagnostics and parcel size sensitivity."""
import sys, json, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
V, R3, PR, DOY, O = sys.argv[1:6]; PART = sys.argv[6] if len(sys.argv) > 6 else "all"
src = open("/home/claude/scripts/03_model_findik.py").read().split('if __name__ == "__main__":')[0]
ns = {"__file__": "x"}; exec(compile(src, "m03", "exec"), ns)
df, _ = ns["oku"](V); d = ns["hedef_hazirla"](df).dropna(subset=["dNDMI"]).reset_index(drop=True)
d = d.merge(pd.read_csv(PR), on=["PID", "YIL"], how="left")
d["AM"] = d.dNDMI; d["PK"] = d.NDMI_JA - d.P / 1e4; d["SPR_AM"] = d.NDMI_JA - d.dNDMI; d["SPR_PK"] = d.P / 1e4
feats = json.load(open(f"{R3}/secili_degiskenler.json")); allf = [c for c in ns["IKLIM"] + ns["TOPO"] + ns["TOPRAK"] + ns["MESCERE"] if c in d.columns]
x = (d.LON - 27) * 111.32 * np.cos(np.radians(41)); yk = (d.LAT - 38) * 110.57; XY = np.c_[x, yk]
d["CELL"] = (np.floor(d.LON / 0.1)).astype(int).astype(str) + "_" + (np.floor(d.LAT / 0.1)).astype(int).astype(str)
R = {}
def save(): json.dump(R, open(f"{O}/final_checks_{PART}.json", "w"), indent=1, default=float)
def base(tr, te, t):
    m = d.iloc[tr].groupby(["YIL", "BOLGE_ADI"])[t].mean(); my = d.iloc[tr].groupby("YIL")[t].mean()
    k = pd.MultiIndex.from_frame(d.iloc[te][["YIL", "BOLGE_ADI"]])
    return pd.Series(m.reindex(k).values).fillna(pd.Series(d.iloc[te].YIL.map(my).values)).values
if PART in ("all", "spatial"):
    for t in ["AM", "PK"]:
        y = d[t].values
        for scheme in ["buffer10", "cell"]:
            g = d.BLOK if scheme == "buffer10" else d.CELL
            b = np.zeros(len(d)); o = np.zeros(len(d)); ntr = []
            for tr, te in GroupKFold(5).split(d, y, g):
                if scheme == "buffer10":
                    dist, _ = cKDTree(XY[te]).query(XY[tr], k=1); tr = tr[dist > 10]
                ntr.append(len(tr)); b[te] = base(tr, te, t)
                o[te] = ns["fit_pred"]("RF", ns["modeller"]()["RF"], d.loc[tr, feats], d.loc[tr, t], d.loc[te, feats])
            R[f"{t}_{scheme}"] = dict(RF=r2_score(y, o), base=r2_score(y, b), n_train_mean=float(np.mean(ntr)))
            if t == "AM" and scheme == "buffer10":
                d["OOF_B"] = o
            print(t, scheme, R[f"{t}_{scheme}"], flush=True); save()
        if t == "AM":
            o = np.zeros(len(d))
            for tr, te in GroupKFold(5).split(d, y, d.BLOK):
                o[te] = ns["fit_pred"]("RF", ns["modeller"]()["RF"], d.loc[tr, feats], d.loc[tr, t], d.loc[te, feats])
            d["OOF"] = o
            # parcel size sensitivity
            d["SZ"] = pd.cut(d.AREA_HA, [0, 0.5, 1, 2, 5, 1000], labels=["<0.5", "0.5-1", "1-2", "2-5", ">5"])
            d["BLKMEAN"] = d.groupby(["YIL", "BLOK"]).AM.transform("mean")
            sz = {}
            for k, gg in d.groupby("SZ", observed=True):
                sz[str(k)] = dict(n=int(len(gg)), r2=r2_score(gg.AM, gg.OOF), rmse=float(np.sqrt(((gg.AM - gg.OOF) ** 2).mean())),
                                  sd=float(gg.AM.std()), r_block=float(np.corrcoef(gg.AM, gg.BLKMEAN)[0, 1]),
                                  sd_resid_block=float((gg.AM - gg.BLKMEAN).std()))
            R["size"] = sz; print(sz, flush=True); save()
if PART in ("all", "temporal"):
    def select(dt, tcol):
        X = dt[allf].fillna(dt[allf].median())
        def vif(cols):
            c = np.corrcoef(X[cols].values, rowvar=False); return pd.Series(np.diag(np.linalg.pinv(c)), index=cols)
        keep = list(allf)
        while True:
            v = vif(keep)
            if v.max() <= 10: break
            keep.remove(v.idxmax())
        tr_ = dt[allf + [tcol]].corr(method="spearman")[tcol].abs()
        while True:
            C = X[keep].corr().abs().values.copy(); np.fill_diagonal(C, 0)
            if C.max() <= 0.80: break
            i, j = np.unravel_index(C.argmax(), C.shape); a, bb = keep[i], keep[j]
            keep.remove(a if tr_[a] < tr_[bb] else bb)
        return keep
    T = {}
    for name, trm, tem in [("2025_past_only", d.YIL <= 2024, d.YIL == 2025), ("2025_all_other", d.YIL != 2025, d.YIL == 2025), ("2026", d.YIL <= 2025, d.YIL == 2026)]:
        tr, te = d[trm], d[tem]; ks = select(tr, "AM"); r = {"n_sel": len(ks), "jaccard": len(set(ks) & set(feats)) / len(set(ks) | set(feats))}
        for m in ["MLR", "RF", "LightGBM", "CatBoost"]:
            for lab, fs in [("global", feats), ("train_sel", ks)]:
                p = ns["fit_pred"](m, ns["modeller"]()[m], tr[fs], tr.AM, te[fs])
                r[f"{m}_{lab}"] = dict(R2=r2_score(te.AM, p), bias=float(p.mean() - te.AM.mean()), rho=float(spearmanr(te.AM, p)[0]))
        T[name] = r; print(name, {k: (round(v["R2"], 3) if isinstance(v, dict) else v) for k, v in r.items()}, flush=True)
        R["temporal"] = T; save()
if PART in ("all", "comp"):
    west = d.BOLGE_ADI.isin(["SAKARYA", "ZONGULDAK"]); east = d.BOLGE_ADI.isin(["GİRESUN", "TRABZON"])
    cols = ["SPR_AM", "SPR_PK", "NDMI_JA", "AM", "PK"]
    comp = {}
    for nm, mk in [("west", west), ("east", east)]:
        g = d[mk].dropna(subset=["P"])
        comp[nm] = {"2025": g[g.YIL == 2025][cols].mean().to_dict(), "other": g[g.YIL != 2025][cols].mean().to_dict()}
    R["components_2025"] = comp; print(json.dumps(comp, indent=1), flush=True)
    dy = pd.read_csv(DOY).merge(d[["PID", "YIL", "BOLGE_ADI"]], on=["PID", "YIL"], how="left")
    rows = []
    for (yy, rd), g in dy.dropna(subset=["BOLGE_ADI"]).groupby(["YIL", "BOLGE_ADI"]):
        if len(g) >= 30: rows.append(dict(YIL=int(yy), RD=rd, n=len(g), rho=float(spearmanr(g.ELEV, g.DOY)[0]), slope=float(np.polyfit(g.ELEV, g.DOY, 1)[0] * 100)))
    rr = pd.DataFrame(rows); R["doy_within"] = rows
    R["doy_edge"] = dict(early=float((dy.DOY <= 97).mean()), late=float((dy.DOY >= 175).mean()),
                         early_by_year=dy.groupby("YIL").DOY.apply(lambda s: float((s <= 97).mean())).to_dict(),
                         late_by_year=dy.groupby("YIL").DOY.apply(lambda s: float((s >= 175).mean())).to_dict())
    print(rr.round(3).to_string(), R["doy_edge"], flush=True); save()
print("DONE")
