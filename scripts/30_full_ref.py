"""All stands: compare April to May, May to June and spring NDVI peak references (parcel means) under 10 and 20 km spatial CV, by RD, 2025 hold out and SHAP."""
import sys, json, numpy as np, pandas as pd, shap
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
V, R3, PR, O = sys.argv[1:5]; TG = sys.argv[5].split(",") if len(sys.argv) > 5 else ["AM", "MJ", "PK"]
src = open("/home/claude/scripts/03_model_findik.py").read().split('if __name__ == "__main__":')[0]
ns = {"__file__": "x"}; exec(compile(src, "m03", "exec"), ns)
df, _ = ns["oku"](V); d = ns["hedef_hazirla"](df).dropna(subset=["dNDMI"]).reset_index(drop=True)
d = d.merge(pd.read_csv(PR), on=["PID", "YIL"]).dropna(subset=["P"]).reset_index(drop=True)
d["AM"] = d.dNDMI; d["MJ"] = d.NDMI_JA - d.M / 1e4; d["PK"] = d.NDMI_JA - d.P / 1e4
x = (d.LON - 27) * 111.32 * np.cos(np.radians(41)); yy = (d.LAT - 38) * 110.57
d["B20"] = (x // 20).astype(int).astype(str) + "_" + (yy // 20).astype(int).astype(str)
feats = json.load(open(f"{R3}/secili_degiskenler.json")); MODELS = ["MLR", "RF", "LightGBM", "CatBoost"]
G = {"climate": [f for f in feats if any(k in f for k in ["P_", "PANOM", "CWB", "VPD", "T_JA", "TMAX", "TANOM", "SW", "LST"])]}
R = {"n": int(len(d)), "n_parcels": int(d.PID.nunique()), "corr_AM_PK": float(np.corrcoef(d.AM, d.PK)[0, 1]), "corr_AM_MJ": float(d[["AM", "MJ"]].corr().iloc[0, 1])}
main = ["SAKARYA", "ZONGULDAK", "GİRESUN", "TRABZON"]
def base(tr, te, t):
    m = d.iloc[tr].groupby(["YIL", "BOLGE_ADI"])[t].mean(); my = d.iloc[tr].groupby("YIL")[t].mean()
    k = pd.MultiIndex.from_frame(d.iloc[te][["YIL", "BOLGE_ADI"]])
    return pd.Series(m.reindex(k).values).fillna(pd.Series(d.iloc[te].YIL.map(my).values)).values
for t in TG:
    y = d[t].values; r = {"sd": float(d[t].std()), "mean": float(d[t].mean())}
    for bl in ["BLOK", "B20"]:
        b = np.zeros(len(d)); oof = {m: np.zeros(len(d)) for m in MODELS}
        for tr, te in GroupKFold(5).split(d, y, d[bl]):
            b[te] = base(tr, te, t)
            for m in MODELS: oof[m][te] = ns["fit_pred"](m, ns["modeller"]()[m], d.loc[tr, feats], d.loc[tr, t], d.loc[te, feats])
        k = "10" if bl == "BLOK" else "20"
        r[f"base_{k}"] = r2_score(y, b); r.update({f"{m}_{k}": r2_score(y, oof[m]) for m in MODELS})
        print(t, k, {m: round(r[f'{m}_{k}'], 3) for m in MODELS}, round(r[f"base_{k}"], 3), flush=True)
        if k == "10":
            for rd in main:
                mk = (d.BOLGE_ADI == rd).values
                r[f"RD_{rd}"] = [r2_score(y[mk], b[mk]), r2_score(y[mk], oof["RF"][mk])]
    tr = d.YIL != 2025; te = ~tr; ho = {}
    for m in ["RF", "CatBoost"]:
        p = ns["fit_pred"](m, ns["modeller"]()[m], d.loc[tr, feats], d.loc[tr, t], d.loc[te, feats]); yt = d.loc[te, t].values
        ho[m] = dict(R2=r2_score(yt, p), bias=float(p.mean() - yt.mean()), rho=float(spearmanr(yt, p)[0]), obs=float(yt.mean()))
    r["ho2025"] = ho; print(t, "2025", ho, flush=True)
    cb = ns["modeller"]()["CatBoost"]; cb.set_params(verbose=0); cb.fit(d[feats], d[t]); Xs = d[feats].sample(5000, random_state=42)
    imp = pd.Series(np.abs(shap.TreeExplainer(cb).shap_values(Xs)).mean(0), index=feats).sort_values(ascending=False)
    r["shap_top5"] = list(imp.index[:5]); r["elev_shap"] = float(imp["ELEV"]); r["elev_rank"] = int(list(imp.index).index("ELEV") + 1)
    r["elev_share"] = float(100 * imp["ELEV"] / imp.sum()); r["climate_share"] = float(100 * imp[G["climate"]].sum() / imp.sum())
    r["r_elev"] = float(np.corrcoef(d.ELEV, y)[0, 1]); print(t, r["shap_top5"], r["elev_rank"], round(r["elev_share"], 1), flush=True)
    R[t] = r; json.dump(R, open(f"{O}/full_ref_{len(d)}.json", "w"), indent=1)
print("DONE")
