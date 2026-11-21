"""Merkez pikselde üç referansın adil karşılaştırması: Nisan-Mayıs medyanı, Mayıs-Haziran medyanı, ilkbahar NDVI tepe tarihi."""
import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
V, R3, PR, O = sys.argv[1:5]
src = open("/home/claude/scripts/03_model_findik.py").read().split('if __name__ == "__main__":')[0]
ns = {"__file__": "x"}; exec(compile(src, "m03", "exec"), ns)
df, _ = ns["oku"](V); d = ns["hedef_hazirla"](df).dropna(subset=["dNDMI"]).reset_index(drop=True)
c = pd.read_csv(PR)
d = d.merge(c, on=["PID", "YIL"], how="inner").dropna(subset=["M", "P"]).reset_index(drop=True)
d["dMJp"] = d.NDMI_JA - d.M / 1e4; d["dPKp"] = d.NDMI_JA - d.P / 1e4
feats = json.load(open(f"{R3}/secili_degiskenler.json")); X = d[feats]
T = {"dNDMI": "April to May", "dMJp": "May to June", "dPKp": "Spring NDVI peak"}
R = {"n": int(len(d)), "parcels": int(d.PID.nunique()), "r_AM_PK": float(np.corrcoef(d.dNDMI, d.dPKp)[0, 1]), "r_AM_MJ": float(np.corrcoef(d.dNDMI, d.dMJp)[0, 1])}
for t in T:
    y = d[t].values; R[t] = {"mean": float(y.mean()), "r_elev": float(np.corrcoef(d.ELEV, y)[0, 1])}
    for name in ["RF", "CatBoost", "MLR"]:
        o = np.zeros(len(d))
        for tr, te in GroupKFold(5).split(X, y, d.BLOK):
            o[te] = ns["fit_pred"](name, ns["modeller"]()[name], X.iloc[tr], d[t].iloc[tr], X.iloc[te])
        R[t][name] = r2_score(y, o)
    o = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(X, y, d.BLOK):
        m = d.iloc[tr].groupby(["YIL", "BOLGE_ADI"])[t].mean(); my = d.iloc[tr].groupby("YIL")[t].mean()
        k = pd.MultiIndex.from_frame(d.iloc[te][["YIL", "BOLGE_ADI"]])
        o[te] = pd.Series(m.reindex(k).values).fillna(pd.Series(d.iloc[te].YIL.map(my).values)).values
    R[t]["baseline"] = r2_score(y, o); print(t, {k: round(v, 3) for k, v in R[t].items()}, flush=True)
import shap
grp = {v: "Climate" for v in ns["IKLIM"]} | {v: "Terrain" for v in ns["TOPO"]} | {v: "Soil" for v in ns["TOPRAK"]} | {v: "Stand" for v in ns["MESCERE"]}
Xs = X.sample(5000, random_state=42); SV = {}
for t in T:
    cb = ns["modeller"]()["CatBoost"]; cb.fit(X, d[t]); sv = shap.TreeExplainer(cb).shap_values(Xs); SV[t] = sv
    imp = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False); g = imp.groupby(imp.index.map(grp)).sum()
    R[t].update(elev_shap=float(imp["ELEV"]), elev_rank=int(list(imp.index).index("ELEV") + 1), top3=list(imp.index[:3]),
                climate_share=float(g["Climate"] / g.sum()), panom_shap=float(imp["PANOM_AMJJA"]))
json.dump(R, open(f"{O}/ref_compare_parcel.json", "w"), indent=1); print(json.dumps(R, indent=1))
plt.rcParams.update({"font.size": 8})
fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.4), sharey=True)
j = feats.index("ELEV")
for a, (t, lab), l in zip(ax, T.items(), "abc"):
    a.scatter(Xs.ELEV, SV[t][:, j], s=1.5, alpha=0.3, color="#2f6f4e", edgecolors="none")
    a.axhline(0, color="0.5", lw=0.6); a.set_xlabel("ELEV (m)"); a.set_title(f"({l}) {lab}", loc="left", fontsize=7)
ax[0].set_ylabel("SHAP value of ELEV")
fig.tight_layout(); fig.savefig("/home/claude/figs3/Fig_ref_compare_parcel.png", dpi=300, bbox_inches="tight"); print("DONE")
