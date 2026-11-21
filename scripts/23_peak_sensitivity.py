"""Sensitivity of the spring NDVI peak reference: all parcel years, suspicious peaks removed, and the mean of May to June and peak references."""
import sys, json, numpy as np, pandas as pd, shap
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
V, R3, PR, O = sys.argv[1:5]
src = open("/home/claude/scripts/03_model_findik.py").read().split('if __name__ == "__main__":')[0]
ns = {"__file__": "x"}; exec(compile(src, "m03", "exec"), ns)
df, _ = ns["oku"](V); d = ns["hedef_hazirla"](df).dropna(subset=["dNDMI"]).reset_index(drop=True)
d = d.merge(pd.read_csv(PR), on=["PID", "YIL"]).dropna(subset=["M", "P"]).reset_index(drop=True)
d["AMv"] = d.NDMI_JA - d.dNDMI; d["Mv"] = d.M / 1e4; d["Pv"] = d.P / 1e4
d["dPK"] = d.NDMI_JA - d.Pv; d["dCMB"] = d.NDMI_JA - (d.Mv + d.Pv) / 2
bad = (d.Pv < np.minimum(d.AMv, d.Mv) - 0.05) | (d.Pv < 0.1)
feats = json.load(open(f"{R3}/secili_degiskenler.json"))
def run(dd, t):
    X, y = dd[feats], dd[t].values; o = np.zeros(len(dd)); b = np.zeros(len(dd))
    for tr, te in GroupKFold(5).split(X, y, dd.BLOK):
        o[te] = ns["fit_pred"]("RF", ns["modeller"]()["RF"], X.iloc[tr], dd[t].iloc[tr], X.iloc[te])
        m = dd.iloc[tr].groupby(["YIL", "BOLGE_ADI"])[t].mean(); my = dd.iloc[tr].groupby("YIL")[t].mean()
        k = pd.MultiIndex.from_frame(dd.iloc[te][["YIL", "BOLGE_ADI"]])
        b[te] = pd.Series(m.reindex(k).values).fillna(pd.Series(dd.iloc[te].YIL.map(my).values)).values
    cb = ns["modeller"]()["CatBoost"]; cb.fit(X, dd[t]); Xs = X.sample(5000, random_state=42)
    imp = pd.Series(np.abs(shap.TreeExplainer(cb).shap_values(Xs)).mean(0), index=feats).sort_values(ascending=False)
    r = dict(n=int(len(dd)), RF=r2_score(y, o), baseline=r2_score(y, b), elev_shap=float(imp["ELEV"]),
             elev_rank=int(list(imp.index).index("ELEV") + 1), panom_rank=int(list(imp.index).index("PANOM_AMJJA") + 1),
             r_elev=float(np.corrcoef(dd.ELEV, y)[0, 1]))
    r["gain"] = r["RF"] - r["baseline"]; print(t, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True); return r
R = {"share_removed": float(bad.mean()), "n_removed": int(bad.sum()),
     "peak_all": run(d, "dPK"), "peak_filtered": run(d[~bad].reset_index(drop=True), "dPK"),
     "am_filtered": run(d[~bad].reset_index(drop=True), "dNDMI"), "combined": run(d, "dCMB")}
json.dump(R, open(f"{O}/peak_sensitivity.json", "w"), indent=1); print("DONE", R["share_removed"])
