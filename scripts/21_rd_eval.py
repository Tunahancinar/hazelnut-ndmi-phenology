import json, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
src = open("/home/claude/scripts/03_model_findik.py").read().split('if __name__ == "__main__":')[0]
ns = {"__file__": "x"}; exec(compile(src, "m03", "exec"), ns)
df, _ = ns["oku"]("/home/claude/data/csv"); d = ns["hedef_hazirla"](df).dropna(subset=["dNDMI"]).reset_index(drop=True)
mt = pd.read_csv("/home/claude/results3/model_tahminleri.csv")
d = d.merge(mt, on=["PID", "YIL", "BLOK"], how="inner")
y = d.dNDMI.values; base = np.zeros(len(d))
for tr, te in GroupKFold(5).split(d, y, d.BLOK):
    m = d.iloc[tr].groupby(["YIL", "BOLGE_ADI"]).dNDMI.mean(); my = d.iloc[tr].groupby("YIL").dNDMI.mean()
    k = pd.MultiIndex.from_frame(d.iloc[te][["YIL", "BOLGE_ADI"]])
    base[te] = pd.Series(m.reindex(k).values).fillna(pd.Series(d.iloc[te].YIL.map(my).values)).values
d["BASE"] = base
print("overall", r2_score(y, d.OOF_RF), r2_score(y, base))
rows = []
for rd, g in d.groupby("BOLGE_ADI"):
    if g.PID.nunique() < 100: continue
    g25 = g[g.YIL == 2025]
    rows.append(dict(RD=rd, parcels=g.PID.nunique(), py=len(g), R2_RF=r2_score(g.dNDMI, g.OOF_RF), R2_CatBoost=r2_score(g.dNDMI, g.OOF_CatBoost),
                     R2_MLR=r2_score(g.dNDMI, g.OOF_MLR), R2_base=r2_score(g.dNDMI, g.BASE),
                     RMSE_RF=float(np.sqrt(np.mean((g.dNDMI - g.OOF_RF) ** 2))),
                     obs2025=g25.dNDMI.mean(), pred2025_CB=g25.T2025_CatBoost.mean(), bias2025_CB=(g25.T2025_CatBoost - g25.dNDMI).mean(),
                     rho2025_CB=g25[["dNDMI", "T2025_CatBoost"]].corr("spearman").iloc[0, 1]))
t = pd.DataFrame(rows); t["gain"] = t.R2_RF - t.R2_base
print(t.round(3).to_string()); t.to_csv("/home/claude/results8/rd_eval.csv", index=False)
# Sakarya dışı birleşik
o = d[d.BOLGE_ADI != "SAKARYA"]
print("non-Sakarya", len(o), r2_score(o.dNDMI, o.OOF_RF), r2_score(o.dNDMI, o.BASE))
json.dump({"nonSak_RF": r2_score(o.dNDMI, o.OOF_RF), "nonSak_base": r2_score(o.dNDMI, o.BASE), "nonSak_n": int(o.PID.nunique())}, open("/home/claude/results8/rd_extra.json", "w"))
