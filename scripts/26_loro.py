"""Leave one regional directorate out: train on all other RDs, test on the held out RD."""
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
V, R3, O = sys.argv[1:4]
src = open("/home/claude/scripts/03_model_findik.py").read().split('if __name__ == "__main__":')[0]
ns = {"__file__": "x"}; exec(compile(src, "m03", "exec"), ns)
df, _ = ns["oku"](V); d = ns["hedef_hazirla"](df).dropna(subset=["dNDMI"]).reset_index(drop=True)
feats = json.load(open(f"{R3}/secili_degiskenler.json")); rows = []
for rd in ["SAKARYA", "ZONGULDAK", "GİRESUN", "TRABZON"]:
    tr, te = d[d.BOLGE_ADI != rd], d[d.BOLGE_ADI == rd]
    r = dict(RD=rd, n_train=len(tr), n_test=len(te))
    for m in ["RF", "CatBoost", "MLR"]:
        p = ns["fit_pred"](m, ns["modeller"]()[m], tr[feats], tr.dNDMI, te[feats])
        r[f"R2_{m}"] = r2_score(te.dNDMI, p); r[f"bias_{m}"] = float((p - te.dNDMI).mean())
        r[f"rho_{m}"] = float(np.mean([spearmanr(g.dNDMI, p[g.index - te.index[0]] if False else pd.Series(p, index=te.index)[g.index])[0] for _, g in te.groupby("YIL")]))
    b = te.YIL.map(tr.groupby("YIL").dNDMI.mean()); r["R2_base_year"] = r2_score(te.dNDMI, b)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True); rows.append(r)
pd.DataFrame(rows).to_csv(f"{O}/loro.csv", index=False)
