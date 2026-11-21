"""Mean SHAP contribution of elevation by elevation band (April to May and peak references) against the median peak day of year."""
import json, numpy as np, pandas as pd, shap
V, R3, PR, DOY, O = "/home/claude/data/csv", "/home/claude/results3", "/home/claude/data/mj/pr_all.csv", "/home/claude/results10/peak_doy.csv", "/home/claude/results10"
src = open("/home/claude/scripts/03_model_findik.py").read().split('if __name__ == "__main__":')[0]
ns = {"__file__": "x"}; exec(compile(src, "m03", "exec"), ns)
df, _ = ns["oku"](V); d = ns["hedef_hazirla"](df).dropna(subset=["dNDMI"]).reset_index(drop=True)
d = d.merge(pd.read_csv(PR), on=["PID", "YIL"]).dropna(subset=["M", "P"]).reset_index(drop=True)
d["AM"] = d.dNDMI; d["PK"] = d.NDMI_JA - d.P / 1e4
feats = json.load(open(f"{R3}/secili_degiskenler.json")); bins = [-1, 100, 250, 400, 550, 700, 850, 2000]
out = {}
Xs = d.sample(6000, random_state=1)
for t in ["AM", "PK"]:
    cb = ns["modeller"]()["CatBoost"]; cb.set_params(verbose=0); cb.fit(d[feats], d[t])
    sv = shap.TreeExplainer(cb).shap_values(Xs[feats])[:, feats.index("ELEV")]
    out[t] = pd.Series(sv, index=Xs.index).groupby(pd.cut(Xs.ELEV, bins)).mean()
doy = pd.read_csv(DOY); doy["b"] = pd.cut(doy.ELEV, bins)
r = pd.DataFrame({"SHAP_AM": out["AM"], "SHAP_PK": out["PK"], "DOY_median": doy.groupby("b").DOY.median(),
                  "n": pd.cut(Xs.ELEV, bins).value_counts().sort_index()}).round(4)
print(r); r.index = r.index.astype(str); r.to_csv(f"{O}/elev_shap_doy.csv")
