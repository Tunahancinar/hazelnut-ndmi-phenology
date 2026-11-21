"""Robustness of annual and RD-year climate relations: with and without 2025, leave one year out range, and three spring references on the parcel subsample."""
import sys, json, numpy as np, pandas as pd
from scipy.stats import pearsonr
V, PR, O = sys.argv[1:4]
src = open("/home/claude/scripts/03_model_findik.py").read().split('if __name__ == "__main__":')[0]
ns = {"__file__": "x"}; exec(compile(src, "m03", "exec"), ns)
df, _ = ns["oku"](V); d = ns["hedef_hazirla"](df).dropna(subset=["dNDMI"]).reset_index(drop=True)
C = ["CWB_AMJJA", "PANOM_AMJJA", "VPD_JA", "SWANOM_JA", "TANOM_JA"]
def rel(g, y):
    out = {}
    for c in C:
        r_all = pearsonr(g[y], g[c])[0]; m = g.YIL != 2025; r_no = pearsonr(g[y][m], g[c][m])[0]
        loo = [pearsonr(g[y][g.YIL != k], g[c][g.YIL != k])[0] for k in sorted(g.YIL.unique())]
        out[c] = dict(all=round(r_all, 3), no2025=round(r_no, 3), loo_min=round(min(loo), 3), loo_max=round(max(loo), 3), n=int(len(g)))
    return out
R = {}
a = d.groupby("YIL")[["dNDMI"] + C].mean().reset_index(); R["annual"] = rel(a, "dNDMI")
main = ["SAKARYA", "ZONGULDAK", "GİRESUN", "TRABZON"]
print(d.BOLGE_ADI.unique())
dr = d[d.BOLGE_ADI.isin(main)]; b = dr.groupby(["BOLGE_ADI", "YIL"])[["dNDMI"] + C].mean().reset_index(); R["rd_year"] = rel(b, "dNDMI")
# parcel subsample, three references
s = d.merge(pd.read_csv(PR), on=["PID", "YIL"]).dropna(subset=["M", "P"])
s["AM"] = s.dNDMI; s["MJ"] = s.NDMI_JA - s.M / 1e4; s["PK"] = s.NDMI_JA - s.P / 1e4
for t in ["AM", "MJ", "PK"]:
    g = s.groupby("YIL")[[t] + C].mean().reset_index(); R[f"sub_annual_{t}"] = rel(g, t)
    g = s[s.BOLGE_ADI.isin(main)].groupby(["BOLGE_ADI", "YIL"])[[t] + C].mean().reset_index(); R[f"sub_rd_{t}"] = rel(g, t)
json.dump(R, open(f"{O}/climate_robust.json", "w"), indent=1)
for k, v in R.items(): print(k, {c: (v[c]["all"], v[c]["no2025"], v[c]["loo_min"], v[c]["loo_max"]) for c in ["CWB_AMJJA", "PANOM_AMJJA", "VPD_JA", "SWANOM_JA"]})
