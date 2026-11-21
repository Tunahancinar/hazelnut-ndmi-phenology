"""Değişken seçimini (korelasyon + VIF + hedefle ilişki) yalnızca eğitim katlarında tekrarlayan duyarlılık analizi."""
import sys, json, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
V, R3, O = sys.argv[1:4]
src = open("/home/claude/scripts/03_model_findik.py").read().split('if __name__ == "__main__":')[0]
ns = {"__file__": "x"}; exec(compile(src, "m03", "exec"), ns)
df, _ = ns["oku"](V); d = ns["hedef_hazirla"](df).dropna(subset=["dNDMI"]).reset_index(drop=True)
feats = [c for c in ns["IKLIM"] + ns["TOPO"] + ns["TOPRAK"] + ns["MESCERE"] if c in d.columns]
glob = json.load(open(f"{R3}/secili_degiskenler.json"))
def select(dt):
    X = dt[feats].fillna(dt[feats].median())
    def vif(cols):
        c = np.corrcoef(X[cols].values, rowvar=False); return pd.Series(np.diag(np.linalg.pinv(c)), index=cols)
    keep = list(feats)
    while True:
        v = vif(keep)
        if v.max() <= 10: break
        keep.remove(v.idxmax())
    tr = dt[feats + ["dNDMI"]].corr(method="spearman")["dNDMI"].abs()
    while True:
        C = X[keep].corr().abs().values.copy(); np.fill_diagonal(C, 0)
        if C.max() <= 0.80: break
        i, j = np.unravel_index(C.argmax(), C.shape); a, b = keep[i], keep[j]
        keep.remove(a if tr[a] < tr[b] else b)
    return keep
y = d.dNDMI.values; models = ["MLR", "RF", "LightGBM", "CatBoost"]
OOF = {m: np.zeros(len(d)) for m in models}; sets = []
for k, (tr, te) in enumerate(GroupKFold(5).split(d, y, d.BLOK)):
    ks = select(d.iloc[tr]); sets.append(ks)
    print("fold", k, len(ks), "jaccard", round(len(set(ks) & set(glob)) / len(set(ks) | set(glob)), 3), flush=True)
    for m in models:
        OOF[m][te] = ns["fit_pred"](m, ns["modeller"]()[m], d.iloc[tr][ks], d.dNDMI.iloc[tr], d.iloc[te][ks])
    print({m: round(r2_score(y[te], OOF[m][te]), 3) for m in models}, flush=True)
freq = pd.Series([f for s in sets for f in s]).value_counts()
R = {"r2_nested": {m: r2_score(y, OOF[m]) for m in models}, "n_selected": [len(s) for s in sets],
     "jaccard_with_global": [len(set(s) & set(glob)) / len(set(s) | set(glob)) for s in sets],
     "always_selected": int((freq == 5).sum()), "selected_in_global_not_all_folds": [f for f in glob if freq.get(f, 0) < 5],
     "selected_in_folds_not_global": sorted(set(freq.index) - set(glob)), "freq": freq.to_dict()}
json.dump(R, open(f"{O}/nested_selection.json", "w"), indent=1); print(json.dumps(R, indent=1)); print("DONE")
