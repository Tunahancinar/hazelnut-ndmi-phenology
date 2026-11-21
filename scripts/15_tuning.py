"""İç içe mekânsal çapraz doğrulama ile hiperparametre ayarı (hakem yorumu 9)."""
import sys, os, json, time, itertools
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.base import clone

V, R3, O = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_model_findik.py")).read().split('if __name__ == "__main__":')[0]
ns = {"__file__": "x"}; exec(compile(src, "m03", "exec"), ns)
df, _ = ns["oku"](V); d = ns["hedef_hazirla"](df).dropna(subset=["dNDMI"]).reset_index(drop=True)
feats = json.load(open(f"{R3}/secili_degiskenler.json")); X = d[feats]; y = d.dNDMI.values; G = d.BLOK.values
base = ns["modeller"]()
def setp(m, **kw):
    m = clone(m)
    est = m.steps[-1][0] if hasattr(m, "steps") else None
    m.set_params(**({f"{est}__{k}": v for k, v in kw.items()} if est else kw)); return m
grids = {
 "SVR": [dict(C=c, gamma=g) for c in (0.3, 1, 3) for g in ("scale", 0.01)],
 "kNN": [dict(n_neighbors=k) for k in (5, 15, 30, 50)],
 "MLP": [dict(hidden_layer_sizes=h, alpha=a) for h in ((64, 32), (128, 64)) for a in (1e-4, 1e-2)],
 "RF": [dict(max_features=f, min_samples_leaf=l) for f, l in ((0.2, 3), (0.4, 3), (0.6, 3), (0.4, 10))],
 "ET": [dict(max_features=f, min_samples_leaf=l) for f, l in ((0.3, 3), (0.5, 3), (0.8, 3), (0.5, 10))],
 "GBM": [dict(learning_rate=lr, max_leaf_nodes=n) for lr in (0.03, 0.1) for n in (15, 63)],
 "XGBoost": [dict(learning_rate=lr, max_depth=dp, n_estimators=800 if lr < 0.05 else 300) for lr in (0.03, 0.1) for dp in (4, 8)],
 "LightGBM": [dict(learning_rate=lr, num_leaves=n, n_estimators=800 if lr < 0.05 else 300) for lr in (0.03, 0.1) for n in (15, 63)],
 "CatBoost": [dict(learning_rate=lr, depth=dp) for lr in (0.05, 0.1) for dp in (4, 8)],
}
inner_fast = {"RF": dict(n_estimators=100), "ET": dict(n_estimators=100), "CatBoost": dict(iterations=400)}
rng = np.random.default_rng(42); res = {}; chosen = {}
for name in ["MLR"] + list(grids):
    t0 = time.time(); oof = np.zeros(len(d)); picks = []
    for tr, te in GroupKFold(5).split(X, y, G):
        if name == "MLR":
            best = base["MLR"]
        else:
            cap = 6000 if name == "SVR" else 15000
            sub = tr if len(tr) <= cap else rng.choice(tr, cap, replace=False)
            scores = []
            for p in grids[name]:
                sc = []
                for itr, ite in GroupKFold(3).split(sub, y[sub], G[sub]):
                    m = setp(base[name], **p, **inner_fast.get(name, {}))
                    m.fit(X.iloc[sub[itr]], y[sub[itr]]); sc.append(r2_score(y[sub[ite]], m.predict(X.iloc[sub[ite]])))
                scores.append(np.mean(sc))
            bp = grids[name][int(np.argmax(scores))]; picks.append(bp); best = setp(base[name], **bp)
        oof[te] = ns["fit_pred"](name, best, X.iloc[tr], d.dNDMI.iloc[tr], X.iloc[te])
    res[name] = r2_score(y, oof); chosen[name] = [str(p) for p in picks]
    print(time.strftime("%H:%M:%S"), name, round(res[name], 4), picks[:1], round(time.time() - t0), "s", flush=True)
    json.dump(dict(tuned_spatial_R2=res, chosen=chosen), open(f"{O}/tuning.json", "w"), indent=1)
print("DONE")
