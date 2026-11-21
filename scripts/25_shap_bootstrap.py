"""Normalized SHAP importance of elevation for three spring references and a 10 km block bootstrap of its change."""
import sys, json, time, numpy as np, pandas as pd, shap
V, R3, PR, O, B = sys.argv[1:6]; B = int(B)
src = open("/home/claude/scripts/03_model_findik.py").read().split('if __name__ == "__main__":')[0]
ns = {"__file__": "x"}; exec(compile(src, "m03", "exec"), ns)
df, _ = ns["oku"](V); d = ns["hedef_hazirla"](df).dropna(subset=["dNDMI"]).reset_index(drop=True)
d = d.merge(pd.read_csv(PR), on=["PID", "YIL"]).dropna(subset=["M", "P"]).reset_index(drop=True)
d["AM"] = d.dNDMI; d["MJ"] = d.NDMI_JA - d.M / 1e4; d["PK"] = d.NDMI_JA - d.P / 1e4
feats = json.load(open(f"{R3}/secili_degiskenler.json")); T = ["AM", "MJ", "PK"]
def imp(dd, t, seed):
    cb = ns["modeller"]()["CatBoost"]; cb.set_params(random_seed=seed, verbose=0); cb.fit(dd[feats], dd[t])
    Xs = dd[feats].sample(min(3000, len(dd)), random_state=seed); sv = np.abs(shap.TreeExplainer(cb).shap_values(Xs)).mean(0)
    s = pd.Series(sv, index=feats); sd = dd[t].std()
    return dict(elev=float(s["ELEV"]), share=float(100 * s["ELEV"] / s.sum()), rel_sd=float(s["ELEV"] / sd), rank=int((s > s["ELEV"]).sum() + 1))
R = {"full": {}}
for t in T:
    R["full"][t] = imp(d, t, 42) | {"sd": float(d[t].std())}; print(t, R["full"][t], flush=True)
blocks = d.BLOK.unique(); rng = np.random.default_rng(1); boot = []; t0 = time.time()
for b in range(B):
    pick = rng.choice(blocks, len(blocks), replace=True)
    dd = pd.concat([d[d.BLOK == k] for k in pick], ignore_index=True)
    boot.append({t: imp(dd, t, b) for t in T})
    if b == 0: print("sec per boot", time.time() - t0, flush=True)
def ci(x): return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
S = {}
for t in T:
    for k in ["elev", "share", "rel_sd"]: S[f"{t}_{k}"] = ci([x[t][k] for x in boot])
    S[f"{t}_rank"] = ci([x[t]["rank"] for x in boot])
for t in ["MJ", "PK"]:
    for k in ["elev", "share", "rel_sd"]:
        red = [100 * (1 - x[t][k] / x["AM"][k]) for x in boot]; S[f"reduction_{t}_{k}"] = ci(red) + [float(np.median(red))]
        S[f"p_not_lower_{t}_{k}"] = float(np.mean([x[t][k] >= x["AM"][k] for x in boot]))
R["boot"] = S; R["B"] = B; R["n_blocks"] = int(len(blocks))
json.dump(R, open(f"{O}/shap_bootstrap.json", "w"), indent=1); print(json.dumps(S, indent=1))
