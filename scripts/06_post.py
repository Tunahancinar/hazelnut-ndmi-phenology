"""Model sonrası analizler: CatBoost SHAP, rastgele ve mekânsal CV farkı, 2025 sapması, stres tipi, şekiller."""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GroupKFold

veri, out = sys.argv[1], sys.argv[2]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_model_findik.py")).read()
src = src.split('if __name__ == "__main__":')[0]
ns = {"__file__": "x"}
exec(compile(src, "m03", "exec"), ns)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

df, ki = ns["oku"](veri)
df = ns["hedef_hazirla"](df)
feats = [c for c in ns["IKLIM"] + ns["TOPO"] + ns["TOPRAK"] + ns["MESCERE"] if c in df.columns]
if len(sys.argv) > 3:
    feats = json.load(open(sys.argv[3]))
d = df.dropna(subset=["dNDMI"]).reset_index(drop=True)
X, y = d[feats], d["dNDMI"]
R = {}

# 1 Rastgele ve mekânsal CV (LightGBM)
from sklearn.metrics import r2_score, mean_squared_error
oof_r = np.zeros(len(y)); oof_s = np.zeros(len(y))
for tr, te in KFold(5, shuffle=True, random_state=42).split(X):
    m = ns["modeller"]()["LightGBM"]; m.fit(X.iloc[tr], y.iloc[tr]); oof_r[te] = m.predict(X.iloc[te])
for tr, te in GroupKFold(5).split(X, y, d["BLOK"]):
    m = ns["modeller"]()["LightGBM"]; m.fit(X.iloc[tr], y.iloc[tr]); oof_s[te] = m.predict(X.iloc[te])
R["lgbm_random_cv_r2"] = r2_score(y, oof_r); R["lgbm_spatial_cv_r2"] = r2_score(y, oof_s)
R["lgbm_random_cv_rmse"] = float(np.sqrt(mean_squared_error(y, oof_r)))
print(R, flush=True)

# 2 2025 sapması (CatBoost, LightGBM, MLR, kNN)
bias = {}
tr5, te5 = d.YIL != 2025, d.YIL == 2025
for ad in ["MLR", "kNN", "RF", "LightGBM", "CatBoost"]:
    m = ns["modeller"]()[ad]
    m.fit(X[tr5], y[tr5])
    p = m.predict(X[te5])
    bias[ad] = {"obs_mean": float(y[te5].mean()), "pred_mean": float(p.mean()),
                "r_spearman": float(pd.Series(p).corr(y[te5].reset_index(drop=True), method="spearman"))}
R["bias_2025"] = bias
print(bias, flush=True)
# eğitim aralığı dışında kalan iklim
for v in ["PANOM_AMJJA", "SWANOM_JA", "VPD_JA", "CWB_AMJJA"]:
    R[f"oor_2025_{v}"] = float(((d.loc[te5, v] < d.loc[tr5, v].min()) | (d.loc[te5, v] > d.loc[tr5, v].max())).mean())

# 3 CatBoost SHAP
import shap
cb = ns["modeller"]()["CatBoost"]; cb.fit(X, y)
Xs = X.sample(5000, random_state=42)
sv = shap.TreeExplainer(cb).shap_values(Xs)
imp = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False)
imp.rename("mean_abs_SHAP").to_csv(os.path.join(out, "shap_importance_catboost.csv"))
grp = {v: "Climate" for v in ns["IKLIM"]} | {v: "Terrain" for v in ns["TOPO"]} | {v: "Soil" for v in ns["TOPRAK"]} | {v: "Stand" for v in ns["MESCERE"]}
gimp = imp.groupby(imp.index.map(grp)).sum(); R["shap_group_share"] = (gimp / gimp.sum()).round(4).to_dict()
R["shap_top15"] = imp.head(15).round(5).to_dict()
plt.figure()
shap.summary_plot(sv, Xs, max_display=15, show=False, plot_size=(6.5, 5.5))
plt.tight_layout(); plt.savefig(os.path.join(out, "Fig_shap_summary.png"), dpi=300); plt.close()
top = list(imp.index[:6])
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.8))
for a_, v in zip(axs.ravel(), top):
    j = feats.index(v)
    a_.scatter(Xs[v], sv[:, j], s=2, alpha=0.3, color="#2f6f4e", edgecolors="none")
    a_.axhline(0, color="0.5", lw=0.6); a_.set_xlabel(v, fontsize=8); a_.set_ylabel("SHAP value (ΔNDMI)", fontsize=8)
    a_.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(os.path.join(out, "Fig_shap_dependence.png"), dpi=300); plt.close(fig)

# 4 Yıllık hedef ve iklim
yillar = sorted(d.YIL.unique())
fig, ax = plt.subplots(2, 1, figsize=(6.5, 5.4), sharex=True)
ax[0].boxplot([d.loc[d.YIL == t, "dNDMI"] for t in yillar], positions=range(len(yillar)), widths=0.6, showfliers=False,
              patch_artist=True, boxprops=dict(facecolor="#cfe3d6", edgecolor="#2f6f4e"), medianprops=dict(color="#2f6f4e"))
ax[0].axhline(d.dNDMI.median(), color="0.5", lw=0.8, ls="--"); ax[0].set_ylabel("ΔNDMI (JA − AM)")
g = d.groupby("YIL")[["PANOM_AMJJA", "SWANOM_JA", "TANOM_JA"]].mean().loc[yillar]
ax[1].bar(np.arange(len(yillar)) - 0.2, g.PANOM_AMJJA, 0.4, color="#4c78a8", label="Precipitation anomaly, Apr–Aug")
ax[1].bar(np.arange(len(yillar)) + 0.2, g.SWANOM_JA, 0.4, color="#c9a227", label="Soil water anomaly, Jul–Aug")
ax[1].axhline(0, color="k", lw=0.6)
a2 = ax[1].twinx(); a2.plot(range(len(yillar)), g.TANOM_JA, "o-", color="#b4443a", label="Temperature anomaly, Jul–Aug")
a2.set_ylabel("Temperature anomaly (°C)"); ax[1].set_ylabel("Standardized anomaly")
ax[1].set_xticks(range(len(yillar)), yillar)
h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
ax[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
fig.tight_layout(); fig.savefig(os.path.join(out, "Fig_interannual.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
ym = d.groupby("YIL")[["dNDMI", "DSI", "PANOM_AMJJA", "SWANOM_JA", "TANOM_JA", "VPD_JA", "CWB_AMJJA", "P_JA", "LST_JA"]].mean()
ym.round(4).to_csv(os.path.join(out, "yearly_means.csv"))
R["year_corr_dNDMI"] = ym.corr()["dNDMI"].round(3).to_dict()
# yıllık stresli oran
d["STRES"] = d.DSI <= -1
R["stressed_share_by_year"] = d.groupby("YIL").STRES.mean().round(4).to_dict()
oz = pd.read_csv(os.path.join(veri, "parsel_oznitelik.csv"))
dd = d
R["dNDMI_2025_by_region"] = dd[dd.YIL == 2025].groupby("BOLGE_ADI").dNDMI.agg(["mean", "count"]).round(4).to_dict()
R["stressed_2025_by_region"] = dd[dd.YIL == 2025].groupby("BOLGE_ADI").STRES.mean().round(3).to_dict()

# 5 Stres tipi
st = ns["stres_tipi"](d, None, out)
t = st.div(st.sum(1), axis=0) * 100
R["stress_type_counts"] = st.to_dict()
R["isolated_share_of_stressed"] = float(pd.read_csv(os.path.join(out, "parsel_yil_stres_tipi.csv")).query("STRES").IZOLE.mean())
sdf = pd.read_csv(os.path.join(out, "parsel_yil_stres_tipi.csv"))
s2 = sdf[sdf.STRES]
R["biotic_isolated_crosstab"] = pd.crosstab(s2.STRES_TIPI, s2.IZOLE).to_dict()
fig, ax = plt.subplots(figsize=(6.5, 3.4))
alt = np.zeros(len(t))
for k, c, l in [("Kuraklık tipi", "#c9a227", "Drought type"), ("Biyotik şüphesi", "#b4443a", "Biotic suspected")]:
    if k in t: ax.bar(t.index.astype(str), t[k], bottom=alt, color=c, label=l); alt += t[k].values
ax.set_ylabel("Share of parcels (%)"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(out, "Fig_stress_type.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# 6 Tanımlayıcı istatistik
R["n"] = len(d); R["parcels"] = int(d.PID.nunique()); R["blocks"] = int(d.BLOK.nunique())
R["dNDMI_mean"] = float(d.dNDMI.mean()); R["dNDMI_sd"] = float(d.dNDMI.std())
R["nobs_AM_median"] = float(d.NOBS_AM.median()); R["nobs_JA_median"] = float(d.NOBS_JA.median())
R["spearman_dNDMI_vs"] = d[feats + ["dNDMI"]].corr(method="spearman")["dNDMI"].drop("dNDMI").sort_values().round(3).to_dict()
json.dump(R, open(os.path.join(out, "post_summary.json"), "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps(R, ensure_ascii=False, default=str)[:6000])
