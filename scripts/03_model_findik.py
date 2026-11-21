"""Fındık parsellerinde yaz su stresi modeli: çok yöntemli karşılaştırma.

Girdi klasörü (GEE çıktıları + öznitelik):
  FINDIK_STATIK.csv, FINDIK_DINAMIK_<yil>.csv, FINDIK_KENARIC_<yil>.csv, parsel_oznitelik.csv
Kullanım:
  python -I 03_model_findik.py <veri_klasoru> <cikti_klasoru> [--hedef dNDMI|dNDVI|NDMI_JA_z]
"""
import argparse, glob, json, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
RS = 42

IKLIM = ["P_AMJJA", "CWB_AMJJA", "P_JFMAMJ", "P_JA", "T_JA", "TMAX_JA", "VPD_JA",
         "SW1_JA", "SW2_JA", "SW3_JA", "PANOM_AMJJA", "TANOM_JA", "SWANOM_JA", "LST_JA"]
TOPO = ["ELEV", "SLOPE", "NORTHNESS", "EASTNESS", "TPI300", "TWI"]
TOPRAK = [f"{v}_{d}" for v in ["CLAY", "SAND", "SILT", "SOC", "BDOD", "CFVO", "PHH2O", "CEC"]
          for d in ["0_30", "30_100"]] + ["FC33_0_100", "SOIL_DEPTH"]
MESCERE = ["CROWN_CLOSURE", "SITE_CLASS", "AREA_HA"]
YENI_AD = {"KAPALILIK": "CROWN_CLOSURE", "BONITET": "SITE_CLASS", "ALAN_HA": "AREA_HA", "SPI_AMJJA": "PANOM_AMJJA"}


def oku(d):
    stat = pd.read_csv(os.path.join(d, "FINDIK_STATIK.csv"))
    s2_cols = lambda df: [c for c in df.columns if c in ("PID", "YIL") or c.endswith(("_AM", "_JA", "_SO")) and not c.startswith("S1_")]
    parts = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(d, "FINDIK_S2_*.csv")))]
    for f in sorted(glob.glob(os.path.join(d, "FINDIK_DINAMIK_*.csv"))):
        k = pd.read_csv(f)
        if not parts or k.YIL.iloc[0] not in pd.concat(parts).YIL.unique():
            parts.append(k)
    s2 = pd.concat([p[s2_cols(p)] for p in parts])
    s2 = s2[[c for c in s2.columns if c not in IKLIM]]
    ik = pd.read_csv(glob.glob(os.path.join(d, "FINDIK_IKLIM_*.csv"))[0]).rename(columns=YENI_AD)
    ik = ik[[c for c in ik.columns if c in IKLIM + ["PID", "YIL"]]]
    din = s2.merge(ik, on=["PID", "YIL"], how="left")
    oz = pd.read_csv(os.path.join(d, "parsel_oznitelik.csv"))
    parts = []
    for f in sorted(glob.glob(os.path.join(d, "FINDIK_KENARIC_*.csv"))):
        k = pd.read_csv(f)
        if "YIL" in k.columns:
            parts.append(k[[c for c in ["PID", "PARCA", "YIL", "NDVI", "NDMI", "NDRE"] if c in k.columns]])
        else:  # geniş biçim: NDVI_2020, NDMI_2020 ...
            long = []
            for y in sorted({int(c.split("_")[1]) for c in k.columns if c[:4] in ("NDVI", "NDMI", "NDRE") and "_" in c}):
                t = k[["PID", "PARCA"]].copy(); t["YIL"] = y
                for b in ("NDVI", "NDMI", "NDRE"):
                    t[b] = k.get(f"{b}_{y}")
                long.append(t)
            parts.append(pd.concat(long))
    ki = pd.concat(parts) if parts else None
    drop = [c for c in ["system:index", ".geo", "ISL"] if c in stat.columns]
    stat = stat.drop(columns=drop)
    din = din.drop(columns=[c for c in ["system:index", ".geo", "ISL"] if c in din.columns])
    hw = os.path.join(d, "FINDIK_HWSD.csv")
    if os.path.exists(hw):
        stat = stat.merge(pd.read_csv(hw)[["PID", "SOIL_DEPTH"]], on="PID", how="left")
    df = din.merge(stat, on="PID", how="left").merge(oz, on="PID", how="left")
    df = df.rename(columns=YENI_AD)
    return df, ki


def hedef_hazirla(df):
    df = df.copy()
    # yalnızca saf fındık meşcereleri (OGM meşcere tipi Fna, Fna3)
    df = df[df["MESCERE"].astype(str).str.match(r"^Fn[a-d]*\d?(-T)?$")].copy()
    ok = (df["NOBS_AM"] >= 2) & (df["NOBS_JA"] >= 2) & (df["NDVI_AM"] > 0.4)
    df = df[ok].copy()
    df["dNDMI"] = df["NDMI_JA"] - df["NDMI_AM"]
    df["dNDVI"] = df["NDVI_JA"] - df["NDVI_AM"]
    g = df.groupby("PID")["NDMI_JA"]
    df["NDMI_JA_z"] = (df["NDMI_JA"] - g.transform("mean")) / g.transform("std")
    gd = df.groupby("PID")["dNDMI"]
    df["DSI"] = (df["dNDMI"] - gd.transform("mean")) / gd.transform("std")
    # 10 km mekânsal blok (UTM 36N yaklaşık)
    x = (df["LON"] - 27) * 111.32 * np.cos(np.radians(41))
    y = (df["LAT"] - 38) * 110.57
    df["BLOK"] = (x // 10).astype(int).astype(str) + "_" + (y // 10).astype(int).astype(str)
    return df


def modeller():
    sc = lambda m: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)
    imp = lambda m: make_pipeline(SimpleImputer(strategy="median"), m)
    return {
        "MLR": sc(LinearRegression()),
        "SVR": sc(SVR(C=1.0, epsilon=0.01, gamma="scale")),
        "kNN": sc(KNeighborsRegressor(n_neighbors=15, weights="distance")),
        "MLP": sc(MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                               max_iter=500, random_state=RS)),
        "RF": imp(RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.4,
                                        n_jobs=-1, random_state=RS)),
        "ET": imp(ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5,
                                      n_jobs=-1, random_state=RS)),
        "GBM": HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                             l2_regularization=1.0, random_state=RS),
        "XGBoost": XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=6, subsample=0.8,
                                colsample_bytree=0.7, reg_lambda=1.0, n_jobs=-1, random_state=RS),
        "LightGBM": LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=RS),
        "CatBoost": CatBoostRegressor(iterations=800, learning_rate=0.05, depth=6, verbose=0,
                                      random_seed=RS, thread_count=-1),
    }


SVR_MAX = 15000  # SVR O(n^2), eğitim alt örneklemi


def fit_pred(ad, m, Xtr, ytr, Xte):
    if ad == "SVR" and len(Xtr) > SVR_MAX:
        i = np.random.default_rng(RS).choice(len(Xtr), SVR_MAX, replace=False)
        Xtr, ytr = Xtr.iloc[i], ytr.iloc[i]
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def metrik(y, p):
    return dict(R2=r2_score(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=mean_absolute_error(y, p))


def degerlendir(df, X, y, out):
    sonuc, oof_all = [], {}
    tahmin = df[["PID", "YIL", "BLOK"]].copy()
    gkf = GroupKFold(n_splits=5)
    son_yil = int(df["YIL"].max())
    tr_t, te_t = df["YIL"] < son_yil, df["YIL"] == son_yil
    for ad, m in modeller().items():
        t0 = time.time()
        oof = np.full(len(y), np.nan)
        for tr, te in gkf.split(X, y, df["BLOK"]):
            oof[te] = fit_pred(ad, modeller()[ad], X.iloc[tr], y.iloc[tr], X.iloc[te])
        r = {"Model": ad, "Doğrulama": "Mekânsal blok CV (5 kat)", **metrik(y, oof)}
        sonuc.append(r); oof_all[ad] = oof
        pt = fit_pred(ad, modeller()[ad], X[tr_t], y[tr_t], X[te_t])
        sonuc.append({"Model": ad, "Doğrulama": f"Zamansal ({son_yil} test)", **metrik(y[te_t], pt)})
        tr5, te5 = df["YIL"] != 2025, df["YIL"] == 2025
        p5 = fit_pred(ad, modeller()[ad], X[tr5], y[tr5], X[te5])
        sonuc.append({"Model": ad, "Doğrulama": "Zamansal (2025 test)", **metrik(y[te5], p5)})
        tahmin[f"OOF_{ad}"] = oof
        tahmin.loc[te5.values, f"T2025_{ad}"] = p5
        tahmin.loc[te_t.values, f"T{son_yil}_{ad}"] = pt
        print(f"{ad:9s} CV R2={r['R2']:.3f} RMSE={r['RMSE']:.4f} | {son_yil} R2={sonuc[-2]['R2']:.3f} | 2025 R2={sonuc[-1]['R2']:.3f} ({time.time()-t0:.0f} s)", flush=True)
    tahmin.to_csv(os.path.join(out, "model_tahminleri.csv"), index=False)
    s = pd.DataFrame(sonuc)
    s.to_csv(os.path.join(out, "model_karsilastirma.csv"), index=False, encoding="utf-8-sig")
    return s, oof_all


def grafikler(s, oof_all, y, out, hedef):
    cv = s[s["Doğrulama"].str.startswith("Mekânsal")].sort_values("R2")
    tm = s[s["Doğrulama"].str.startswith("Zamansal") & (s["Doğrulama"] != "Zamansal (2025 test)")].set_index("Model").loc[cv["Model"]]
    t5 = s[s["Doğrulama"] == "Zamansal (2025 test)"].set_index("Model").loc[cv["Model"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    i = np.arange(len(cv))
    ax.barh(i - 0.27, cv["R2"], 0.27, color="#2f6f4e", label="Spatial block CV")
    ax.barh(i, tm["R2"], 0.27, color="#c9a227", label="Temporal, 2026 held out")
    ax.barh(i + 0.27, t5["R2"], 0.27, color="#8c6bb1", label="Temporal, 2025 held out")
    ax.set_yticks(i, cv["Model"]); ax.set_xlabel("R²"); ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "sekil_model_r2.png"), dpi=300); plt.close(fig)

    best = cv["Model"].iloc[-1]
    p = oof_all[best]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.hexbin(y, p, gridsize=60, cmap="Greens", mincnt=1, bins="log")
    lo, hi = np.nanpercentile(np.r_[y, p], [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], color="k", lw=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed ΔNDMI"); ax.set_ylabel(f"Predicted ΔNDMI ({best})")
    fig.tight_layout(); fig.savefig(os.path.join(out, "sekil_gozlenen_tahmin.png"), dpi=300); plt.close(fig)
    return best


def shap_analiz(best, X, y, out):
    import shap
    m = modeller()[best]
    if best not in ("RF", "ET", "GBM", "XGBoost", "LightGBM", "CatBoost"):
        m = modeller()["LightGBM"]; best = "LightGBM"
    m.fit(X, y)
    est = m[-1] if hasattr(m, "steps") else m
    Xs = X.sample(min(5000, len(X)), random_state=RS)
    Xi = m[:-1].transform(Xs) if hasattr(m, "steps") else Xs
    sv = shap.TreeExplainer(est).shap_values(Xi)
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
    imp.rename("ortalama_|SHAP|").to_csv(os.path.join(out, f"shap_onem_{best}.csv"), encoding="utf-8-sig")
    plt.figure()
    shap.summary_plot(sv, pd.DataFrame(np.asarray(Xi), columns=X.columns), max_display=20, show=False)
    plt.gcf().set_size_inches(7, 7); plt.tight_layout()
    plt.savefig(os.path.join(out, f"sekil_shap_{best}.png"), dpi=300); plt.close()
    return imp


def stres_tipi(df, ki, out):
    """Belgedeki ayrım tablosuna dayalı sezgisel sınıflama (saha etiketi yok).
    Spektral ölçüt: NDRE z-anomalisi NDMI z-anomalisinden en az 1 SS daha düşükse biyotik şüphesi.
    Mekânsal ölçüt: 10 km blokta stresli parsel oranı (izole stres biyotik olasılığını artırır)."""
    d = df[["PID", "YIL", "BLOK", "DSI", "NDRE_JA", "NDMI_JA"]].copy()
    gr = d.groupby("PID")
    d["NDRE_z"] = (d["NDRE_JA"] - gr["NDRE_JA"].transform("mean")) / gr["NDRE_JA"].transform("std")
    d["NDMI_z"] = (d["NDMI_JA"] - gr["NDMI_JA"].transform("mean")) / gr["NDMI_JA"].transform("std")
    d["STRES"] = d["DSI"] <= -1
    d["BLOK_STRES_ORANI"] = d.groupby(["BLOK", "YIL"])["STRES"].transform("mean")
    biyotik = d["STRES"] & ((d["NDRE_z"] - d["NDMI_z"]) <= -1)
    d["STRES_TIPI"] = np.select([biyotik, d["STRES"]], ["Biyotik şüphesi", "Kuraklık tipi"], "Stres yok")
    d["IZOLE"] = d["STRES"] & (d["BLOK_STRES_ORANI"] < 0.2)
    d.to_csv(os.path.join(out, "parsel_yil_stres_tipi.csv"), index=False, encoding="utf-8-sig")
    return d.groupby(["YIL", "STRES_TIPI"]).size().unstack(fill_value=0)


def ek_grafikler(d, st_df, imp, best, X, y, out, hedef):
    import shap
    # Yıllara göre hedef dağılımı ve iklim anomalisi
    yillar = sorted(d.YIL.unique())
    fig, ax = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)
    ax[0].boxplot([d.loc[d.YIL == t, hedef].dropna() for t in yillar], positions=range(len(yillar)),
                  widths=0.6, showfliers=False, patch_artist=True,
                  boxprops=dict(facecolor="#cfe3d6", edgecolor="#2f6f4e"), medianprops=dict(color="#2f6f4e"))
    ax[0].axhline(d[hedef].median(), color="0.5", lw=0.8, ls="--")
    ax[0].set_ylabel(f"{hedef}")
    g = d.groupby("YIL")[["PANOM_AMJJA", "SWANOM_JA", "TANOM_JA"]].mean().loc[yillar]
    ax[1].bar(np.arange(len(yillar)) - 0.2, g.PANOM_AMJJA, 0.4, color="#4c78a8", label="SPI (Apr-Aug)")
    ax[1].bar(np.arange(len(yillar)) + 0.2, g.SWANOM_JA, 0.4, color="#c9a227", label="Soil water anomaly (Jul-Aug)")
    ax[1].axhline(0, color="k", lw=0.6)
    ax2 = ax[1].twinx(); ax2.plot(range(len(yillar)), g.TANOM_JA, "o-", color="#b4443a", label="Temperature anomaly (Jul-Aug)")
    ax2.set_ylabel("Temperature anomaly (°C)")
    ax[1].set_ylabel("Standardized anomaly"); ax[1].set_xticks(range(len(yillar)), yillar)
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "sekil_yillik.png"), dpi=300); plt.close(fig)
    # SHAP bağımlılık
    m = modeller()[best] if best in ("RF","ET","GBM","XGBoost","LightGBM","CatBoost") else modeller()["LightGBM"]
    m.fit(X, y)
    est = m[-1] if hasattr(m, "steps") else m
    Xs = X.sample(min(5000, len(X)), random_state=RS)
    Xi = pd.DataFrame(np.asarray(m[:-1].transform(Xs)) if hasattr(m, "steps") else Xs.values, columns=X.columns)
    sv = shap.TreeExplainer(est).shap_values(Xi)
    top = list(imp.index[:6])
    fig, axs = plt.subplots(2, 3, figsize=(9, 5.6))
    for a_, v in zip(axs.ravel(), top):
        j = list(X.columns).index(v)
        a_.scatter(Xi[v], sv[:, j], s=3, alpha=0.35, color="#2f6f4e", edgecolors="none")
        a_.axhline(0, color="0.5", lw=0.6); a_.set_xlabel(v); a_.set_ylabel("SHAP value")
    fig.tight_layout(); fig.savefig(os.path.join(out, "sekil_shap_bagimlilik.png"), dpi=300); plt.close(fig)
    # Stres tipi oranları
    if st_df is not None:
        t = st_df.groupby(["YIL", "STRES_TIPI"]).size().unstack(fill_value=0)
        t = t.div(t.sum(1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        alt = np.zeros(len(t))
        renk = {"Kuraklık tipi": "#c9a227", "Biyotik şüphesi": "#b4443a"}
        etik = {"Kuraklık tipi": "Drought type", "Biyotik şüphesi": "Biotic suspected"}
        for k in ["Kuraklık tipi", "Biyotik şüphesi"]:
            if k in t: ax.bar(t.index.astype(str), t[k], bottom=alt, color=renk[k], label=etik[k]); alt += t[k].values
        ax.set_ylabel("Share of parcels (%)"); ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(out, "sekil_stres_tipi.png"), dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("veri"); ap.add_argument("out")
    ap.add_argument("--hedef", default="dNDMI")
    ap.add_argument("--feats", default=None)
    ap.add_argument("--lag", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, ki = oku(a.veri)
    df = hedef_hazirla(df)
    feats = [c for c in IKLIM + TOPO + TOPRAK + MESCERE if c in df.columns]
    if a.feats:
        feats = json.load(open(a.feats))
    d = df.dropna(subset=[a.hedef]).reset_index(drop=True)
    if a.lag:  # önceki yılın gözlenen ΔNDMI değeri (otoregresif terim)
        d = d.sort_values(["PID", "YIL"])
        prev = d.groupby("PID")[[a.hedef, "YIL"]].shift(1)
        d["dNDMI_LAG1"] = np.where(prev["YIL"] == d["YIL"] - 1, prev[a.hedef], np.nan)
        d = d.dropna(subset=["dNDMI_LAG1"]).reset_index(drop=True)
        feats = feats + ["dNDMI_LAG1"]
    X, y = d[feats], d[a.hedef]
    print(f"{len(d)} parsel-yıl, {d.PID.nunique()} parsel, {len(feats)} değişken, {d.BLOK.nunique()} blok", flush=True)
    d[["PID", "YIL", "BLOK", a.hedef, "DSI"] + feats].to_csv(
        os.path.join(a.out, "model_veri_seti.csv"), index=False, encoding="utf-8-sig")
    yil_ozet = d.groupby("YIL")[[a.hedef, "DSI", "PANOM_AMJJA", "TANOM_JA", "VPD_JA"]].mean().round(4)
    yil_ozet.to_csv(os.path.join(a.out, "yil_ozet.csv"), encoding="utf-8-sig")
    s, oof = degerlendir(d, X, y, a.out)
    best = grafikler(s, oof, y, a.out, a.hedef)
    imp = shap_analiz(best, X, y, a.out)
    st = stres_tipi(d, ki, a.out)
    st_df = pd.read_csv(os.path.join(a.out, "parsel_yil_stres_tipi.csv")) if st is not None else None
    ek_grafikler(d, st_df, imp, best, X, y, a.out, a.hedef)
    ozet = {"n": len(d), "parsel": int(d.PID.nunique()), "en_iyi": best,
            "ilk10_shap": imp.head(10).round(5).to_dict(),
            "stres_tipi": None if st is None else st.to_dict()}
    json.dump(ozet, open(os.path.join(a.out, "ozet.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(s.pivot(index="Model", columns="Doğrulama", values="R2").round(3).to_string())
    print(json.dumps(ozet, ensure_ascii=False, default=str)[:1500])


if __name__ == "__main__":
    main()
