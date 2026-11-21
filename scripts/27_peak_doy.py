"""Day of year of the spring NDVI peak observation by elevation class (parcel median DOY, 30% subsample)."""
import sys, glob, json, numpy as np, pandas as pd
D, O = sys.argv[1:3]
rows = []
for f in glob.glob(f"{D}/doy_*.txt"):
    for line in open(f):
        k, v = line.strip().split(":", 1); y = int(k[1:5])
        for it in v.split(";"):
            if it: p, d = it.split(","); rows.append((int(p), y, int(d)))
t = pd.DataFrame(rows, columns=["PID", "YIL", "DOY"]).drop_duplicates(["PID", "YIL"])
st = pd.read_csv("/home/claude/data/csv/FINDIK_STATIK.csv")[["PID", "ELEV"]]
t = t.merge(st, on="PID"); t["EC"] = pd.cut(t.ELEV, [-1, 250, 500, 750, 2000], labels=["<250", "250-500", "500-750", ">750"])
g = t.groupby("EC", observed=True).DOY.agg(["count", "median", "mean", lambda x: x.quantile(.25), lambda x: x.quantile(.75)]).round(1)
g.columns = ["n", "median", "mean", "q25", "q75"]; print(g); print(t.groupby("YIL").DOY.median())
from scipy.stats import spearmanr
r = spearmanr(t.ELEV, t.DOY); print("rho", r)
yr = {int(y): float(spearmanr(g2.ELEV, g2.DOY)[0]) for y, g2 in t.groupby("YIL")}
lm = np.polyfit(t.ELEV, t.DOY, 1)[0] * 100
json.dump({"n": int(len(t)), "years": sorted(map(int, t.YIL.unique())), "by_class": g.reset_index().astype({"EC": str}).to_dict("records"),
           "rho": float(r[0]), "rho_by_year": yr, "slope_days_per_100m": float(lm),
           "share_doy_ge_june": float((t.DOY >= 152).mean())}, open(f"{O}/peak_doy.json", "w"), indent=1)
print("slope d/100m", lm, yr)
t.to_csv(f"{O}/peak_doy.csv", index=False)
