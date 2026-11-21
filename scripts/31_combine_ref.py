"""Combine the 30% subsample (M, P) with the Earth Engine output for the other 70% of stands into one table PID, YIL, M, P (x 1e4)."""
import glob, pandas as pd, numpy as np
rows = []
for f in sorted(glob.glob("/home/claude/data/full/ref_*.txt")):
    for line in open(f):
        k, v = line.strip().split(":", 1); y = int(k[1:5])
        for it in v.split(";"):
            if not it: continue
            a = it.split(",")
            rows.append((int(a[0]), y, float(a[1]) if len(a) == 3 else np.nan, float(a[-1])))
n = pd.DataFrame(rows, columns=["PID", "YIL", "M", "P"])
s = pd.read_csv("/home/claude/data/mj/pr_all.csv")
c = pd.concat([s, n[~n.set_index(["PID", "YIL"]).index.isin(s.set_index(["PID", "YIL"]).index)]]).drop_duplicates(["PID", "YIL"])
c.to_csv("/home/claude/data/full/pr_full.csv", index=False)
print(len(s), len(n), len(c), c.PID.nunique(), c.groupby("YIL").size().to_dict())
