import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t = pd.read_csv("/home/claude/results10/peak_doy.csv"); order = ["<250", "250-500", "500-750", ">750"]
lab = ["< 250", "250 to\n500", "500 to\n750", "> 750"]
plt.rcParams.update({"font.size": 8})
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1, 1.4]})
ax[0].boxplot([t[t.EC == c].DOY for c in order], tick_labels=lab, showfliers=False, widths=0.6,
              medianprops=dict(color="#2f6f4e"), boxprops=dict(color="0.3"), whiskerprops=dict(color="0.3"), capprops=dict(color="0.3"))
ax[0].set_xlabel("Elevation class (m)"); ax[0].set_ylabel("Day of year of spring NDVI peak"); ax[0].set_title("(a)", loc="left")
m = t.groupby(["YIL", "EC"]).DOY.median().unstack()[order]
cols = ["#c9b27c", "#8fae7e", "#4f8a6b", "#1f4e3d"]
for c, l, col in zip(order, lab, cols): ax[1].plot(m.index, m[c], marker="o", ms=3, lw=1.2, color=col, label=l.replace("\n", " ") + " m")
ax[1].set_xlabel("Year"); ax[1].set_ylabel("Median day of year"); ax[1].set_title("(b)", loc="left"); ax[1].legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig("/home/claude/figs3/Fig_peak_doy.png", dpi=300, bbox_inches="tight")
