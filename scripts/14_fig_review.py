"""Fig_model_r2 (spatial R2 with block bootstrap CIs and baselines) and Fig_loyo."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, json
R5="/home/claude/results5"; F="/home/claude/figs3"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8})
ci=pd.read_csv(f"{R5}/bootstrap_ci.csv",index_col=0)
order=["RF","ET","XGBoost","CatBoost","LightGBM","GBM","kNN","MLP","SVR","MLR","Year x RD mean","Year mean"]
lab={"Year x RD mean":"Year × RD mean","Year mean":"Year mean"}
ci=ci.loc[order]
fig,ax=plt.subplots(figsize=(4.6,3.6))
yy=np.arange(len(ci))[::-1]
col=["#2c7c5a"]*6+["#c9a227"]*4+["0.6"]*2
ax.barh(yy,ci.R2,color=col,height=0.65)
ax.errorbar(ci.R2,yy,xerr=[ci.R2-ci.R2_lo,ci.R2_hi-ci.R2],fmt="none",ecolor="k",elinewidth=0.8,capsize=2)
ax.set_yticks(yy); ax.set_yticklabels([lab.get(i,i) for i in ci.index])
ax.set_xlabel("Spatial block cross validation R² (10 km blocks)"); ax.set_xlim(0,0.7)
for s in ("top","right"): ax.spines[s].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2c7c5a",label="Tree ensembles"),Patch(color="#c9a227",label="Other algorithms"),Patch(color="0.6",label="Baselines")],
          loc="upper center",bbox_to_anchor=(0.45,-0.15),ncol=3,frameon=False,fontsize=7)
fig.tight_layout(); fig.savefig(f"{F}/Fig_model_r2.png",dpi=300,bbox_inches="tight"); plt.close(fig)

lo=pd.read_csv(f"{R5}/loyo.csv",index_col=0)
ym=pd.read_csv("/home/claude/results3/yearly_means.csv",index_col=0)
mods=["MLR","SVR","kNN","MLP","RF","ET","GBM","XGBoost","LightGBM","CatBoost"]
yrs=lo.index.tolist()
fig,ax=plt.subplots(1,2,figsize=(7.2,3.0))
ens=["RF","ET","GBM","XGBoost","LightGBM","CatBoost"]; oth=["MLR","SVR","kNN","MLP"]
x=np.arange(len(yrs))
ax[0].plot(x,lo[ens].max(axis=1),color="#2c7c5a",lw=0); 
ax[0].fill_between(x,lo[ens].min(axis=1),lo[ens].max(axis=1),color="#2c7c5a",alpha=0.35,label="Tree ensembles (range)")
ax[0].fill_between(x,lo[oth].min(axis=1),lo[oth].max(axis=1),color="#c9a227",alpha=0.35,label="Other algorithms (range)")
ax[0].plot(x,lo["CatBoost"],color="#2c7c5a",marker="o",ms=3,lw=1,label="CatBoost")
ax[0].plot(x,lo["Climatology (RD mean)"],color="0.4",marker="s",ms=3,lw=1,ls="--",label="RD climatology")
ax[0].axhline(0,color="k",lw=0.5); ax[0].set_xticks(x); ax[0].set_xticklabels(yrs,rotation=45)
ax[0].set_ylabel("R² for the held out year"); ax[0].set_title("(a)",loc="left")
ax[0].legend(fontsize=6,frameon=False,loc="lower left")
bias=lo[[m+"_bias" for m in mods]]
ax[1].boxplot([bias.loc[yv].values for yv in yrs],positions=x,widths=0.5,flierprops=dict(ms=2))
ax[1].axhline(0,color="k",lw=0.5); ax[1].set_xticks(x); ax[1].set_xticklabels(yrs,rotation=45)
ax[1].set_ylabel("Mean bias of the ten models"); ax[1].set_title("(b)",loc="left")
for a in ax:
    for s in ("top","right"): a.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{F}/Fig_loyo.png",dpi=300,bbox_inches="tight"); plt.close(fig)
print("ok")
