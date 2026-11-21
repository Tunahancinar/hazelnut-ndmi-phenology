import pandas as pd, numpy as np, geopandas as gpd, glob, json, sys
from scipy import stats
raw = pd.read_csv('results5/tuik/tuik_raw.txt', sep=r'\s+')
provs = list(raw.columns[1:])
rows=[]
for _,r in raw.iterrows():
    k=r['PROV']; kind=k[0]; y=int(k[1:])
    for p in provs: rows.append(dict(PROV=p,YIL=y,kind=kind,val=float(r[p])))
t = pd.DataFrame(rows).pivot_table(index=['PROV','YIL'],columns='kind',values='val').reset_index()
t['YIELD'] = t['P']*1000/t['A']   # kg/da
t.to_csv('results5/tuik/tuik_hazelnut.csv',index=False)
mt = pd.read_csv('results3/model_tahminleri.csv', usecols=['PID','YIL'])
s2 = pd.concat([pd.read_csv(f, usecols=['PID','YIL','NDMI_AM','NDMI_JA']) for f in sorted(glob.glob('data/csv/FINDIK_S2_20*.csv'))])
s2['DNDMI']=s2.NDMI_JA-s2.NDMI_AM
d = mt.merge(s2,on=['PID','YIL'])
att = pd.read_csv('data/csv/parsel_oznitelik.csv', usecols=['PID','LON','LAT'])
g = gpd.GeoDataFrame(att, geometry=gpd.points_from_xy(att.LON,att.LAT), crs=4326)
adm = gpd.read_file('/home/claude/work/bolge_adm1.gpkg')
namecol = [c for c in adm.columns if c.lower() in ('name','shapename','name_1','adm1_name','name_en')][0]
j = gpd.sjoin(g, adm[[namecol,'geometry']], how='left', predicate='within')[['PID',namecol]].rename(columns={namecol:'PROV'})
d = d.merge(j,on='PID',how='left')
print(d.PROV.value_counts().head(20))
fix={'Duzce':'Düzce','Bartin':'Bartın','Gumushane':'Gümüşhane'}
d['PROV']=d.PROV.replace(fix)
py = d.groupby(['PROV','YIL']).agg(DNDMI=('DNDMI','mean'),n=('PID','nunique')).reset_index()
m = py.merge(t,on=['PROV','YIL'])
m = m[m.n>=50]
# anomalies within province
for c in ['DNDMI','YIELD']:
    m[c+'_an'] = m[c]-m.groupby('PROV')[c].transform('mean')
out={}
r,p = stats.pearsonr(m.DNDMI_an, m.YIELD_an); out['all']=dict(r=r,p=p,n=len(m))
mm=m[m.YIL<2025]; r,p = stats.pearsonr(mm.DNDMI_an, mm.YIELD_an); out['excl2025']=dict(r=r,p=p,n=len(mm))
rs,ps = stats.spearmanr(m.DNDMI_an, m.YIELD_an); out['spearman_all']=dict(r=rs,p=ps)
# lagged: canopy decline in year t vs yield in t+1
m2 = m.sort_values(['PROV','YIL']).copy(); m2['YIELD_an_next']=m2.groupby('PROV').YIELD_an.shift(-1)
mm2=m2.dropna(subset=['YIELD_an_next']); r,p=stats.pearsonr(mm2.DNDMI_an,mm2.YIELD_an_next); out['lag1']=dict(r=r,p=p,n=len(mm2))
out['provinces']=sorted(m.PROV.unique().tolist())
print(m.round(3).to_string()); print(json.dumps(out,indent=1))
m.to_csv('results5/tuik/yield_vs_dndmi.csv',index=False)
json.dump(out,open('results5/tuik/yield_check.json','w'),indent=1)
