import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
from prep import load, SETUP
al = load(impute=True)
tr = al[al.is_test==0].copy()
tr['circuit'] = tr.circuit.astype(str); tr['track_status']=tr.track_status.astype(str); tr['compound']=tr.compound.astype(str)
gm = tr[(tr.track_status=='GREEN')&tr.compound.isin(['Soft','Medium','Hard'])&(tr.rain_mm_h.fillna(0)==0)]
tr['cm'] = tr.circuit.map(gm.groupby('circuit').lap_time_sec.median())
tr['r'] = tr.lap_time_sec / tr.cm
print(tr.groupby('track_status').r.describe().round(4))
print(tr.groupby('track_status').apply(lambda d: (d.lap_time_sec-d.cm).describe()).round(3))
g = tr[(tr.track_status=='GREEN')]
g['rb'] = pd.cut(g.rain_mm_h, [-0.1,0,0.5,1,2,4,10])
print(g.groupby(['rb','compound']).r.agg(['mean','count']).query('count>0').round(4))
s1 = tr[(tr.src_ch1==1)&(tr.track_status=='GREEN')&tr.compound.isin(['Soft','Medium','Hard'])]
s1['rr'] = s1.r
for c in ['wing_level','suspension_stiffness','engine_mode','cooling_level','pu_age_races']:
    q = pd.cut(s1[c], 5) if s1[c].nunique()>8 else s1[c]
    print('\n==', c); print(s1.groupby([s1.circuit, q]).rr.mean().unstack().round(4))
