import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
from prep import load, add_base, SETUP, CATS
from cv import cv, CTX
al = add_base(load(impute=True))
tr = al[al.is_test==0].reset_index(drop=True)
oof = cv(al, CATS+CTX+SETUP, target='logbase', name='best')
tr['res'] = tr.lap_time_sec - oof; tr['lres'] = np.log(tr.lap_time_sec/oof)
b = tr[(tr.src_ch1==1)&(tr.src_ch5==1)].copy()
rm = b.groupby('race').lres.mean()
print("race-mean log-resid std:", rm.std().round(5), "within:", (b.lres-b.race.map(rm)).std().round(5))
print("race-mean abs resid std:", b.groupby('race').res.mean().std().round(3), " within:", (b.res - b.groupby('race').res.transform('mean')).std().round(3))
for c in ['track_status','compound','season','engine_mode','circuit']:
    print(b.groupby(c, observed=True).res.agg(['mean','std','count']).round(3).T)
b['ab']=b.res.abs()
print(b.sort_values('ab',ascending=False).head(12)[['race','lap','driver','track_status','compound','tyre_age_laps','rain_mm_h','engine_mode','lap_time_sec','res']])
# does residual correlate with any numeric?
num = ['lap','tyre_age_laps','stint','laps_remaining','quali_position','gap_ahead_sec','rain_mm_h','track_temp_c','lap_frac']+SETUP
print(b[num].corrwith(b.res).round(3))
# race-level: do race-mean resids relate to race-level feature stats?
rs = b.groupby('race').agg(res=('res','mean'), temp=('track_temp_c','mean'), rain=('rain_mm_h','mean'), season=('season','first'))
print(rs.corr().round(3))
