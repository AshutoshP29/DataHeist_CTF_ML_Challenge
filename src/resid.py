import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from prep import load, SETUP, CATS
from cv import cv, CTX
al = load(impute=False)
tr = al[al.is_test==0].reset_index(drop=True)
oof = cv(al, CATS+CTX+SETUP, name='B raw')
tr['res'] = tr.lap_time_sec - oof
b = tr[(tr.src_ch1==1)&(tr.src_ch5==1)]
rm = b.groupby('race').res.mean()
print("race-mean residual std:", rm.std().round(3), " within-race resid std:", (b.res - b.race.map(rm)).std().round(3))
print(rm.sort_values().round(2).to_string())
for c in ['track_status','compound','circuit','season']:
    print(b.groupby(c).res.agg(['mean','std','count']).round(3))
# big residuals
print(b.loc[b.res.abs().sort_values(ascending=False).index[:15], ['race','lap','track_status','compound','tyre_age_laps','rain_mm_h','engine_mode','lap_time_sec','res']])
