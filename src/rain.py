import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
from frame import get_data
tr, te = get_data()
al = pd.concat([tr, te], ignore_index=True)
rl = al.groupby(['race', 'lap']).rain_mm_h.first().reset_index()
# full lap grid per race
grid = []
for r, d in rl.groupby('race'):
    full = pd.DataFrame({'lap': np.arange(1, d.lap.max() + 1)}); full['race'] = r
    full = full.merge(d, on=['race', 'lap'], how='left'); full['rain_mm_h'] = full.rain_mm_h.interpolate(limit_direction='both')
    grid.append(full)
g = pd.concat(grid)
print("laps with rain>0 per race:"); print(g.groupby('race').rain_mm_h.apply(lambda s: (s > 0).sum()).sort_values().tail(15).to_dict())
# example rainy race
rr = g.groupby('race').rain_mm_h.sum().idxmax(); print(rr)
x = g[g.race == rr]
t = tr[tr.race == rr].copy(); t['ratio'] = t.lap_time_sec / t.cbase
print(x.set_index('lap').rain_mm_h.round(2).to_string()[:3000])
print(t[t.track_status == 'GREEN'].groupby(['lap', 'compound'], observed=True).ratio.mean().unstack().round(3).to_string())
