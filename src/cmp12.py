import pandas as pd, numpy as np
pd.set_option('display.width', 250)
from frame import get_data
tr, te = get_data()
S = '../submissions/'
v1 = pd.read_csv(S+'v1_lgb_logbase.csv'); v2 = pd.read_csv(S+'v2_lgb_rain_tuned.csv')
te = te.merge(v1.rename(columns={'lap_time_sec':'p1'}), on='lap_id').merge(v2.rename(columns={'lap_time_sec':'p2'}), on='lap_id')
te['d'] = te.p2 - te.p1
te['seg'] = te.track_status.astype(str) + '|' + te.compound.astype(str) + '|' + pd.cut(te.rain_mm_h, [-1,0,1,3,20], labels=['dry','r<1','r1-3','r>3']).astype(str)
print('overall rms diff', np.sqrt((te.d**2).mean()).round(3))
g = te.groupby('seg').d.agg(n='size', mean='mean', rms=lambda s: np.sqrt((s**2).mean()))
g['msd_share'] = g.n * g.rms**2 / (te.d**2).sum()
print(g.sort_values('msd_share', ascending=False).round(3).head(15))
print(te.groupby('race').d.agg(['mean', lambda s: np.sqrt((s**2).mean())]).round(3))
