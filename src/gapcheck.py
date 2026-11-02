import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from frame import get_data
tr, te = get_data()
v4 = np.load('../oof/v4_test.npy'); te['pred'] = v4
for nm, d, y in [('train', tr[tr.src_ch5 == 1].copy(), 'lap_time_sec'), ('test', te.copy(), 'pred')]:
    d = d.sort_values(['race', 'driver', 'lap'])
    d['prev_lap'] = d.groupby(['race', 'driver'])['lap'].shift(); d['prev_gap'] = d.groupby(['race', 'driver']).gap_ahead_sec.shift()
    c = d[d.prev_lap == d['lap'] - 1].copy(); c['dgap'] = c.gap_ahead_sec - c.prev_gap
    c['rel'] = c[y] - c.groupby(['race', 'lap'])[y].transform('mean')
    print(nm, 'consecutive pairs', len(c), ' corr(dgap, lap time rel to field)', round(c[['dgap', 'rel']].corr().iloc[0, 1], 3), ' gap autocorr', round(c[['gap_ahead_sec', 'prev_gap']].corr().iloc[0, 1], 3))
print('test rows per race', te.groupby('race').size().to_dict())
print('train rows per race (ch5)', tr[tr.src_ch5 == 1].groupby('race').size().describe().round(0).to_dict())
