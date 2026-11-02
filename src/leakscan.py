import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from frame import folds
from twostage import tr, te, wet_mask
from drycv import DRYF
ALLF = [f for f in DRYF if f not in ('status_ord',)] + ['ride_height_mm', 'brake_bias_pct']
P = dict(learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
f = folds(tr)
gtr = ((tr.track_status.astype(str) == 'GREEN') & ~wet_mask(tr)).values
gte = ((te.track_status.astype(str) == 'GREEN') & ~wet_mask(te)).values
BASE = [x for x in ALLF if x not in ('engine_mode', 'ride_height_mm', 'brake_bias_pct')]
rows = []
for feat in ALLF:
    F = [x for x in BASE if x != feat] + (['status_ord'])
    o = np.zeros(len(tr))
    for k in range(5):
        o[f == k] = lgb.LGBMRegressor(**P, n_estimators=600).fit(tr.loc[f != k, F], tr.yt[f != k]).predict(tr.loc[f == k, F])
    pt = lgb.LGBMRegressor(**P, n_estimators=600).fit(tr[F], tr.yt).predict(te[F])
    a = tr.loc[gtr].assign(p=o[gtr]); b = te.loc[gte].assign(p=pt[gte])
    a = a[a[feat].notna()]; b = b[b[feat].notna()]
    if a[feat].dtype.name == 'category' or a[feat].nunique() <= 12:
        # per-level mean of no-feature prediction (circuit-demeaned) — compare train vs test pattern
        da = (a.p - a.groupby('circuit').p.transform('mean')).groupby(a[feat].astype(str)).mean() * 100
        db = (b.p - b.groupby('circuit').p.transform('mean')).groupby(b[feat].astype(str)).mean() * 100
        j = pd.concat([da, db], axis=1, keys=['tr', 'te']).dropna()
        rows.append((feat, 'levels', round(j.tr.std(), 3), round(j.te.std(), 3), round(j.tr.corr(j.te), 3) if len(j) > 2 else np.nan))
    else:
        dm = lambda d, c: d[c] - d.groupby('circuit')[c].transform('mean')
        rows.append((feat, 'numeric', round(np.corrcoef(dm(a, feat), dm(a, 'p'))[0, 1], 3), round(np.corrcoef(dm(b, feat), dm(b, 'p'))[0, 1], 3), np.nan))
    print(rows[-1], flush=True)
print('\n(feature, type, train strength, test strength, train-test pattern corr)')
