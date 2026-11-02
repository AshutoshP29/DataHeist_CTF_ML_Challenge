import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from frame import folds
from twostage import tr, te, wet_mask
from drycv import DF
NOEM = [f for f in DF if f != 'engine_mode']
P = dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
f = folds(tr); o = np.zeros(len(tr))
for k in range(5):
    o[f == k] = lgb.LGBMRegressor(**P, n_estimators=1200).fit(tr.loc[f != k, NOEM], tr.yt[f != k]).predict(tr.loc[f == k, NOEM])
pt = lgb.LGBMRegressor(**P, n_estimators=1200).fit(tr[NOEM], tr.yt).predict(te[NOEM])
tr['ctx'] = o; te['ctx'] = pt
for nm, d in [('train (ch1 rows)', tr[tr.src_ch1 == 1]), ('test', te)]:
    g = d[(d.track_status.astype(str) == 'GREEN') & (d.engine_mode >= 2)]
    print(f'{nm:18s} GREEN, mode 2-5: corr(engine_mode, no-EM prediction) = {g[["engine_mode", "ctx"]].corr().iloc[0, 1]:+.3f}   mean no-EM pred by mode:', g.groupby('engine_mode').ctx.mean().mul(100).round(2).to_dict())
g = tr[(tr.src_ch1 == 1) & (tr.track_status.astype(str) == 'GREEN') & (tr.engine_mode >= 2)]
print('train GREEN: corr(engine_mode, actual yt) =', round(g[['engine_mode', 'yt']].corr().iloc[0, 1], 3), ' mean actual yt by mode:', g.groupby('engine_mode').yt.mean().mul(100).round(2).to_dict())
