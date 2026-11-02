import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
from frame import get_data, folds, FEATS
from rainfe import add_rain, RAINF2
from prep import SETUP
tr, te = get_data(); tr, te = add_rain(tr, te)
DRYF = [f for f in FEATS if f not in ('ride_height_mm', 'brake_bias_pct', 'rain_mm_h', 'rain_slick', 'rain_inter', 'rain_wet', 'tyre_mismatch', 'compound', 'comp_ord', 'is_slick')] + ['compound']
P = dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
wetish = (tr.rain_ema10 > 0.01) | (tr.is_slick == 0) | (tr.rain_mm_h > 0)
print('wet-ish train rows', wetish.sum(), ' test', ((te.rain_ema10 > 0.01) | (te.is_slick == 0) | (te.rain_mm_h > 0)).sum())
# dry model OOF on all rows (trained on dry rows of other races)
f = folds(tr); dry = np.zeros(len(tr))
for k in range(5):
    fi = (f != k) & ~wetish
    m = lgb.LGBMRegressor(**P, n_estimators=1200).fit(tr.loc[fi, DRYF], tr.yt[fi]); dry[f == k] = m.predict(tr.loc[f == k, DRYF])
tr['wf'] = tr.yt - dry
w = tr[wetish].copy()
print('dry-row RMSE of dry model', np.sqrt(np.mean((np.exp(dry + np.log(tr.cbase)) - tr.lap_time_sec)[~wetish] ** 2)).round(3))
w['rb'] = pd.cut(w.rain_mm_h, [-1, 0, 0.3, 0.7, 1.2, 2, 3, 4.5, 6, 10])
w['eb'] = pd.cut(w.rain_ema5, [-1, 0.05, 0.3, 0.7, 1.2, 2, 3, 4.5, 6, 10])
print('weather factor (pct) by current rain:'); print((w.groupby(['rb', 'compound'], observed=True).wf.mean().unstack() * 100).round(2))
print('count:'); print(w.groupby(['rb', 'compound'], observed=True).size().unstack())
print('weather factor (pct) by rain_ema5:'); print((w.groupby(['eb', 'compound'], observed=True).wf.mean().unstack() * 100).round(2))
print('std of wf within cells:', w.groupby(['rb', 'compound'], observed=True).wf.std().mean().round(4))
# which rain representation explains wf best per compound (R2 of a small GBM)
for c in ['rain_mm_h', 'rain_lag1', 'rain_ema2', 'rain_ema5', 'rain_ema10']:
    r2 = []
    for comp in ['slick', 'Intermediate', 'Wet']:
        s = w[w.is_slick == 1] if comp == 'slick' else w[w.compound.astype(str) == comp]
        cc = np.corrcoef(s[c], s.wf)[0, 1]; r2.append(round(cc, 3))
    print(c, 'corr with wf  slick/inter/wet:', r2)
