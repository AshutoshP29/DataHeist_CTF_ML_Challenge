import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from frame import folds
from twostage import tr, te, wet_mask
from drycv import DF
F = [f for f in DF if f != 'engine_mode']
P = dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
dry = ~wet_mask(tr).values; both = ((tr.src_ch1 == 1) & (tr.src_ch5 == 1)).values
y = tr.lap_time_sec.values; cb = np.log(tr.cbase.values)
NUMF = [f for f in F if f not in ('circuit', 'driver', 'team', 'track_status', 'compound')]

def m_plain(trn, X):
    return lgb.LGBMRegressor(**P, n_estimators=1200).fit(trn[F], trn.yt).predict(X[F])
def m_linear_tree(trn, X):
    return lgb.LGBMRegressor(**P, n_estimators=1200, linear_tree=True, linear_lambda=1.0).fit(trn[F], trn.yt).predict(X[F])
def m_detrend(trn, X):
    # global within-circuit linear temp slope, fitted on dry green rows, then GBM on the residual
    d = trn[(trn.track_status.astype(str) == 'GREEN')].dropna(subset=['track_temp_c'])
    tc = d.track_temp_c - d.groupby('circuit').track_temp_c.transform('mean'); yc = d.yt - d.groupby('circuit').yt.transform('mean')
    # partial out other effects with a quick GBM without temp
    Fn = [f for f in F if f != 'track_temp_c']
    r = yc - (lambda p: p - pd.Series(p, d.index).groupby(d.circuit).transform('mean'))(lgb.LGBMRegressor(**P, n_estimators=600).fit(d[Fn], d.yt).predict(d[Fn]))
    slope = np.polyfit(tc, r, 1)[0]
    off = lambda Z: slope * (Z.track_temp_c.fillna(40) - 40)
    g = lgb.LGBMRegressor(**P, n_estimators=1200).fit(trn[F], trn.yt - off(trn))
    return g.predict(X[F]) + off(X), slope

models = {'plain': m_plain, 'linear_tree': m_linear_tree, 'detrend': lambda a, b: m_detrend(a, b)[0]}
# split A: hold out hottest race per circuit (train on cooler races)
rt = tr[tr.src_ch5 == 1].groupby('race').agg(c=('circuit', 'first'), t=('track_temp_c', 'mean'))
hot = rt.loc[rt.groupby('c').t.idxmax()].index
ho = tr.race.isin(hot).values
trn = tr[~ho & dry].reset_index(drop=True)
m = ho & dry & both
abv = np.zeros(len(tr), bool)
mx = tr[~ho].groupby('circuit').track_temp_c.max()
abv = (tr.track_temp_c > tr.circuit.map(mx).astype(float)).values
print('hot-holdout rows (dry, full):', m.sum(), ' of which above train max:', (m & abv).sum())
for nm, fn in models.items():
    p = np.exp(fn(trn, tr[m]) + cb[m]); e = (p - y[m])
    print(f'{nm:12s} hot-holdout RMSE {np.sqrt(np.mean(e**2)):.4f}  bias {(-e).mean():+.3f}   above-max rows RMSE {np.sqrt(np.mean(e[abv[m]]**2)):.4f} bias {(-e[abv[m]]).mean():+.3f}')
# standard grouped split (fold seed 7) for sanity
f = folds(tr, seed=7)
for nm, fn in models.items():
    o = np.full(len(tr), np.nan)
    for k in range(5):
        fi = (f != k) & dry; va = (f == k) & dry
        o[va] = fn(tr[fi].reset_index(drop=True), tr[va])
    mm = dry & both
    print(f'{nm:12s} standard split7 dry RMSE {np.sqrt(np.mean((np.exp(o[mm] + cb[mm]) - y[mm])**2)):.4f}')
print('detrend slope (log-ratio per degC):', m_detrend(tr[dry].reset_index(drop=True), te.head(5))[1])
