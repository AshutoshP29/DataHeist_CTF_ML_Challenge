"""Final robust ensemble: blended dry-pace models (LGBM x2 + NN) + weather-factor model, log-ratio target."""
import numpy as np, pandas as pd, lightgbm as lgb, warnings, time, sys; warnings.filterwarnings('ignore')
from frame import folds, score
from twostage import tr, te, wet_mask
from drycv import DF
from nn import nn_fp
from submit import write_submission
WF = ['compound', 'rain_mm_h', 'rain_lag1', 'status_ord', 'tyre_age_laps', 'laps_since_rain', 'rain_ema5', 'engine_mode', 'rain_max_so_far', 'rain_ema2']
WP = dict(learning_rate=0.03, num_leaves=4, min_child_samples=15, subsample=0.8, subsample_freq=1, reg_lambda=5, verbose=-1)
PA = dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
PB = dict(learning_rate=0.0134, num_leaves=57, min_child_samples=14, subsample=0.583, subsample_freq=1, colsample_bytree=0.949, reg_lambda=0.036, cat_smooth=18.9, min_data_per_group=43, verbose=-1)
SEEDS = (0, 1, 2)

def components(trn, X):
    """Return dict of dry-model preds on X and weather-factor pred on X (yt space)."""
    dm = ~wet_mask(trn).values
    Xd, yd = trn.loc[dm, DF], trn.yt.values[dm]
    out = {}
    out['lgbA'] = np.mean([lgb.LGBMRegressor(**PA, n_estimators=1200, random_state=s).fit(Xd, yd).predict(X[DF]) for s in SEEDS], 0)
    out['lgbB'] = np.mean([lgb.LGBMRegressor(**PB, n_estimators=2956, random_state=s).fit(Xd, yd).predict(X[DF]) for s in SEEDS], 0)
    (out['nn'],) = nn_fp(epochs=80, seeds=(0, 1))(trn[dm].reset_index(drop=True), yd, [X])
    # weather factor relative to lgbA dry pace (out-of-sample on wet rows since dry model never saw them)
    mA = lgb.LGBMRegressor(**PA, n_estimators=1200, random_state=0).fit(Xd, yd)
    d_tr = mA.predict(trn[DF]); wm = wet_mask(trn).values
    wf = np.zeros(len(X)); xw = wet_mask(X).values
    for s in range(5):
        use = wm | (np.random.RandomState(s).rand(len(trn)) < 0.15)
        mw = lgb.LGBMRegressor(**WP, n_estimators=600, random_state=s).fit(trn.loc[use, WF], (trn.yt.values - d_tr)[use])
        if xw.any(): wf[xw] += mw.predict(X.loc[xw, WF]) / 5
    out['wf'] = wf
    return out

def combine(c, w):
    return w[0] * c['lgbA'] + w[1] * c['lgbB'] + w[2] * c['nn'] + c['wf']

if __name__ == '__main__':
    t = time.time()
    comps = {k: np.zeros(len(tr)) for k in ['lgbA', 'lgbB', 'nn', 'wf']}
    f = folds(tr, seed=0)
    for k in range(5):
        c = components(tr[f != k].reset_index(drop=True), tr[f == k])
        for n in comps: comps[n][f == k] = c[n]
        print('fold', k, f'{time.time()-t:.0f}s', flush=True)
    np.savez('../oof/final_oof.npz', **comps)
    y = tr.lap_time_sec.values; cb = np.log(tr.cbase.values)
    dry = ~wet_mask(tr).values; both = ((tr.src_ch1 == 1) & (tr.src_ch5 == 1)).values
    r = lambda o, m: np.sqrt(np.mean((np.exp(o[m] + cb[m]) - y[m]) ** 2))
    for nm in ['lgbA', 'lgbB', 'nn']: print(nm, 'dry', round(r(comps[nm], dry & both), 4))
    best = None
    for a in np.arange(0, 1.01, 0.1):
        for b in np.arange(0, 1.01 - a, 0.1):
            w = (a, b, 1 - a - b); s = r(combine(comps, w), dry & both)
            if best is None or s < best[0]: best = (s, w)
    print('best dry blend', round(best[0], 4), np.round(best[1], 2))
    for nm, w in [('lgbA only (=v5)', (1, 0, 0)), ('equal', (1/3, 1/3, 1/3)), ('best', best[1])]:
        o = combine(comps, w)
        print(f'{nm:18s} full-row CV {r(o, both):.4f}  testcirc {score(tr, np.exp(o + cb))[1]:.4f}  dry {r(o, dry & both):.4f}  wet {r(o, ~dry):.4f}')
