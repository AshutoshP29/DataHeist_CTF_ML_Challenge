"""v8: v7 ensemble design on 4-channel merge (26k train rows) + honest engine mode (ch3)."""
import numpy as np, pandas as pd, lightgbm as lgb, warnings, sys, time; warnings.filterwarnings('ignore')
from twostage import wet_mask
from v8cv import load2, V7DRY, V7WF
import nn
tr, te = load2()
DRYF = V7DRY + ['em_h']; WF = V7WF + ['em_h']
# rebuild NN vocab/stats on merged data, honest engine mode as category
nn.CATC = ['circuit', 'driver', 'team', 'track_status', 'compound', 'season', 'em_h', 'stint']
AL = pd.concat([tr, te])
nn.VOC = {c: {v: i + 1 for i, v in enumerate(sorted(set(str(v) for v in AL[c].astype(object).unique())))} for c in nn.CATC}
nn.STATS = {c: (AL[c].mean(), AL[c].std()) for c in nn.NUMC}
PA = dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
PB = dict(learning_rate=0.0134, num_leaves=57, min_child_samples=14, subsample=0.583, subsample_freq=1, colsample_bytree=0.949, reg_lambda=0.036, cat_smooth=18.9, min_data_per_group=43, verbose=-1)
WP = dict(learning_rate=0.03, num_leaves=4, min_child_samples=15, subsample=0.8, subsample_freq=1, reg_lambda=5, verbose=-1)
SEEDS = (0, 1, 2)

def components(trn, X, dryf=None, wff=None):
    dryf = dryf or DRYF; wff = wff or WF
    dm = ~wet_mask(trn).values; Xd, yd = trn.loc[dm, dryf], trn.yt.values[dm]
    out = {}
    out['lgbA'] = np.mean([lgb.LGBMRegressor(**PA, n_estimators=1200, random_state=s).fit(Xd, yd).predict(X[dryf]) for s in SEEDS], 0)
    out['lgbB'] = np.mean([lgb.LGBMRegressor(**PB, n_estimators=2956, random_state=s).fit(Xd, yd).predict(X[dryf]) for s in SEEDS], 0)
    (out['nn'],) = nn.nn_fp(epochs=80, seeds=(0, 1))(trn[dm].reset_index(drop=True), yd, [X])
    mA = lgb.LGBMRegressor(**PA, n_estimators=1200, random_state=0).fit(Xd, yd)
    d_tr = mA.predict(trn[dryf]); wm = wet_mask(trn).values
    wf = np.zeros(len(X)); xw = wet_mask(X).values
    for s in range(5):
        use = wm | (np.random.RandomState(s).rand(len(trn)) < 0.15)
        mw = lgb.LGBMRegressor(**WP, n_estimators=600, random_state=s).fit(trn.loc[use, wff], (trn.yt.values - d_tr)[use])
        if xw.any(): wf[xw] += mw.predict(X.loc[xw, wff]) / 5
    out['wf'] = wf
    return out

def combine(c): return (c['lgbA'] + c['lgbB'] + c['nn']) / 3 + c['wf']

if __name__ == '__main__':
    from submit import write_submission
    c = components(tr, te); np.savez('../oof/v8_test_components.npz', **c)
    pred = np.exp(combine(c) + np.log(te.cbase.values))
    v7 = pd.read_csv('../submissions/v7_ensemble_noEM.csv')
    p7 = te[['lap_id']].merge(v7, on='lap_id').lap_time_sec.values
    print('rms diff vs v7', np.sqrt(np.mean((pred - p7) ** 2)).round(3), ' mean diff by engine mode', pd.Series(pred - p7).groupby(te.em_h.values).mean().round(3).to_dict())
    write_submission(te.lap_id, pred, 'v8_4ch_honestEM')
