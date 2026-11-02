import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from frame import folds
from twostage import tr, te, wet_mask, DRYF
import lightgbm as lgb

def lin_X(d, cols_ref=None):
    X = pd.DataFrame(index=d.index)
    for c in ['circuit', 'driver', 'team', 'track_status', 'compound']:
        X = X.join(pd.get_dummies(d[c].astype(str), prefix=c).astype(float))
    X = X.join(pd.get_dummies(d.season.astype(int), prefix='season').astype(float))
    em = d.engine_mode.fillna(-1).astype(int)
    X = X.join(pd.get_dummies(em, prefix='em').astype(float))
    comp = d.compound.astype(str)
    for c in comp.unique():
        m = (comp == c).astype(float); a = d.tyre_age_laps
        X[f'age_{c}'] = m * a; X[f'age2_{c}'] = m * a ** 2; X[f'age3_{c}'] = m * a ** 3 / 100
    for circ in d.circuit.astype(str).unique():
        m = (d.circuit.astype(str) == circ).astype(float)
        X[f'deg_{circ}'] = m * d.tyre_age_laps; X[f'fuel_{circ}'] = m * d.laps_remaining
    X['laps_rem'] = d.laps_remaining; X['lap_frac'] = d.lap_frac; X['lap_frac2'] = d.lap_frac ** 2
    t = d.track_temp_c.fillna(40) - 40; X['temp'] = t; X['temp2'] = t ** 2
    X['quali'] = d.quali_position.fillna(10.5); X['quali_na'] = d.quali_position.isna().astype(float)
    g = d.gap_ahead_sec; X['gap_na'] = g.isna().astype(float); g = g.fillna(2.4)
    X['gap'] = np.minimum(g, 5); X['dirty'] = np.exp(-g); X['drs'] = (g < 1).astype(float)
    X['stint'] = d.stint.fillna(2); X['lap1'] = (d['lap'] == 1).astype(float)
    X['first_lap_stint'] = (d.tyre_age_laps <= 1).astype(float)
    if cols_ref is not None: X = X.reindex(columns=cols_ref, fill_value=0.0)
    return X

def lin_fp(trn, Xs, alpha=1.0):
    dm = ~wet_mask(trn)
    Xt = lin_X(trn[dm]); mu, sd = Xt.mean(), Xt.std().replace(0, 1)
    m = Ridge(alpha=alpha).fit((Xt - mu) / sd, trn.yt[dm])
    return [m.predict((lin_X(X, Xt.columns) - mu) / sd) for X in Xs]

if __name__ == '__main__':
    dry = ~wet_mask(tr).values; both = ((tr.src_ch1 == 1) & (tr.src_ch5 == 1)).values
    P = dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    weak = ['wing_level', 'suspension_stiffness', 'pu_age_races', 'cooling_level']
    DF = [f for f in DRYF if f not in weak]
    for alpha in [1, 10, 50]:
        res = {}
        for fs in (0, 1):
            f = folds(tr, seed=fs); ol = np.zeros(len(tr)); og = np.zeros(len(tr))
            for k in range(5):
                trn = tr[f != k].reset_index(drop=True); va = tr[f == k]
                (ol[f == k],) = lin_fp(trn, [va], alpha)
                if alpha == 1:
                    dm = ~wet_mask(trn)
                    og[f == k] = lgb.LGBMRegressor(**P, n_estimators=1200).fit(trn.loc[dm, DF], trn.yt[dm]).predict(va[DF])
            y = tr.lap_time_sec.values; cb = tr.cbase.values
            r = lambda o, m: np.sqrt(np.mean((np.exp(o + np.log(cb)) - y)[m] ** 2))
            res.setdefault('lin', []).append(r(ol, dry & both))
            if alpha == 1:
                res.setdefault('gbm', []).append(r(og, dry & both))
                for w in (0.3, 0.5, 0.7): res.setdefault(f'blend{w}', []).append(r(w * ol + (1 - w) * og, dry & both))
        print('alpha', alpha, {k: round(np.mean(v), 4) for k, v in res.items()}, flush=True)
