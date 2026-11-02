import pandas as pd, numpy as np, lightgbm as lgb, warnings
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')
from prep import load, SETUP, CATS
CTX = ['season','round','lap','tyre_age_laps','stint','laps_remaining','quali_position','gap_ahead_sec','rain_mm_h','track_temp_c','total_laps','lap_frac']
TEST_CIRC = ['Monaco','Interlagos','Mexico City','Hungaroring','Austin','Bahrain','Suzuka','Silverstone']
P = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_estimators=3000)

def cv(al, feats, train_mask=None, params=P, n=5, seeds=(0,), name='', target='raw', verbose=True):
    tr = al[al.is_test == 0].reset_index(drop=True)
    y = tr.lap_time_sec.values
    if target == 'log': yt = np.log(y); inv = np.exp
    elif target == 'logbase':
        base = np.log(tr['cbase'].values); yt = np.log(y) - base; inv = None
    else: yt = y; inv = None
    if train_mask is not None: trm = train_mask(tr)
    else: trm = np.ones(len(tr), bool)
    oof = np.full(len(tr), np.nan)
    races = tr.race.unique()
    for s in seeds:
        rng = np.random.RandomState(s); perm = dict(zip(races, rng.permutation(len(races)) % n))
        fold = tr.race.map(perm).values
        p = np.zeros(len(tr))
        for k in range(n):
            fit = (fold != k) & trm; va = fold == k
            m = lgb.LGBMRegressor(**params, random_state=s)
            # early stop on held-out race full rows (slightly optimistic but ok for comparison)
            ev = va & (tr.src_ch1 == 1) & (tr.src_ch5 == 1)
            m.fit(tr.loc[fit, feats], yt[fit], eval_set=[(tr.loc[ev, feats], yt[ev])], callbacks=[lgb.early_stopping(200, verbose=False)])
            pr = m.predict(tr.loc[va, feats])
            if target == 'log': pr = np.exp(pr)
            elif target == 'logbase': pr = np.exp(pr + base[va])
            p[va] = pr
        oof = p if s == seeds[0] else oof + p
    oof /= len(seeds)
    both = (tr.src_ch1 == 1) & (tr.src_ch5 == 1); tc = tr.circuit.isin(TEST_CIRC)
    r = lambda msk: np.sqrt(np.mean((oof[msk] - tr.lap_time_sec[msk]) ** 2))
    if verbose: print(f'{name:40s} both:{r(both):.4f}  both&testcirc:{r(both & tc):.4f}  ch5rows:{r(tr.src_ch5 == 1):.4f}')
    return oof

if __name__ == '__main__':
    raw = load(impute=False); imp = load(impute=True)
    base = CATS + CTX
    cv(raw, [f for f in base if f not in ('total_laps','lap_frac')], train_mask=lambda t: t.src_ch5 == 1, name='A: ch5 rows only, no setup')
    cv(raw, base + SETUP, name='B raw merged')
    cv(imp, base + SETUP, name='B imputed')
    cv(imp, base + SETUP, train_mask=lambda t: t.src_ch1 == 1, name='B imputed, ch1 rows only')
    cv(imp, base + SETUP, train_mask=lambda t: (t.src_ch1 == 1) & (t.src_ch5 == 1), name='B both rows only')
