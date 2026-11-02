import warnings, numpy as np, pandas as pd, lightgbm as lgb, os, time; warnings.filterwarnings('ignore')
from prep import load, add_base, SETUP, CATS
from fe import add_fe
from cv import CTX, TEST_CIRC
FE = ['is_slick','comp_ord','rain_slick','rain_inter','rain_wet','tyre_mismatch','status_ord','stint_lap']
FEATS = CATS + CTX + SETUP + FE
OUT = '../oof/'; os.makedirs(OUT, exist_ok=True)

def get_data():
    al = add_fe(add_base(load(impute=True)))
    tr = al[al.is_test == 0].reset_index(drop=True); te = al[al.is_test == 1].reset_index(drop=True)
    tr['yt'] = np.log(tr.lap_time_sec) - np.log(tr.cbase)
    return tr, te

def folds(tr, n=5, seed=0):
    races = np.sort(tr.race.unique()); rng = np.random.RandomState(seed)
    return tr.race.map(dict(zip(races, rng.permutation(len(races)) % n))).values

def score(tr, oof):
    both = ((tr.src_ch1 == 1) & (tr.src_ch5 == 1)).values; tc = tr.circuit.isin(TEST_CIRC).values
    r = lambda m: np.sqrt(np.mean((oof[m] - tr.lap_time_sec.values[m]) ** 2))
    return r(both), r(both & tc)

def run_model(name, fit_predict, tr, te, fold_seeds=(0, 1), save=True, verbose=True):
    """fit_predict(Xtr_df, ytr, Xva_df_list) -> list of preds (in yt space)."""
    t = time.time(); oofs = []
    for fs in fold_seeds:
        f = folds(tr, seed=fs); oof = np.zeros(len(tr))
        for k in range(5):
            fi = f != k; va = f == k
            (p,) = fit_predict(tr[fi].reset_index(drop=True), tr.yt.values[fi], [tr[va]])
            oof[va] = np.exp(p + np.log(tr.cbase.values[va]))
        oofs.append(oof)
    sc = [score(tr, o) for o in oofs]
    b = np.mean([s[0] for s in sc]); c = np.mean([s[1] for s in sc])
    if verbose: print(f'{name:35s} both {b:.4f}  both&tc {c:.4f}   ({time.time()-t:.0f}s)', flush=True)
    if save:
        (pt,) = fit_predict(tr, tr.yt.values, [te])
        np.save(OUT + name + '_oof.npy', oofs[0]); np.save(OUT + name + '_test.npy', np.exp(pt + np.log(te.cbase.values)))
    return b, c

def lgb_fp(params, n_est, feats=FEATS, weight_both=1.0, seeds=(0,)):
    def fp(Xtr, y, Xs):
        w = np.where((Xtr.src_ch1 == 1) & (Xtr.src_ch5 == 1), weight_both, 1.0)
        preds = [np.zeros(len(X)) for X in Xs]
        for s in seeds:
            m = lgb.LGBMRegressor(**params, n_estimators=n_est, random_state=s, verbose=-1).fit(Xtr[feats], y, sample_weight=w)
            for i, X in enumerate(Xs): preds[i] += m.predict(X[feats]) / len(seeds)
        return preds
    return fp

if __name__ == '__main__':
    tr, te = get_data()
    P = dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
    for n in [600, 1200, 2500]:
        run_model(f'lgb_n{n}', lgb_fp(P, n), tr, te, save=False)
    for w in [2.0, 4.0]:
        run_model(f'lgb_w{w}', lgb_fp(P, 1200, weight_both=w), tr, te, save=False)
