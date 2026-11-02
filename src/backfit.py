import warnings, numpy as np, pandas as pd, lightgbm as lgb; warnings.filterwarnings('ignore')
from prep import load, add_base, SETUP, CATS
from fe import add_fe
from cv import CTX, TEST_CIRC
FE = ['is_slick','comp_ord','rain_slick','rain_inter','rain_wet','tyre_mismatch','status_ord','stint_lap']
CTXF = CATS + CTX + FE
SETF = SETUP + ['circuit','track_status','status_ord','compound','is_slick','rain_mm_h']
PF = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

def fit_backfit(tr, yt, iters=4, nf=1500, ng=1500, seed=0):
    has1 = (tr.src_ch1 == 1).values
    g = np.zeros(len(tr)); models = []
    for it in range(iters):
        mf = lgb.LGBMRegressor(**PF, n_estimators=nf, random_state=seed).fit(tr[CTXF], yt - g)
        f = mf.predict(tr[CTXF])
        mg = lgb.LGBMRegressor(**PF, n_estimators=ng, random_state=seed).fit(tr.loc[has1, SETF], (yt - f)[has1])
        g = np.where(has1, mg.predict(tr[SETF]), 0.0)
        # center g on ch1 rows so rows without setup (g=0) are unbiased
        models = (mf, mg)
    return models

def run(al, iters=3, nf=800, ng=800, n=5, seed=0):
    tr = al[al.is_test == 0].reset_index(drop=True)
    yt = np.log(tr.lap_time_sec.values) - np.log(tr.cbase.values)
    races = tr.race.unique(); rng = np.random.RandomState(seed)
    fold = tr.race.map(dict(zip(races, rng.permutation(len(races)) % n))).values
    oof = np.zeros(len(tr))
    for k in range(n):
        fi = fold != k; va = fold == k
        mf, mg = fit_backfit(tr[fi].reset_index(drop=True), yt[fi], iters, nf, ng, seed)
        oof[va] = np.exp(mf.predict(tr.loc[va, CTXF]) + mg.predict(tr.loc[va, SETF]) + np.log(tr.cbase[va]))
    both = ((tr.src_ch1 == 1) & (tr.src_ch5 == 1)).values; tc = tr.circuit.isin(TEST_CIRC).values
    r = lambda m: np.sqrt(np.mean((oof[m] - tr.lap_time_sec.values[m]) ** 2))
    print(f'backfit it={iters} nf={nf} ng={ng}: both {r(both):.4f}  both&tc {r(both & tc):.4f}')
    return oof

if __name__ == '__main__':
    al = add_fe(add_base(load(impute=True)))
    for it, nf, ng in [(1, 800, 800), (3, 800, 800), (3, 400, 400), (5, 600, 300)]:
        run(al, it, nf, ng)
