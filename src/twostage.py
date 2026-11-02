import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
from frame import get_data, folds, FEATS, score
from rainfe import add_rain
tr, te = get_data(); tr, te = add_rain(tr, te)
LEAK = ['ride_height_mm', 'brake_bias_pct']
RAINISH = ['rain_mm_h', 'rain_slick', 'rain_inter', 'rain_wet', 'tyre_mismatch', 'comp_ord', 'is_slick']
DRYF = [f for f in FEATS if f not in LEAK + RAINISH]
JOINTF = [f for f in FEATS if f not in LEAK]
P = dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

def wet_mask(d): return (d.is_slick == 0) | (d.rain_mm_h > 0)

def fit_predict(trn, Xs, wf_feats, wf_params, wf_n, dry_n=1200, seed=0, dryf=None, dryp=None):
    DRYF_ = dryf or DRYF; P_ = dryp or P
    dm = ~wet_mask(trn)
    md = lgb.LGBMRegressor(**P_, n_estimators=dry_n, random_state=seed).fit(trn.loc[dm, DRYF_], trn.yt[dm])
    # weather factor on wet rows (dry model is out-of-sample there) + a sample of dry rows anchoring factor≈0
    wm = wet_mask(trn)
    d_tr = md.predict(trn[DRYF_])
    use = wm | (np.random.RandomState(seed).rand(len(trn)) < 0.15)
    mw = lgb.LGBMRegressor(**wf_params, n_estimators=wf_n, random_state=seed, verbose=-1).fit(trn.loc[use, wf_feats], (trn.yt - d_tr)[use])
    out = []
    for X in Xs:
        p = md.predict(X[DRYF_])
        w = wet_mask(X).values
        if w.any(): p[w] += mw.predict(X.loc[w, wf_feats])
        out.append(p)
    return out

def cv(name, wf_feats, wf_params, wf_n, fold_seeds=(0, 1), **kw):
    sc = []; oofs = []
    for fs in fold_seeds:
        f = folds(tr, seed=fs); oof = np.zeros(len(tr))
        for k in range(5):
            (p,) = fit_predict(tr[f != k].reset_index(drop=True), [tr[f == k]], wf_feats, wf_params, wf_n, **kw)
            oof[f == k] = np.exp(p + np.log(tr.cbase.values[f == k]))
        sc.append(score(tr, oof)); oofs.append(oof)
    o = oofs[0]; w = wet_mask(tr).values; se = (o - tr.lap_time_sec.values) ** 2
    print(f'{name:30s} both {np.mean([s[0] for s in sc]):.4f} tc {np.mean([s[1] for s in sc]):.4f} | dry {np.sqrt(se[~w].mean()):.3f} wet {np.sqrt(se[w].mean()):.3f} wetbias {(tr.lap_time_sec.values - o)[w].mean():+.3f}', flush=True)
    return oofs[0]

if __name__ == '__main__':
    WP = dict(learning_rate=0.03, num_leaves=7, min_child_samples=15, subsample=0.8, subsample_freq=1, reg_lambda=5)
    cv('wf: compound+rain', ['compound', 'rain_mm_h'], WP, 300)
    cv('wf: +lag1', ['compound', 'rain_mm_h', 'rain_lag1'], WP, 300)
    cv('wf: +lag1+status+age', ['compound', 'rain_mm_h', 'rain_lag1', 'status_ord', 'tyre_age_laps'], WP, 300)
    cv('wf: +lag1+circuit', ['compound', 'rain_mm_h', 'rain_lag1', 'circuit'], WP, 300)
    # joint reference
    from frame import lgb_fp
    sc = []
    f = folds(tr); oof = np.zeros(len(tr))
    for k in range(5):
        (p,) = lgb_fp({kk: v for kk, v in P.items() if kk != 'verbose'}, 1200, feats=JOINTF)(tr[f != k].reset_index(drop=True), tr.yt.values[f != k], [tr[f == k]]); oof[f == k] = np.exp(p + np.log(tr.cbase.values[f == k]))
    w = wet_mask(tr).values; se = (oof - tr.lap_time_sec.values) ** 2
    print(f'joint ref (seed0)               both {score(tr, oof)[0]:.4f} | dry {np.sqrt(se[~w].mean()):.3f} wet {np.sqrt(se[w].mean()):.3f} wetbias {(tr.lap_time_sec.values - oof)[w].mean():+.3f}')
