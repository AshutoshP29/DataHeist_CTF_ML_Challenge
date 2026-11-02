import numpy as np, pandas as pd, warnings, sys; warnings.filterwarnings('ignore')
from catboost import CatBoostRegressor
import xgboost as xgb
from frame import get_data, run_model, FEATS
from prep import CATS
tr, te = get_data()

def cb_fp(params, feats=FEATS):
    def fp(Xtr, y, Xs):
        def prep(X):
            X = X[feats].copy()
            for c in CATS: X[c] = X[c].astype(str)
            return X
        m = CatBoostRegressor(**params, cat_features=CATS, verbose=0, thread_count=8).fit(prep(Xtr), y)
        return [m.predict(prep(X)) for X in Xs]
    return fp

def xgb_fp(params, n, feats=FEATS):
    def fp(Xtr, y, Xs):
        m = xgb.XGBRegressor(**params, n_estimators=n, enable_categorical=True, tree_method='hist', n_jobs=8).fit(Xtr[feats], y)
        return [m.predict(X[feats]) for X in Xs]
    return fp

which = sys.argv[1]
if which == 'cb':
    for d, it in [(6, 3000), (8, 3000)]:
        run_model(f'cb_d{d}', cb_fp(dict(depth=d, iterations=it, learning_rate=0.03, l2_leaf_reg=3, random_seed=0)), tr, te, save=False)
if which == 'xgb':
    for d, n in [(6, 1500), (8, 1000)]:
        run_model(f'xgb_d{d}', xgb_fp(dict(max_depth=d, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1, max_cat_to_onehot=1), n), tr, te, save=False)
