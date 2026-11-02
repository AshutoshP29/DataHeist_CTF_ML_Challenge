import optuna, numpy as np, lightgbm as lgb, json, warnings; warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
from drycv import dry_oof, DF
def obj(t):
    P = dict(learning_rate=t.suggest_float('lr', 0.01, 0.08, log=True), num_leaves=t.suggest_int('num_leaves', 7, 96, log=True),
             min_child_samples=t.suggest_int('mcs', 5, 80, log=True), subsample=t.suggest_float('subsample', 0.5, 1.0), subsample_freq=1,
             colsample_bytree=t.suggest_float('colsample', 0.3, 1.0), reg_lambda=t.suggest_float('l2', 1e-3, 30, log=True),
             cat_smooth=t.suggest_float('cat_smooth', 1, 50, log=True), min_data_per_group=t.suggest_int('mdpg', 5, 100, log=True), verbose=-1)
    n = t.suggest_int('n_est', 300, 4000, log=True)
    fp = lambda X, y, Xs: [lgb.LGBMRegressor(**P, n_estimators=n).fit(X[DF], y).predict(Xs[0][DF])]
    _, r = dry_oof(fp)
    print(t.number, round(r, 4), t.params, flush=True); return r
st = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1)); st.optimize(obj, n_trials=70)
print('BEST', st.best_value, st.best_params); json.dump(st.best_params, open('../oof/dry_lgb_best.json', 'w'))
