import optuna, numpy as np, warnings, json; warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
from frame import get_data, run_model, lgb_fp
tr, te = get_data()
def obj(t):
    P = dict(learning_rate=t.suggest_float('lr', 0.01, 0.1, log=True),
             num_leaves=t.suggest_int('num_leaves', 7, 127, log=True),
             min_child_samples=t.suggest_int('mcs', 5, 100, log=True),
             subsample=t.suggest_float('subsample', 0.5, 1.0), subsample_freq=1,
             colsample_bytree=t.suggest_float('colsample', 0.3, 1.0),
             reg_lambda=t.suggest_float('l2', 1e-3, 30, log=True),
             min_split_gain=0.0, max_bin=t.suggest_categorical('max_bin', [63, 255, 511]),
             cat_smooth=t.suggest_float('cat_smooth', 1, 50, log=True))
    n = t.suggest_int('n_est', 300, 4000, log=True)
    b, c = run_model(f't{t.number}', lgb_fp(P, n), tr, te, save=False, verbose=False)
    print(t.number, round(b, 4), round(c, 4), t.params, flush=True)
    return b
st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=0))
st.optimize(obj, n_trials=80)
print('BEST', st.best_value, st.best_params)
json.dump(st.best_params, open('../oof/lgb_best.json', 'w'))
