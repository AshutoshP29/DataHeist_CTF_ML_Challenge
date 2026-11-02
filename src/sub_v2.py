import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from frame import get_data, run_model, lgb_fp, FEATS, folds
from rainfe import add_rain, RAINF2
from submit import write_submission
tr, te = get_data(); tr, te = add_rain(tr, te)
T = dict(learning_rate=0.0109, num_leaves=28, min_child_samples=23, subsample=0.742, subsample_freq=1, colsample_bytree=0.685, reg_lambda=5.86, cat_smooth=1.76)
F = FEATS + RAINF2
run_model('lgb_rain_tuned', lgb_fp(T, 2154, feats=F), tr, te, save=False)
# segment errors
f = folds(tr); oof = np.zeros(len(tr))
for k in range(5):
    (p,) = lgb_fp(T, 2154, feats=F)(tr[f != k].reset_index(drop=True), tr.yt.values[f != k], [tr[f == k]]); oof[f == k] = np.exp(p + np.log(tr.cbase.values[f == k]))
tr['se'] = (oof - tr.lap_time_sec) ** 2
wet = (tr.rain_mm_h > 0) | ~tr.is_slick.astype(bool)
print('RMSE dry-slick', np.sqrt(tr.se[~wet].mean()).round(3), ' wet/rain rows', np.sqrt(tr.se[wet].mean()).round(3), 'n', wet.sum())
(p,) = lgb_fp(T, 2154, feats=F, seeds=(0, 1, 2))(tr, tr.yt.values, [te])
write_submission(te.lap_id, np.exp(p + np.log(te.cbase.values)), 'v2_lgb_rain_tuned')
