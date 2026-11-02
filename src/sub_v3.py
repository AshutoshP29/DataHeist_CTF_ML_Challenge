import numpy as np, warnings; warnings.filterwarnings('ignore')
from frame import get_data, run_model, lgb_fp, FEATS
from rainfe import add_rain, RAINF2
from submit import write_submission
tr, te = get_data(); tr, te = add_rain(tr, te)
LEAK = ['ride_height_mm', 'brake_bias_pct']
F_base = [f for f in FEATS if f not in LEAK]
F_rain = F_base + RAINF2
P = dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
T = dict(learning_rate=0.0109, num_leaves=28, min_child_samples=23, subsample=0.742, subsample_freq=1, colsample_bytree=0.685, reg_lambda=5.86, cat_smooth=1.76)
run_model('noleak_base', lgb_fp(P, 1200, feats=F_base), tr, te, save=False)
run_model('noleak_rain', lgb_fp(P, 1200, feats=F_rain), tr, te, save=False)
run_model('noleak_rain_tuned', lgb_fp(T, 2154, feats=F_rain), tr, te, save=False)
for nm, F in [('v3_noleak_rain', F_rain), ('v3b_noleak_norain', F_base)]:
    (p,) = lgb_fp(P, 1200, feats=F, seeds=(0, 1, 2))(tr, tr.yt.values, [te])
    write_submission(te.lap_id, np.exp(p + np.log(te.cbase.values)), nm)
