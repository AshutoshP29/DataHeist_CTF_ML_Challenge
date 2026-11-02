import numpy as np, warnings; warnings.filterwarnings('ignore')
from frame import get_data, run_model, lgb_fp, FEATS
from rainfe import add_rain, RAINF2
tr, te = get_data(); tr, te = add_rain(tr, te)
P = dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
run_model('lgb_base', lgb_fp(P, 1200), tr, te, save=False)
run_model('lgb_rain', lgb_fp(P, 1200, feats=FEATS + RAINF2), tr, te, save=False)
