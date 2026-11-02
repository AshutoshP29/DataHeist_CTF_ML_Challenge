import numpy as np
from frame import get_data, lgb_fp
from submit import write_submission
tr, te = get_data()
P = dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
(p,) = lgb_fp(P, 1200, seeds=(0, 1, 2))(tr, tr.yt.values, [te])
write_submission(te.lap_id, np.exp(p + np.log(te.cbase.values)), 'v1_lgb_logbase')
