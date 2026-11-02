import numpy as np, warnings; warnings.filterwarnings('ignore')
from twostage import fit_predict, tr, te
from drycv import DF
from submit import write_submission
WF = ['compound', 'rain_mm_h', 'rain_lag1', 'status_ord', 'tyre_age_laps', 'laps_since_rain', 'rain_ema5', 'engine_mode', 'rain_max_so_far', 'rain_ema2']
WP = dict(learning_rate=0.03, num_leaves=4, min_child_samples=15, subsample=0.8, subsample_freq=1, reg_lambda=5)
p = np.mean([fit_predict(tr, [te], WF, WP, 600, seed=s, dryf=DF)[0] for s in range(5)], axis=0)
pred = np.exp(p + np.log(te.cbase.values)); np.save('../oof/v5_test.npy', pred)
write_submission(te.lap_id, pred, 'v5_twostage_noweak')
