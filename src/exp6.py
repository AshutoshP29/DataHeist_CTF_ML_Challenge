import warnings; warnings.filterwarnings('ignore')
from twostage import cv
B = ['compound', 'rain_mm_h', 'rain_lag1', 'status_ord', 'tyre_age_laps', 'laps_since_rain', 'rain_ema5']
WP = dict(learning_rate=0.03, num_leaves=7, min_child_samples=15, subsample=0.8, subsample_freq=1, reg_lambda=5)
cv('C', B, WP, 300)
cv('C leaves4 n600', B, dict(WP, num_leaves=4), 600)
cv('C+em leaves4', B + ['engine_mode'], dict(WP, num_leaves=4), 600)
cv('C+em+maxsofar l4', B + ['engine_mode', 'rain_max_so_far', 'rain_ema2'], dict(WP, num_leaves=4), 600)
cv('C+em l4 n1000', B + ['engine_mode'], dict(WP, num_leaves=4), 1000)
cv('C+em l5 mcs30', B + ['engine_mode'], dict(WP, num_leaves=5, min_child_samples=30), 600)
