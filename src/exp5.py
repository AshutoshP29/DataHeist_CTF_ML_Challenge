import warnings; warnings.filterwarnings('ignore')
from twostage import cv
B = ['compound', 'rain_mm_h', 'rain_lag1', 'status_ord', 'tyre_age_laps']
WP = dict(learning_rate=0.03, num_leaves=7, min_child_samples=15, subsample=0.8, subsample_freq=1, reg_lambda=5)
cv('B', B, WP, 300)
cv('B+engine_mode', B + ['engine_mode'], WP, 300)
cv('B+temp', B + ['track_temp_c'], WP, 300)
cv('B+lead1+trend', B + ['rain_lead1', 'rain_trend'], WP, 300)
cv('B+laps_since', B + ['laps_since_rain', 'rain_ema5'], WP, 300)
cv('B leaves15 n500', B, dict(WP, num_leaves=15), 500)
cv('B leaves4 n600', B, dict(WP, num_leaves=4), 600)
