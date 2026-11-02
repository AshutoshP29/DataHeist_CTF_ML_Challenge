import warnings; warnings.filterwarnings('ignore')
from twostage import cv, DRYF
WF = ['compound', 'rain_mm_h', 'rain_lag1', 'status_ord', 'tyre_age_laps', 'laps_since_rain', 'rain_ema5', 'engine_mode', 'rain_max_so_far', 'rain_ema2']
WP = dict(learning_rate=0.03, num_leaves=4, min_child_samples=15, subsample=0.8, subsample_freq=1, reg_lambda=5)
print(DRYF)
weak = ['wing_level', 'suspension_stiffness', 'pu_age_races', 'cooling_level']
cv('v4 dry', WF, WP, 600)
cv('dry -weak setup', WF, WP, 600, dryf=[f for f in DRYF if f not in weak])
cv('dry -weak -driver/team', WF, WP, 600, dryf=[f for f in DRYF if f not in weak + ['driver', 'team']])
cv('dry -weak -gap/quali', WF, WP, 600, dryf=[f for f in DRYF if f not in weak + ['gap_ahead_sec', 'quali_position']])
cv('dry -weak -round/total', WF, WP, 600, dryf=[f for f in DRYF if f not in weak + ['round', 'total_laps']])
