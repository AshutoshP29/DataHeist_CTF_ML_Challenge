import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
from frame import get_data, folds, lgb_fp, FEATS
tr, te = get_data()
P = dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
f = folds(tr); oof = np.zeros(len(tr))
for k in range(5):
    (p,) = lgb_fp(P, 1200)(tr[f != k].reset_index(drop=True), tr.yt.values[f != k], [tr[f == k]]); oof[f == k] = np.exp(p + np.log(tr.cbase.values[f == k]))
tr['se'] = (oof - tr.lap_time_sec) ** 2
def seg(d):
    rain = pd.cut(d.rain_mm_h.fillna(0), [-1, 0, 1, 3, 20], labels=['dry', 'r<1', 'r1-3', 'r>3'])
    return d.track_status.astype(str) + '|' + d.compound.astype(str) + '|' + rain.astype(str)
tr['seg'] = seg(tr); te['seg'] = seg(te)
both = tr[(tr.src_ch1 == 1) & (tr.src_ch5 == 1)]
for nm, src in [('both', both), ('all', tr)]:
    m = src.groupby('seg').se.agg(['mean', 'count'])
    tc = te.seg.value_counts()
    j = m.join(tc.rename('te'), how='right')
    miss = j['mean'].isna()
    j['mean'] = j['mean'].fillna(src.se.mean() * 4)
    print(nm, 'plain RMSE', np.sqrt(src.se.mean()).round(3), ' test-mix RMSE', np.sqrt((j['mean'] * j.te).sum() / j.te.sum()).round(3), ' test rows in unseen segs:', j.te[miss].sum())
j = tr.groupby('seg').se.agg(['mean', 'count']).join(te.seg.value_counts().rename('te'), how='outer').fillna(0)
j['rmse'] = np.sqrt(j['mean']); j['te_contrib'] = j['mean'] * j.te / j.te.sum()
print(j.sort_values('te_contrib', ascending=False).round(3).head(25))
# covariate shift: test vs train-both numeric
for c in ['tyre_age_laps', 'lap', 'gap_ahead_sec', 'track_temp_c', 'rain_mm_h', 'quali_position', 'wing_level', 'ride_height_mm', 'brake_bias_pct', 'engine_mode', 'suspension_stiffness', 'cooling_level', 'pu_age_races', 'total_laps']:
    print(f'{c:22s} train-both mean {both[c].mean():8.3f} std {both[c].std():7.3f} | test mean {te[c].mean():8.3f} std {te[c].std():7.3f}')
print(pd.crosstab(te.race, te.engine_mode)); print(pd.crosstab(both.circuit, both.engine_mode))
