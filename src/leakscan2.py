import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from frame import folds
from twostage import wet_mask
tr = pd.read_pickle('../oof/tr2.pkl'); te = pd.read_pickle('../oof/te2.pkl')
BASE = ['circuit', 'driver', 'team', 'track_status', 'compound', 'season', 'lap', 'tyre_age_laps', 'stint', 'laps_remaining', 'quali_position', 'track_temp_c', 'lap_frac', 'status_ord', 'stint_lap']
NEW = ['fuel_load_kg', 'fuel_used_kg_2', 'speed_trap_kmh', 'full_throttle_pct', 'brake_temp_c', 'tyre_temp_c', 'ers_mj', 'gear_shifts', 'drs_open_s', 'fuel_used_kg_3', 'fuel_remaining_kg', 'engine_mode_3', 'wing_level_3', 'engine_mode_1', 'wing_level_1']
P = dict(learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
# context prediction without any new feature (OOF on train, full fit for test)
f = folds(tr); o = np.zeros(len(tr))
for k in range(5):
    o[f == k] = lgb.LGBMRegressor(**P, n_estimators=800).fit(tr.loc[f != k, BASE], tr.yt[f != k]).predict(tr.loc[f == k, BASE])
pt = lgb.LGBMRegressor(**P, n_estimators=800).fit(tr[BASE], tr.yt).predict(te[BASE])
tr['ctx'] = o; te['ctx'] = pt; tr['res'] = tr.yt - tr.ctx
gtr = (tr.track_status.astype(str) == 'GREEN') & ~wet_mask(tr); gte = (te.track_status.astype(str) == 'GREEN') & ~wet_mask(te)
dm = lambda d, c: d[c] - d.groupby('circuit')[c].transform('mean')
print(f'{"feature":18s} {"n_tr":>6s}  corr(feat,ctx) train | test   corr(feat, residual) train   <- residual = info beyond context')
for c in NEW:
    a = tr[gtr & tr[c].notna()]; b = te[gte & te[c].notna()]
    ca = np.corrcoef(dm(a, c), dm(a, 'ctx'))[0, 1]; cb_ = np.corrcoef(dm(b, c), dm(b, 'ctx'))[0, 1]; cr = np.corrcoef(dm(a, c), a.res)[0, 1]
    flag = '  <-- SHIFT' if abs(ca - cb_) > 0.15 else ''
    print(f'{c:18s} {len(a):6d}  {ca:+.3f}        | {cb_:+.3f}       {cr:+.3f}{flag}')
# engine_mode_3 level pattern on green laps
for nm, d in [('train', tr[gtr]), ('test', te[gte])]:
    print(nm, 'mean ctx (pct) by engine_mode_3:', (d.groupby('engine_mode_3').ctx.mean() * 100).round(2).to_dict(), ' by engine_mode_1:', (d.groupby('engine_mode_1').ctx.mean() * 100).round(2).to_dict())
a = tr[gtr]
print('train: mean residual (pct) by engine_mode_3:', (a.groupby('engine_mode_3').res.mean() * 100).round(2).to_dict())
print('train: mean residual (pct) by engine_mode_1:', (a.groupby('engine_mode_1').res.mean() * 100).round(2).to_dict())
