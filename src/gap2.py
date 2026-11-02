import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
from frame import get_data, folds, lgb_fp, FEATS
from rainfe import add_rain, RAINF2
tr, te = get_data(); tr, te = add_rain(tr, te)
F = [f for f in FEATS if f not in ('ride_height_mm', 'brake_bias_pct')]
P = dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
f = folds(tr); oof = np.zeros(len(tr))
for k in range(5):
    (p,) = lgb_fp(P, 1200, feats=F)(tr[f != k].reset_index(drop=True), tr.yt.values[f != k], [tr[f == k]]); oof[f == k] = np.exp(p + np.log(tr.cbase.values[f == k]))
tr['oof'] = oof; tr['se'] = (oof - tr.lap_time_sec) ** 2; tr['res'] = tr.lap_time_sec - oof
np.save('../oof/noleak_oof.npy', oof)
def seg(d):
    return d.track_status.astype(str) + '|' + np.where(d.is_slick == 1, 'slick', d.compound.astype(str)) + '|' + pd.cut(d.rain_mm_h.fillna(0), [-1, 0, 1, 3, 20], labels=['dry', 'r<1', 'r1-3', 'r>3']).astype(str)
tr['seg'] = seg(tr); te['seg'] = seg(te)
both = tr[(tr.src_ch1 == 1) & (tr.src_ch5 == 1)]
for nm, src in [('both', both), ('all', tr)]:
    j = src.groupby('seg').se.mean().rename('mse').to_frame().join(te.seg.value_counts().rename('te'), how='right')
    j['mse'] = j.mse.fillna(src.se.mean() * 4)
    print(nm, 'plain', np.sqrt(src.se.mean()).round(3), 'test-mix', np.sqrt((j.mse * j.te).sum() / j.te.sum()).round(3))
j = tr.groupby('seg').agg(mse=('se', 'mean'), bias=('res', 'mean'), n=('se', 'size')).join(te.seg.value_counts().rename('te'), how='outer').fillna(0)
j['rmse'] = np.sqrt(j.mse); j['contrib'] = j.mse * j.te / j.te.sum()
print(j.sort_values('contrib', ascending=False).round(3).head(14))
print('per-race RMSE (both rows):'); print(both.groupby('race').se.mean().pow(.5).describe().round(3))
print('race-mean residual std:', both.groupby('race').res.mean().std().round(3))
