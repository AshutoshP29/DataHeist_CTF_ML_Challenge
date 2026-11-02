import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
from frame import get_data, lgb_fp, FEATS, folds
from prep import SETUP
from rainfe import add_rain, RAINF2
tr, te = get_data(); tr, te = add_rain(tr, te)
NOSET = [f for f in FEATS + RAINF2 if f not in SETUP]
P = dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
# ctx-only model predictions: OOF for train, full fit for test
f = folds(tr); oof = np.zeros(len(tr))
for k in range(5):
    (p,) = lgb_fp(P, 1200, feats=NOSET)(tr[f != k].reset_index(drop=True), tr.yt.values[f != k], [tr[f == k]]); oof[f == k] = p
(pt,) = lgb_fp(P, 1200, feats=NOSET)(tr, tr.yt.values, [te])
tr['ctx'] = oof; te['ctx'] = pt
b = tr[(tr.src_ch1 == 1)].copy(); b['res'] = b.yt - b.ctx   # what setup must explain
# within-circuit demeaned correlations
def dm(d, c): return d[c] - d.groupby(d.circuit.astype(str))[c].transform('mean')
print(f'{"feat":22s} train corr(feat, yt-resid)  train corr(feat, ctx pred)  test corr(feat, ctx pred)')
for c in SETUP:
    print(f'{c:22s} {np.corrcoef(dm(b,c).fillna(0), b.res)[0,1]:+.3f}            {pd.Series(dm(b,c)).corr(pd.Series(dm(b,"ctx"))):+.3f}                     {pd.Series(dm(te,c)).corr(pd.Series(dm(te,"ctx"))):+.3f}')
print('\ntrain corr(feat, yt) within circuit:', {c: round(pd.Series(dm(b, c)).corr(dm(b, 'yt')), 3) for c in SETUP})
# within ch1-only train rows (ctx imputed) vs both rows
for nm, d in [('ch1-only', tr[(tr.src_ch1==1)&(tr.src_ch5==0)]), ('both', tr[(tr.src_ch1==1)&(tr.src_ch5==1)]), ('test', te)]:
    print(nm, 'corr ride~brake', d[['ride_height_mm','brake_bias_pct']].corr().iloc[0,1].round(3), ' ride~engine', d[['ride_height_mm','engine_mode']].corr().iloc[0,1].round(3), ' brake~engine', d[['brake_bias_pct','engine_mode']].corr().iloc[0,1].round(3))
