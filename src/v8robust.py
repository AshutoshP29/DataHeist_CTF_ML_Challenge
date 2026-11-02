import numpy as np, pandas as pd, time, warnings; warnings.filterwarnings('ignore')
from frame import folds, TEST_CIRC
from twostage import wet_mask
import final2
from final2 import components, combine, tr
from v8cv import V7DRY, V7WF
y = tr.lap_time_sec.values; cb = np.log(tr.cbase.values)
tl = ((tr.src_ch5 == 1) & (tr.src_ch3 == 1)).values
for FS in (7, 0):
    f = folds(tr, seed=FS); res = {}
    for nm, dryf, wff in [('v7cfg', V7DRY, V7WF), ('v8', None, None)]:
        o = np.zeros(len(tr)); t = time.time()
        for k in range(5):
            o[f == k] = combine(components(tr[f != k].reset_index(drop=True), tr[f == k], dryf, wff))
        res[nm] = o; np.save(f'../oof/{nm}_oof_fs{FS}.npy', o)
        print(f'split {FS} {nm}: test-like RMSE {np.sqrt(np.mean((np.exp(o[tl] + cb[tl]) - y[tl]) ** 2)):.4f}  ({time.time()-t:.0f}s)', flush=True)
    d = pd.DataFrame({'race': tr.race, 'a': (np.exp(res['v7cfg'] + cb) - y) ** 2, 'b': (np.exp(res['v8'] + cb) - y) ** 2})[tl]
    pr = d.groupby('race')[['a', 'b']].mean() ** .5
    g = d.groupby('race')[['a', 'b']].agg(['sum', 'count']); races = g.index.values; rng = np.random.RandomState(0); w = []
    for _ in range(5000):
        s = g.loc[rng.choice(races, 8)]; w.append(np.sqrt(s[('b', 'sum')].sum() / s[('b', 'count')].sum()) < np.sqrt(s[('a', 'sum')].sum() / s[('a', 'count')].sum()))
    print(f'split {FS}: v8 better in {(pr.b < pr.a).sum()}/{len(pr)} races, P(v8 better on random 8-race test) = {np.mean(w):.3f}', flush=True)
