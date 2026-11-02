import numpy as np, pandas as pd, time, warnings; warnings.filterwarnings('ignore')
from frame import folds
from final import components, combine, tr
from twostage import wet_mask
FS = 7
t = time.time(); comps = {k: np.zeros(len(tr)) for k in ['lgbA', 'lgbB', 'nn', 'wf']}
f = folds(tr, seed=FS)
for k in range(5):
    c = components(tr[f != k].reset_index(drop=True), tr[f == k])
    for n in comps: comps[n][f == k] = c[n]
    print('fold', k, f'{time.time()-t:.0f}s', flush=True)
np.savez(f'../oof/final_oof_fs{FS}.npz', **comps)
