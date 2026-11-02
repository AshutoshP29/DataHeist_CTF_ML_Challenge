import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from final import tr, te
from twostage import wet_mask
o = dict(np.load('../oof/final_oof.npz')); t = dict(np.load('../oof/final_test_components.npz'))
both = ((tr.src_ch1 == 1) & (tr.src_ch5 == 1)).values
for a, b in [('lgbA', 'lgbB'), ('lgbA', 'nn'), ('lgbB', 'nn')]:
    do = np.exp(o[a] + np.log(tr.cbase)) - np.exp(o[b] + np.log(tr.cbase)); dt = np.exp(t[a] + np.log(te.cbase)) - np.exp(t[b] + np.log(te.cbase))
    print(f'{a} vs {b}: OOF rms diff {np.sqrt(np.mean(do[both] ** 2)):.3f}   test rms diff {np.sqrt(np.mean(dt ** 2)):.3f}   test max |diff| {np.abs(dt).max():.2f}')
nnt = np.exp(t['nn'] + np.log(te.cbase)); lt = np.exp(t['lgbA'] + np.log(te.cbase))
print('test per-race mean (nn - lgbA):', pd.Series(nnt - lt).groupby(te.race.values).mean().round(3).to_dict())
