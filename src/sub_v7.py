import numpy as np, warnings; warnings.filterwarnings('ignore')
import final, drycv
# drop engine_mode everywhere (dry models via drycv.DF used inside final; weather feature list)
final.DF = [f for f in drycv.DF if f != 'engine_mode']
final.WF = [f for f in final.WF if f != 'engine_mode']
import nn; nn.CATC = [c for c in nn.CATC if c != 'engine_mode']
print('dry feats', final.DF); print('wf feats', final.WF); print('nn cats', nn.CATC)
from submit import write_submission
c = final.components(final.tr, final.te)
np.savez('../oof/v7_test_components.npz', **c)
pred = np.exp(final.combine(c, (1/3, 1/3, 1/3)) + np.log(final.te.cbase.values))
v6 = np.load('../oof/final_test_components.npz'); p6 = np.exp(final.combine(dict(v6), (1/3, 1/3, 1/3)) + np.log(final.te.cbase.values))
te = final.te
d = pred - p6
import pandas as pd
print('rms diff vs v6', np.sqrt(np.mean(d ** 2)).round(3)); print('mean diff by engine_mode', pd.Series(d).groupby(te.engine_mode.values).mean().round(3).to_dict())
write_submission(te.lap_id, pred, 'v7_ensemble_noEM')
