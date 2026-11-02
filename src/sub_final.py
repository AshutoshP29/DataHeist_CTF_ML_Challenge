import numpy as np, warnings; warnings.filterwarnings('ignore')
from final import components, combine, tr, te
from submit import write_submission
c = components(tr, te)
np.savez('../oof/final_test_components.npz', **c)
p = combine(c, (1/3, 1/3, 1/3))
pred = np.exp(p + np.log(te.cbase.values))
v5 = np.load('../oof/v5_test.npy')
print('rms diff vs v5', np.sqrt(np.mean((pred - v5) ** 2)).round(3), ' corr', np.corrcoef(pred, v5)[0, 1].round(5))
write_submission(te.lap_id, pred, 'v6_ensemble_equal')
