import pandas as pd, numpy as np
pd.set_option('display.width', 250)
D = '../data/'
At = pd.read_csv(D+'A_ch5_standalone/test.csv'); Bt = pd.read_csv(D+'B_merged_ch5_ch1/test.csv')
Ar = pd.read_csv(D+'A_ch5_standalone/train.csv'); Br = pd.read_csv(D+'B_merged_ch5_ch1/train.csv')
for nm, a, b in [('test', At, Bt), ('train', Ar, Br)]:
    m = a.merge(b, on='lap_id', suffixes=('_a', '_b'))
    print(nm, 'merged rows', len(m))
    for c in [c for c in a.columns if c not in ('lap_id',)]:
        x, y = m[c+'_a'], m[c+'_b']
        diff = ~((x == y) | (x.isna() & y.isna()))
        if x.dtype.kind == 'f' or y.dtype.kind == 'f':
            diff = ~(np.isclose(x.astype(float), y.astype(float), atol=1e-6) | (x.isna() & y.isna()))
        print(f'  {c:16s} differ: {diff.sum()}', (m.loc[diff, [c+"_a", c+"_b"]].head(3).values.tolist() if diff.sum() else ''))
print('lap_id ranges: train', Br.lap_id.min(), Br.lap_id.max(), ' test', Bt.lap_id.min(), Bt.lap_id.max())
