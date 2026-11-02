import pandas as pd, numpy as np
D = '../train_test_splits/'
tr = {c: pd.read_csv(D + f'{c}/train.csv') for c in ['ch1', 'ch2', 'ch3', 'ch5']}
te = {c: pd.read_csv(D + f'{c}/test.csv') for c in ['ch1', 'ch2', 'ch3', 'ch5']}
tr['ch3']['lap_time_sec'] = tr['ch3'].lap_time_ms / 1000
ids = {c: set(te[c].lap_id) for c in te}
print('test id sets identical:', all(ids[c] == ids['ch5'] for c in ids))
B = pd.read_csv('../data/B_merged_ch5_ch1/train.csv')
for c in tr: print(c, 'train rows', len(tr[c]), ' in ch5:', tr[c].lap_id.isin(tr['ch5'].lap_id).sum(), ' in ch1:', tr[c].lap_id.isin(tr['ch1'].lap_id).sum(), ' in test:', tr[c].lap_id.isin(ids['ch5']).sum())
u = set().union(*[set(tr[c].lap_id) for c in tr]); print('union train ids', len(u), ' (B had', len(B), ')')
# label agreement
for a in tr:
    for b in tr:
        if a < b:
            m = tr[a][['lap_id', 'lap_time_sec']].merge(tr[b][['lap_id', 'lap_time_sec']], on='lap_id')
            if len(m): print(a, b, 'shared', len(m), 'max |dt|', np.abs(m.lap_time_sec_x - m.lap_time_sec_y).max().round(4))
# combos
s = pd.DataFrame({'lap_id': sorted(u)})
for c in tr: s[c] = s.lap_id.isin(tr[c].lap_id).astype(int)
print(s.groupby(['ch1', 'ch2', 'ch3', 'ch5']).size())
# ch3 engine_mode vs ch1 engine_mode on shared laps
for nm, A, Bt in [('train', tr['ch3'], tr['ch1']), ('test', te['ch3'], te['ch1'])]:
    m = A[['lap_id', 'engine_mode', 'wing_level']].merge(Bt[['lap_id', 'engine_mode', 'wing_level']], on='lap_id', suffixes=('_3', '_1'))
    print(nm, 'ch3 vs ch1 shared', len(m), ' engine_mode agree', (m.engine_mode_3 == m.engine_mode_1).mean().round(3), ' wing agree', np.isclose(m.wing_level_3, m.wing_level_1).mean().round(3))
    print(pd.crosstab(m.engine_mode_3, m.engine_mode_1))
