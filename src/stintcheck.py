import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from frame import get_data
tr, te = get_data()
for nm, d in [('train ch5 rows', tr[tr.src_ch5 == 1]), ('test', te)]:
    d = d.copy()
    # derived stint: rank of distinct stint_start (lap - tyre_age) within race-driver
    d['ss'] = d['lap'] - d['tyre_age_laps']
    d['derived'] = d.groupby(['race', 'driver']).ss.rank(method='dense')
    ok = (d.derived == d.stint).mean()
    print(f'{nm}: stint == derived {ok:.3f}   stint dist', d.stint.value_counts().sort_index().to_dict())
    print('   compound changes at stint boundaries consistent?', d.groupby(['race', 'driver', 'stint']).compound.nunique().mean().round(3))
