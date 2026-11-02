"""Strict submission writer: header `lap_id,lap_time_sec`, exactly 9,544 rows, test order, no NaN."""
import numpy as np, pandas as pd, os
D = '../'
TEST = pd.read_csv(D + 'data/B_merged_ch5_ch1/test.csv', usecols=['lap_id'])

def write_submission(lap_ids, preds, name):
    sub = pd.DataFrame({'lap_id': np.asarray(lap_ids).astype(np.int64), 'lap_time_sec': np.asarray(preds, float)})
    sub = TEST.merge(sub, on='lap_id', how='left')  # force official test order / id set
    assert len(sub) == 9544, len(sub)
    assert sub.lap_id.is_unique and set(sub.lap_id) == set(TEST.lap_id)
    assert sub.lap_time_sec.notna().all() and np.isfinite(sub.lap_time_sec).all()
    assert sub.lap_time_sec.between(60, 200).all(), sub.lap_time_sec.describe()
    os.makedirs(D + 'submissions', exist_ok=True)
    path = D + f'submissions/{name}.csv'
    sub.to_csv(path, index=False, float_format='%.4f', lineterminator='\n')
    chk = pd.read_csv(path)
    assert list(chk.columns) == ['lap_id', 'lap_time_sec'] and len(chk) == 9544
    print('wrote', path, sub.lap_time_sec.describe().round(2).to_dict())
    return path
