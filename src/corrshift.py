import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from frame import get_data
tr, te = get_data()
b = tr[(tr.src_ch1 == 1) & (tr.src_ch5 == 1)]
num = ['lap','tyre_age_laps','stint','laps_remaining','quali_position','gap_ahead_sec','rain_mm_h','track_temp_c','wing_level','ride_height_mm','brake_bias_pct','suspension_stiffness','engine_mode','pu_age_races','cooling_level','status_ord','comp_ord']
def dm(d): return d[num].astype(float) - d.groupby(d.circuit.astype(str))[num].transform('mean').astype(float)
ct, cs = dm(b).corr(), dm(te).corr()
d = (ct - cs).abs()
rows = [(i, j, ct.loc[i, j], cs.loc[i, j]) for a, i in enumerate(num) for j in num[a+1:] if d.loc[i, j] > 0.15]
for r in sorted(rows, key=lambda r: -abs(r[2] - r[3])): print(f'{r[0]:20s} {r[1]:20s} train {r[2]:+.3f}  test {r[3]:+.3f}')
