import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
from frame import get_data
tr, te = get_data()
b = tr[(tr.src_ch1==1)&(tr.src_ch5==1)]
for nm, d in [('train-both', b), ('test', te)]:
    print('==', nm)
    print(pd.crosstab(d.track_status.astype(str), d.engine_mode))
    print('wing by circuit:', d.groupby(d.circuit.astype(str)).wing_level.mean().round(2).to_dict())
    print('corr cooling~temp', d[['cooling_level','track_temp_c']].corr().iloc[0,1].round(3),
          ' ride~brake', d[['ride_height_mm','brake_bias_pct']].corr().iloc[0,1].round(3),
          ' pu_age~round', d[['pu_age_races','round']].corr().iloc[0,1].round(3))
    # setup consistency within driver-race: pu_age should be constant per race-driver
    print('pu_age nunique per race-driver', d.groupby(['race','driver']).pu_age_races.nunique().mean().round(2))
