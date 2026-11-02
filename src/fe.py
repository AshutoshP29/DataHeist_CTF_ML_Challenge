import numpy as np, pandas as pd
def add_fe(al):
    al = al.copy()
    comp = al.compound.astype(str)
    al['is_slick'] = comp.isin(['Soft','Medium','Hard']).astype(int)
    al['comp_ord'] = comp.map({'Soft':0,'Medium':1,'Hard':2,'Intermediate':3,'Wet':4})
    r = al.rain_mm_h.fillna(0)
    al['rain_slick'] = r * al.is_slick
    al['rain_inter'] = r * (comp == 'Intermediate')
    al['rain_wet'] = r * (comp == 'Wet')
    al['tyre_mismatch'] = np.where(al.is_slick == 1, r, np.where(comp == 'Intermediate', np.abs(r - 1.5), np.maximum(0, 3 - r)))
    al['age_x_comp'] = al.tyre_age_laps * (al.comp_ord + 1)
    al['status_ord'] = al.track_status.astype(str).map({'GREEN':0,'VSC':1,'SC':2})
    al['em1'] = (al.engine_mode == 1).astype(float).where(al.engine_mode.notna())
    al['stint_lap'] = al.tyre_age_laps / al.total_laps
    # race-lap context shared by all cars (computed over train+test X, no target)
    g = al.groupby(['race','lap'])
    al['racelap_n'] = g.lap_id.transform('size')
    al['racelap_sc_frac'] = g.status_ord.transform('mean')
    return al
