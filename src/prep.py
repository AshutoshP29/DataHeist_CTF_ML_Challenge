import pandas as pd, numpy as np
D = '../data/'
SETUP = ['wing_level','ride_height_mm','brake_bias_pct','suspension_stiffness','engine_mode','pu_age_races','cooling_level']
CATS = ['circuit','driver','team','track_status','compound']

def load(impute=True):
    tr = pd.read_csv(D+'B_merged_ch5_ch1/train.csv'); te = pd.read_csv(D+'B_merged_ch5_ch1/test.csv')
    tr['is_test'] = 0; te['is_test'] = 1
    al = pd.concat([tr, te], ignore_index=True)
    al['race'] = al.season.astype(str) + '_' + al.circuit
    al['stint_start'] = al['lap'] - al['tyre_age_laps']
    if impute:
        s5 = al[al.src_ch5 == 1]
        # race-level
        rnd = s5.groupby('race')['round'].first(); al['round'] = al['round'].fillna(al.race.map(rnd))
        tot = (s5['lap'] + s5['laps_remaining']).groupby(s5.race).first()
        al['total_laps'] = al.race.map(tot)
        al['laps_remaining'] = al['laps_remaining'].fillna(al.total_laps - al['lap'])
        # race-lap level (weather)
        for c in ['rain_mm_h', 'track_temp_c']:
            m = s5.dropna(subset=[c]).groupby(['race', 'lap'])[c].first()
            key = pd.MultiIndex.from_frame(al[['race', 'lap']])
            al[c] = al[c].fillna(pd.Series(m.reindex(key).values, index=al.index))
            # fall back: interpolate within race over lap
            al[c] = al.groupby('race', group_keys=False)[c].apply(lambda s: s.loc[al.loc[s.index,'lap'].sort_values().index].interpolate(limit_direction='both').reindex(s.index))
        # race-driver level
        q = s5.dropna(subset=['quali_position']).groupby(['race', 'driver'])['quali_position'].first()
        key = pd.MultiIndex.from_frame(al[['race', 'driver']])
        al['quali_position'] = al['quali_position'].fillna(pd.Series(q.reindex(key).values, index=al.index))
        # stint from same driver-race-stint_start
        st = s5.groupby(['race', 'driver', 'stint_start'])['stint'].first()
        key = pd.MultiIndex.from_frame(al[['race', 'driver', 'stint_start']])
        al['stint'] = al['stint'].fillna(pd.Series(st.reindex(key).values, index=al.index))
    else:
        al['total_laps'] = al['lap'] + al['laps_remaining']
    al['lap_frac'] = al['lap'] / al['total_laps']
    for c in CATS: al[c] = al[c].astype('category')
    return al

if __name__ == '__main__':
    al = load()
    tr = al[al.is_test == 0]
    print(tr.isna().mean().round(3)[tr.isna().mean() > 0])

def add_base(al):
    """Circuit baseline = median lap time of clean dry green laps (train only)."""
    tr = al[al.is_test == 0]
    g = tr[(tr.track_status == 'GREEN') & tr.compound.isin(['Soft', 'Medium', 'Hard']) & (tr.rain_mm_h.fillna(0) == 0)]
    al['cbase'] = al.circuit.astype(str).map(g.groupby(g.circuit.astype(str)).lap_time_sec.median()).astype(float)
    return al
