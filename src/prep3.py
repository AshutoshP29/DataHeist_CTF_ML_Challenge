"""Feature pipeline on the 4-channel merge (mirrors prep.load + add_base + add_fe + add_rain)."""
import pandas as pd, numpy as np
from prep2 import build
from prep import CATS, add_base
from fe import add_fe
from rainfe import add_rain

def load_merged():
    tr, te, _, _ = build()
    tr['is_test'] = 0; te['is_test'] = 1
    al = pd.concat([tr, te], ignore_index=True)
    al['race'] = al.season.astype(int).astype(str) + '_' + al.circuit
    # --- race-lap shared: track_status (all cars same flag on a lap)
    ts = al.dropna(subset=['track_status']).groupby(['race', 'lap']).track_status.agg(lambda s: s.mode()[0])
    k = pd.MultiIndex.from_frame(al[['race', 'lap']])
    al['track_status'] = al.track_status.fillna(pd.Series(ts.reindex(k).values, index=al.index))
    # --- driver-stint: compound / tyre_age from neighbouring laps of same driver when stint is unambiguous
    al['stint_start'] = al['lap'] - al['tyre_age_laps']
    al = al.sort_values(['race', 'driver', 'lap'])
    g = al.groupby(['race', 'driver'])
    for d in ('prev', 'next'):
        sh = 1 if d == 'prev' else -1
        known = al[['stint_start', 'compound']].where(al.compound.notna())
        al[f'ss_{d}'] = known.stint_start.groupby([al.race, al.driver]).transform(lambda s: s.ffill() if sh == 1 else s.bfill())
        al[f'cp_{d}'] = known.compound.groupby([al.race, al.driver]).transform(lambda s: s.ffill() if sh == 1 else s.bfill())
    same = al.compound.isna() & (al.ss_prev == al.ss_next) & al.ss_prev.notna()
    al.loc[same, 'compound'] = al.loc[same, 'cp_prev']; al.loc[same, 'tyre_age_laps'] = al.loc[same, 'lap'] - al.loc[same, 'ss_prev']
    # first stint before any known lap: if next known stint starts at lap ~1 and lap < next start ... leave NaN otherwise
    al.drop(columns=['ss_prev', 'ss_next', 'cp_prev', 'cp_next'], inplace=True)
    al['stint_start'] = al['lap'] - al['tyre_age_laps']
    al = al.sort_index()
    # --- race-level / race-lap (same as prep.load)
    s5 = al[al.src_ch5 == 1]
    al['round'] = al['round'].fillna(al.race.map(s5.groupby('race')['round'].first()))
    tot = (s5['lap'] + s5['laps_remaining']).groupby(s5.race).first()
    al['total_laps'] = al['total_laps'].fillna(al.race.map(tot))
    al['laps_remaining'] = al['laps_remaining'].fillna(al.total_laps - al['lap'])
    for c in ['rain_mm_h', 'track_temp_c']:
        m = al.dropna(subset=[c]).groupby(['race', 'lap'])[c].first()
        al[c] = al[c].fillna(pd.Series(m.reindex(k).values, index=al.index))
        al[c] = al.groupby('race', group_keys=False)[c].apply(lambda s: s.loc[al.loc[s.index, 'lap'].sort_values().index].interpolate(limit_direction='both').reindex(s.index))
    q = s5.dropna(subset=['quali_position']).groupby(['race', 'driver'])['quali_position'].first()
    al['quali_position'] = al['quali_position'].fillna(pd.Series(q.reindex(pd.MultiIndex.from_frame(al[['race', 'driver']])).values, index=al.index))
    st = al.dropna(subset=['stint']).groupby(['race', 'driver', 'stint_start'])['stint'].first()
    al['stint'] = al['stint'].fillna(pd.Series(st.reindex(pd.MultiIndex.from_frame(al[['race', 'driver', 'stint_start']])).values, index=al.index))
    al['lap_frac'] = al['lap'] / al['total_laps']
    al['engine_mode'] = al['engine_mode_1']  # keep name for old code paths
    for c in CATS: al[c] = al[c].astype('category')
    return al

def get_data2():
    al = add_fe(add_base(load_merged()))
    tr = al[al.is_test == 0].reset_index(drop=True); te = al[al.is_test == 1].reset_index(drop=True)
    tr['yt'] = np.log(tr.lap_time_sec) - np.log(tr.cbase)
    tr, te = add_rain(tr, te)
    return tr, te

if __name__ == '__main__':
    tr, te = get_data2()
    print(tr.shape, te.shape)
    print('train missing after imputation:', tr[['track_status', 'compound', 'tyre_age_laps', 'stint', 'rain_mm_h', 'track_temp_c']].isna().mean().round(3).to_dict())
    print('test missing:', te[['track_status', 'compound', 'tyre_age_laps', 'stint']].isna().mean().round(3).to_dict())
    tr.to_pickle('../oof/tr2.pkl'); te.to_pickle('../oof/te2.pkl')
