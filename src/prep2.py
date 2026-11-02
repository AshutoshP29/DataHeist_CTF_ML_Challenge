"""Merge ch1, ch2, ch3, ch5 on lap_id (train union + test)."""
import pandas as pd, numpy as np
D = '../train_test_splits/'
COMP = {'S': 'Soft', 'M': 'Medium', 'H': 'Hard', 'I': 'Intermediate', 'W': 'Wet'}

def read(ch, split):
    d = pd.read_csv(D + f'{ch}/{split}.csv')
    if ch == 'ch3' and 'lap_time_ms' in d: d['lap_time_sec'] = d.pop('lap_time_ms') / 1000
    return d

def build():
    out = {}
    for split in ['train', 'test']:
        c1, c2, c3, c5 = (read(c, split) for c in ['ch1', 'ch2', 'ch3', 'ch5'])
        c2 = c2.rename(columns={'tyre_age': 'tyre_age_laps', 'fuel_used_kg': 'fuel_used_kg_2'})
        c2['compound'] = c2.compound.map(COMP).fillna(c2.compound)
        c3 = c3.rename(columns={'lap_no': 'lap', 'engine_mode': 'engine_mode_3', 'wing_level': 'wing_level_3', 'fuel_used_kg': 'fuel_used_kg_3'})
        c1 = c1.rename(columns={'engine_mode': 'engine_mode_1', 'wing_level': 'wing_level_1'})
        for c, n in [(c1, 1), (c2, 2), (c3, 3), (c5, 5)]: c[f'src_ch{n}'] = 1
        out[split] = (c1, c2, c3, c5)
    # code maps from shared laps (train+test)
    allc3 = pd.concat([out['train'][2], out['test'][2]]); ref = pd.concat([out['train'][3], out['test'][3], out['train'][0], out['test'][0]])[['lap_id', 'circuit', 'driver', 'team']].drop_duplicates('lap_id')
    m = allc3.merge(ref, on='lap_id')
    tmap = m.groupby('track_code').circuit.agg(lambda s: s.mode()[0]); dmap = m.groupby('driver_code').driver.agg(lambda s: s.mode()[0])
    tdmap = m.groupby('driver').team.agg(lambda s: s.mode()[0])
    res = {}
    for split in ['train', 'test']:
        c1, c2, c3, c5 = out[split]
        c3 = c3.copy(); c3['circuit'] = c3.track_code.map(tmap); c3['driver'] = c3.driver_code.map(dmap); c3['team'] = c3.driver.map(tdmap)
        frames = [c5, c1, c2, c3]
        al = frames[0]
        for f in frames[1:]:
            shared = [c for c in f.columns if c in al.columns and c != 'lap_id']
            al = al.merge(f, on='lap_id', how='outer', suffixes=('', '__r'))
            for c in shared:
                al[c] = al[c].combine_first(al[c + '__r']); al.drop(columns=c + '__r', inplace=True)
        for n in (1, 2, 3, 5): al[f'src_ch{n}'] = al[f'src_ch{n}'].fillna(0).astype(int)
        res[split] = al
    return res['train'], res['test'], tmap, dmap

if __name__ == '__main__':
    tr, te, tmap, dmap = build()
    print(tmap.to_dict()); print(len(dmap), 'drivers mapped')
    print('train', tr.shape, ' test', te.shape)
    print('test missing:', te.isna().mean()[te.isna().mean() > 0].round(3).to_dict())
    print('train missing:', tr.isna().mean().round(2).to_dict())
    # consistency of shared columns in test across channels: ch2 vs ch5 temp/rain/stint
    c2 = read('ch2', 'test'); c5 = read('ch5', 'test')
    m = c2.merge(c5, on='lap_id', suffixes=('_2', '_5'))
    for c in ['track_temp_c', 'rain_mm_h', 'stint', 'laps_remaining', 'lap']:
        print('test ch2 vs ch5', c, 'agree', np.isclose(m[c + '_2'], m[c + '_5'], equal_nan=True).mean().round(3))
    c2 = read('ch2', 'train'); c5 = read('ch5', 'train'); m = c2.merge(c5, on='lap_id', suffixes=('_2', '_5'))
    for c in ['track_temp_c', 'rain_mm_h', 'stint', 'laps_remaining']:
        print('train ch2 vs ch5', c, 'agree', np.isclose(m[c + '_2'], m[c + '_5'], equal_nan=True).mean().round(3), 'n', len(m))
    print('compound codes ch2:', read('ch2', 'train').compound.value_counts().to_dict())
