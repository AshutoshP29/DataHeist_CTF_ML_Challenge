import numpy as np, pandas as pd
def rain_grid(al):
    rl = al.groupby(['race', 'lap']).rain_mm_h.first().reset_index()
    out = []
    for r, d in rl.groupby('race'):
        f = pd.DataFrame({'lap': np.arange(1, int(al.loc[al.race == r, 'total_laps'].max()) + 2)}); f['race'] = r
        f = f.merge(d, on=['race', 'lap'], how='left'); s = f.rain_mm_h.interpolate(limit_direction='both').fillna(0)
        f['rain_i'] = s
        for L in (1, 2, 3, 5): f[f'rain_lag{L}'] = s.shift(L).fillna(s.iloc[0])
        f['rain_lead1'] = s.shift(-1).ffill()
        for hl in (2, 5, 10): f[f'rain_ema{hl}'] = s.ewm(halflife=hl, adjust=False).mean()
        f['rain_trend'] = s - f.rain_lag3
        f['rain_cum'] = s.cumsum()
        f['rain_max_so_far'] = s.cummax()
        wet = (s > 0).values; since = np.zeros(len(s)); c = 999
        for i in range(len(s)): c = 0 if wet[i] else c + 1; since[i] = c
        f['laps_since_rain'] = np.minimum(since, 99)
        f['race_rain_total'] = s.sum()
        out.append(f.drop(columns='rain_mm_h'))
    return pd.concat(out)

RAINF = ['rain_lag1','rain_lag2','rain_lag3','rain_lag5','rain_lead1','rain_ema2','rain_ema5','rain_ema10','rain_trend','rain_cum','rain_max_so_far','laps_since_rain','race_rain_total']

def add_rain(tr, te):
    al = pd.concat([tr, te], ignore_index=True)
    g = rain_grid(al)
    def m(d):
        d = d.drop(columns=[c for c in RAINF if c in d], errors='ignore').merge(g.drop(columns='rain_i'), on=['race', 'lap'], how='left')
        comp = d.compound.astype(str)
        for k in ['rain_ema2','rain_ema5']:
            d[k+'_slick'] = d[k] * d.is_slick
        # wet-tyre mismatch vs recent wetness
        w = d.rain_ema2
        d['mis_ema'] = np.where(d.is_slick == 1, w, np.where(comp == 'Intermediate', np.abs(w - 1.5), np.abs(w - 4)))
        return d
    return m(tr), m(te)
RAINF2 = RAINF + ['rain_ema2_slick','rain_ema5_slick','mis_ema']
