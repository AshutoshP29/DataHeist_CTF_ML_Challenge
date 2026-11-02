import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from prep import load, SETUP
al = load(impute=True)
tr = al[al.is_test==0].copy()
b = tr[(tr.src_ch1==1)&(tr.src_ch5==1)].copy()
def design(d, num, dummies=('circuit','driver','team','compound'), log=False):
    X = pd.get_dummies(d[list(dummies)].astype(str), drop_first=False).astype(float)
    X = pd.concat([X, d[num].astype(float)], axis=1)
    return X
num = ['season','lap','tyre_age_laps','laps_remaining','quali_position','rain_mm_h','track_temp_c'] + SETUP
g = b[(b.track_status=='GREEN') & b.compound.isin(['Soft','Medium','Hard'])].dropna(subset=num)
for log in [False, True]:
    y = np.log(g.lap_time_sec) if log else g.lap_time_sec
    X = design(g, num)
    m = LinearRegression().fit(X, y); p = m.predict(X)
    if log: p = np.exp(p)
    print('log' if log else 'lin', 'in-sample RMSE', np.sqrt(np.mean((p-g.lap_time_sec)**2)).round(3), 'n', len(g))
    if log:
        co = pd.Series(m.coef_, X.columns)
        print(co[num].round(5)); print(co.filter(like='compound').round(4)); print(co.filter(like='team').round(4)); print(co.filter(like='driver').round(4).sort_values())
# tyre deg by compound: interact
g = g.assign(ratio=g.lap_time_sec / g.groupby('circuit').lap_time_sec.transform('median'))
for c in ['Soft','Medium','Hard']:
    s = g[g.compound==c]; print(c, np.polyfit(s.tyre_age_laps, s.ratio, 2).round(6))
# SC/VSC ratio
tr['cm'] = tr.circuit.map(tr[tr.track_status=='GREEN'].groupby('circuit').lap_time_sec.median())
print(tr.assign(r=tr.lap_time_sec/tr.cm).groupby('track_status').r.describe())
# rain effect
w = tr[tr.track_status=='GREEN'].assign(r=lambda d: d.lap_time_sec/d.cm)
w['rb'] = pd.cut(w.rain_mm_h, [-0.1,0,0.5,1,2,4,10])
print(w.groupby(['rb','compound']).r.agg(['mean','count']).dropna().round(4))
