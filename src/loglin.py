import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
from sklearn.linear_model import Ridge
from prep import load, add_base, SETUP, CATS
al = add_base(load(impute=True))
for c in CATS: al[c] = al[c].astype(str)
te = al[al.is_test==1]; tr = al[al.is_test==0]
print("rain>0 frac train both:", (tr[(tr.src_ch1==1)&(tr.src_ch5==1)].rain_mm_h>0).mean().round(3), " test:", (te.rain_mm_h>0).mean().round(3))
print(pd.crosstab(pd.cut(te.rain_mm_h,[-0.1,0,0.5,1,2,4,10]), te.compound))
print("test seasons by race:", te.groupby('race').size().to_dict())

def feats(d):
    X = pd.DataFrame(index=d.index)
    for c in ['circuit','track_status','compound','driver','team']:
        X = X.join(pd.get_dummies(d[c], prefix=c).astype(float))
    X = X.join(pd.get_dummies(d.engine_mode.fillna(0).astype(int), prefix='em').astype(float))
    X['season'] = d.season - 2023
    slick = d.compound.isin(['Soft','Medium','Hard']).astype(float)
    r = d.rain_mm_h.fillna(0)
    for comp in ['Soft','Medium','Hard','Intermediate','Wet']:
        m = (d.compound == comp).astype(float)
        X[f'age_{comp}'] = m * d.tyre_age_laps; X[f'age2_{comp}'] = m * d.tyre_age_laps**2
        X[f'rain_{comp}'] = m * r; X[f'rain2_{comp}'] = m * r**2; X[f'rainsqrt_{comp}'] = m*np.sqrt(r)
    X['laps_rem'] = d.laps_remaining; X['lap_frac'] = d.lap_frac
    X['temp'] = d.track_temp_c - 40; X['temp2'] = X.temp**2
    X['quali'] = d.quali_position.fillna(10.5)
    X['gap'] = d.gap_ahead_sec.fillna(2.4); X['gap_lt1'] = (d.gap_ahead_sec < 1).astype(float)
    for c in ['ride_height_mm','brake_bias_pct','wing_level','suspension_stiffness','pu_age_races','cooling_level']:
        X[c] = d[c] - d[c].mean(); X[c+'2'] = X[c]**2
    X['wingXcirc_note'] = 0
    for circ in d.circuit.unique():
        m = (d.circuit == circ).astype(float)
        X[f'wing_{circ}'] = m * X['wing_level']; X[f'wing2_{circ}'] = m * X['wing_level']**2
        X[f'susp_{circ}'] = m * X['suspension_stiffness']; X[f'susp2_{circ}'] = m * X['suspension_stiffness']**2
        X[f'cool_{circ}'] = m * X['cooling_level']
    return X

b = al[(al.src_ch1==1)&(al.src_ch5==1)].dropna(subset=['ride_height_mm','brake_bias_pct','track_temp_c'])
Xall = feats(b); trb = b.is_test==0
X = Xall[trb]; y = np.log(b.lap_time_sec[trb]); races = b.race[trb].values
ur = np.unique(races); rng = np.random.RandomState(0); fmap = dict(zip(ur, rng.permutation(len(ur)) % 5)); fold = np.array([fmap[r] for r in races])
for alpha in [0.01, 0.1, 1, 3]:
    oof = np.zeros(len(y))
    for k in range(5):
        m = Ridge(alpha=alpha).fit(X[fold!=k], y[fold!=k]); oof[fold==k] = m.predict(X[fold==k])
    yy = np.exp(y); pp = np.exp(oof)
    tc = b.circuit[trb].isin(['Monaco','Interlagos','Mexico City','Hungaroring','Austin','Bahrain','Suzuka','Silverstone']).values
    print(f'ridge a={alpha}: both RMSE {np.sqrt(np.mean((pp-yy)**2)):.4f}  testcirc {np.sqrt(np.mean((pp[tc]-yy[tc])**2)):.4f}')
m = Ridge(alpha=0.1).fit(X, y)
co = pd.Series(m.coef_, X.columns)
print(co[[c for c in co.index if not c.startswith(('wing_','wing2_','susp','cool_','driver','team'))]].round(5).to_string())
