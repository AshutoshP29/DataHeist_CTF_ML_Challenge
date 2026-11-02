import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from frame import folds, TEST_CIRC
from twostage import wet_mask
def load2():
    tr = pd.read_pickle('../oof/tr2.pkl'); te = pd.read_pickle('../oof/te2.pkl')
    for d in (tr, te):
        d['em_h'] = d.engine_mode_3                      # honest engine mode (ch3); equals ch1 in test
        d['fuel_rem'] = d.fuel_remaining_kg.combine_first(d.fuel_load_kg)
        d['fuel_used'] = d.fuel_used_kg_2                # ch3 copy is tampered in train
        if 'is_test' in d and d.is_test.iloc[0] == 1: d['fuel_used'] = d.fuel_used_kg_2.combine_first(d.fuel_used_kg_3)
    return tr, te
V7DRY = ['circuit', 'driver', 'team', 'track_status', 'compound', 'season', 'lap', 'tyre_age_laps', 'stint', 'laps_remaining', 'quali_position', 'gap_ahead_sec', 'track_temp_c', 'lap_frac', 'status_ord', 'stint_lap']
TELE = ['em_h', 'fuel_rem', 'fuel_used', 'full_throttle_pct', 'ers_mj', 'speed_trap_kmh', 'wing_level_3', 'drs_open_s', 'gear_shifts', 'brake_temp_c']
V7WF = ['compound', 'rain_mm_h', 'rain_lag1', 'status_ord', 'tyre_age_laps', 'laps_since_rain', 'rain_ema5', 'rain_max_so_far', 'rain_ema2']
PA = dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
WP = dict(learning_rate=0.03, num_leaves=4, min_child_samples=15, subsample=0.8, subsample_freq=1, reg_lambda=5, verbose=-1)

def two_stage(trn, X, dryf, wff, n=1200, seed=0):
    dm = ~wet_mask(trn).values
    md = lgb.LGBMRegressor(**PA, n_estimators=n, random_state=seed).fit(trn.loc[dm, dryf], trn.yt.values[dm])
    d_tr = md.predict(trn[dryf]); wm = wet_mask(trn).values
    use = wm | (np.random.RandomState(seed).rand(len(trn)) < 0.15)
    mw = lgb.LGBMRegressor(**WP, n_estimators=600, random_state=seed).fit(trn.loc[use, wff], (trn.yt.values - d_tr)[use])
    p = md.predict(X[dryf]); xw = wet_mask(X).values
    if xw.any(): p[xw] += mw.predict(X.loc[xw, wff])
    return p

def evaluate(tr, name, dryf, wff, train_mask=None, fold_seeds=(0, 7)):
    y = tr.lap_time_sec.values; cb = np.log(tr.cbase.values)
    tl = ((tr.src_ch5 == 1) & (tr.src_ch3 == 1)).values          # test-like rows: context + telemetry
    tc = tr.circuit.isin(TEST_CIRC).values; wet = wet_mask(tr).values
    out = []
    for fs in fold_seeds:
        f = folds(tr, seed=fs); o = np.zeros(len(tr))
        for k in range(5):
            fi = (f != k) if train_mask is None else ((f != k) & train_mask)
            o[f == k] = two_stage(tr[fi].reset_index(drop=True), tr[f == k], dryf, wff)
        e = np.exp(o + cb) - y
        out.append([np.sqrt(np.mean(e[m] ** 2)) for m in (tl, tl & tc, tl & ~wet, tl & wet)])
    r = np.mean(out, 0)
    print(f'{name:38s} test-like {r[0]:.4f}  testcirc {r[1]:.4f}  dry {r[2]:.4f}  wet {r[3]:.4f}   (per split: {[round(x[0], 4) for x in out]})', flush=True)

if __name__ == '__main__':
    tr, te = load2()
    old = (tr.src_ch5 == 1) | (tr.src_ch1 == 1)
    evaluate(tr, 'v7 feats, old rows only (ch1|ch5)', V7DRY, V7WF, train_mask=old.values)
    evaluate(tr, 'v7 feats, all 26k rows', V7DRY, V7WF)
    evaluate(tr, '+ honest engine mode', V7DRY + ['em_h'], V7WF + ['em_h'])
    evaluate(tr, '+ all consistent telemetry', V7DRY + TELE, V7WF + ['em_h'])
    evaluate(tr, '+ telemetry, -gap', [f for f in V7DRY if f != 'gap_ahead_sec'] + TELE, V7WF + ['em_h'])
