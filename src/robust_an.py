import numpy as np, pandas as pd, sys, warnings; warnings.filterwarnings('ignore')
from final import combine, tr
from twostage import wet_mask
fn = sys.argv[1]
c = dict(np.load(fn))
y = tr.lap_time_sec.values; cb = np.log(tr.cbase.values)
both = ((tr.src_ch1 == 1) & (tr.src_ch5 == 1)).values
v5 = np.exp(combine(c, (1, 0, 0)) + cb); v6 = np.exp(combine(c, (1/3, 1/3, 1/3)) + cb)
d = pd.DataFrame({'race': tr.race, 'se5': (v5 - y) ** 2, 'se6': (v6 - y) ** 2})[both]
print(fn, ' v5', np.sqrt(d.se5.mean()).round(4), ' v6', np.sqrt(d.se6.mean()).round(4))
pr = d.groupby('race')[['se5', 'se6']].mean() ** .5
print('races where v6 beats v5:', (pr.se6 < pr.se5).sum(), '/', len(pr), '  median per-race gain', (pr.se5 - pr.se6).median().round(4))
# bootstrap over races: draw 8 races (like the test), compare
rng = np.random.RandomState(0); g = d.groupby('race')[['se5', 'se6']].agg(['sum', 'count'])
races = g.index.values; wins = []; gains = []
for _ in range(5000):
    s = rng.choice(races, 8, replace=True); gg = g.loc[s]
    r5 = np.sqrt(gg[('se5', 'sum')].sum() / gg[('se5', 'count')].sum()); r6 = np.sqrt(gg[('se6', 'sum')].sum() / gg[('se6', 'count')].sum())
    wins.append(r6 < r5); gains.append(r5 - r6)
print('bootstrap 8-race test sets: P(v6 better) = %.3f   gain mean %.4f  5th pct %.4f' % (np.mean(wins), np.mean(gains), np.percentile(gains, 5)))
# individual components: is any single one dominating / unstable?
for k, w in [('lgbA', (1, 0, 0)), ('lgbB', (0, 1, 0)), ('nn', (0, 0, 1))]:
    o = np.exp(combine(c, w) + cb); print(f'  {k}: {np.sqrt(np.mean((o - y)[both] ** 2)):.4f}')
