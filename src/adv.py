import pandas as pd, numpy as np, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from frame import get_data, FEATS
from prep import SETUP
from rainfe import add_rain, RAINF2
tr, te = get_data(); tr, te = add_rain(tr, te)
b = tr[(tr.src_ch1 == 1) & (tr.src_ch5 == 1)]
b = b[b.circuit.astype(str).isin(te.circuit.astype(str).unique())]
# features excluding race identifiers (season/circuit/round/total_laps make races trivially separable)
F = [f for f in FEATS + RAINF2 if f not in ('season', 'round', 'total_laps', 'circuit', 'race_rain_total', 'rain_cum')]
X = pd.concat([b[F], te[F]]); y = np.r_[np.zeros(len(b)), np.ones(len(te))]
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1)
p = cross_val_predict(m, X, y, cv=5, method='predict_proba')[:, 1]
print('adversarial AUC', round(roc_auc_score(y, p), 3))
m.fit(X, y); imp = pd.Series(m.booster_.feature_importance('gain'), F).sort_values(ascending=False)
print((imp / imp.sum()).round(3).head(12))
F2 = [f for f in F if f not in ('ride_height_mm', 'brake_bias_pct')]
p = cross_val_predict(m, pd.concat([b[F2], te[F2]]), y, cv=5, method='predict_proba')[:, 1]
print('AUC without ride/brake', round(roc_auc_score(y, p), 3))
