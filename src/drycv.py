"""Dry-pace model harness: train on dry rows, score on dry full rows of held-out races (log-ratio target)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from frame import folds
from twostage import tr, te, wet_mask, DRYF
WEAK = ['wing_level', 'suspension_stiffness', 'pu_age_races', 'cooling_level', 'round', 'total_laps']
DF = [f for f in DRYF if f not in WEAK]
DRY = ~wet_mask(tr).values
BOTH = ((tr.src_ch1 == 1) & (tr.src_ch5 == 1)).values

def dry_oof(fp, fold_seeds=(0, 1), feats=DF):
    """fp(train_df, y, [X...]) -> preds. Returns list of OOF arrays (yt space, NaN on wet rows) and mean dry RMSE."""
    oofs, rm = [], []
    for fs in fold_seeds:
        f = folds(tr, seed=fs); o = np.full(len(tr), np.nan)
        for k in range(5):
            fi = (f != k) & DRY; va = (f == k) & DRY
            (o[va],) = fp(tr[fi].reset_index(drop=True), tr.yt.values[fi], [tr[va]])
        oofs.append(o)
        m = DRY & BOTH
        rm.append(np.sqrt(np.mean((np.exp(o[m] + np.log(tr.cbase.values[m])) - tr.lap_time_sec.values[m]) ** 2)))
    return oofs, float(np.mean(rm))

def rmse_of(o, m=None):
    m = DRY & BOTH if m is None else m
    return np.sqrt(np.mean((np.exp(o[m] + np.log(tr.cbase.values[m])) - tr.lap_time_sec.values[m]) ** 2))
