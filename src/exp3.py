import warnings, numpy as np, lightgbm as lgb; warnings.filterwarnings('ignore')
from prep import load, add_base, SETUP, CATS
from fe import add_fe
from cv import cv, CTX, P
al = add_fe(add_base(load(impute=True)))
F0 = CATS + CTX + SETUP
FE = ['is_slick','comp_ord','rain_slick','rain_inter','rain_wet','tyre_mismatch','status_ord','stint_lap']
cv(al, F0, target='logbase', name='base logbase')
cv(al, F0 + FE, target='logbase', name='+FE')
cv(al, F0 + FE + ['cbase'], target='logbase', name='+FE +cbase')
cv(al, [f for f in F0+FE if f not in ('driver','team')], target='logbase', name='+FE -driver/team')
cv(al, [f for f in F0+FE if f not in ('gap_ahead_sec','quali_position')], target='logbase', name='+FE -gap/quali')
