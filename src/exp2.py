import warnings; warnings.filterwarnings('ignore')
from prep import load, add_base, SETUP, CATS
from cv import cv, CTX
for imp in [False, True]:
    al = add_base(load(impute=imp))
    F = CATS + CTX + SETUP
    for t in ['raw','log','logbase']:
        cv(al, F, target=t, name=f'impute={imp} target={t}')
