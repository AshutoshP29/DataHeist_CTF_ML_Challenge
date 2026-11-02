import pandas as pd, numpy as np
pd.set_option('display.width', 250)
D = '../data/'
tr = pd.read_csv(D+'B_merged_ch5_ch1/train.csv'); te = pd.read_csv(D+'B_merged_ch5_ch1/test.csv')
al = pd.concat([tr, te], ignore_index=True)
setup = ['wing_level','ride_height_mm','brake_bias_pct','suspension_stiffness','engine_mode','pu_age_races','cooling_level']
ctx = ['round','stint','laps_remaining','quali_position','gap_ahead_sec','rain_mm_h','track_temp_c']
s1 = al[al.src_ch1==1]
for keys in [['season','circuit','driver'],['season','circuit','team'],['season','circuit'],['season','driver'],['season','circuit','driver','compound']]:
    g = s1.groupby(keys)
    print(keys, 'ngroups', g.ngroups, ' nunique mean per group:', {c: round(g[c].nunique().mean(),2) for c in setup})
s5 = al[al.src_ch5==1]
for keys in [['season','circuit'],['season','circuit','driver'],['season','circuit','lap'],['season','circuit','driver','stint']]:
    g = s5.groupby(keys)
    print(keys, 'ngroups', g.ngroups, {c: round(g[c].nunique().mean(),2) for c in ctx})
# round vs circuit mapping
print(s5.groupby(['season','circuit'])['round'].nunique().describe())
print(s5.groupby(['season','round'])['circuit'].nunique().value_counts())
# laps_remaining + lap = total laps per race?
s5 = s5.assign(tot=s5['lap']+s5['laps_remaining'])
print(s5.groupby(['season','circuit']).tot.nunique().value_counts())
# how many races? test vs train races overlap
print("races train:", tr.groupby(['season','circuit']).ngroups, " test:", te.groupby(['season','circuit']).ngroups)
trr = set(map(tuple, tr[['season','circuit']].drop_duplicates().values)); ter=set(map(tuple, te[['season','circuit']].drop_duplicates().values))
print("test races in train:", len(ter & trr), "of", len(ter))
# continuous setup values within group: show sample
print(s1[s1.season==2023][s1.circuit=='Monaco'][s1.driver=='Kai Moreno'][['lap','compound','tyre_age_laps']+setup].sort_values('lap').head(15))
