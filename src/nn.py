import numpy as np, pandas as pd, torch, torch.nn as nn, warnings, sys, time; warnings.filterwarnings('ignore')
from drycv import dry_oof, rmse_of, DRY, BOTH
from twostage import tr, te
DEV = 'cuda'
CATC = ['circuit', 'driver', 'team', 'track_status', 'compound', 'season', 'engine_mode', 'stint']
NUMC = ['lap', 'tyre_age_laps', 'laps_remaining', 'lap_frac', 'track_temp_c', 'quali_position', 'gap_ahead_sec', 'stint_lap']
# vocab over train+test
AL = pd.concat([tr, te])
VOC = {c: {v: i + 1 for i, v in enumerate(sorted(set(str(v) for v in AL[c].astype(object).unique())))} for c in CATC}
STATS = {c: (AL[c].mean(), AL[c].std()) for c in NUMC}

def enc(d):
    xc = np.stack([d[c].astype(object).map(lambda v: VOC[c].get(str(v), 0)).fillna(0).astype(int).values for c in CATC], 1)
    xn = []
    for c in NUMC:
        v = d[c].values.astype(float); na = np.isnan(v)
        xn.append(np.where(na, 0, (v - STATS[c][0]) / STATS[c][1])); xn.append(na.astype(float))
    # circuit-specific tyre interactions handled by the net; add age^2
    a = (d.tyre_age_laps.values - 15) / 12; xn.append(a ** 2)
    return torch.tensor(xc, dtype=torch.long), torch.tensor(np.stack(xn, 1), dtype=torch.float32)

class Net(nn.Module):
    def __init__(self, nnum, hid=256, drop=0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(len(VOC[c]) + 1, min(16, len(VOC[c]) + 1)) for c in CATC])
        de = sum(e.embedding_dim for e in self.embs)
        self.mlp = nn.Sequential(nn.Linear(de + nnum, hid), nn.SiLU(), nn.Dropout(drop), nn.Linear(hid, hid), nn.SiLU(), nn.Dropout(drop),
                                 nn.Linear(hid, hid // 2), nn.SiLU(), nn.Linear(hid // 2, 1))
    def forward(self, xc, xn):
        e = torch.cat([emb(xc[:, i]) for i, emb in enumerate(self.embs)] + [xn], 1)
        return self.mlp(e).squeeze(1)

def nn_fp(epochs=60, hid=256, drop=0.1, lr=2e-3, wd=1e-4, bs=256, seeds=(0,)):
    def fp(X, y, Xs):
        xc, xn = enc(X); yy = torch.tensor(y * 100, dtype=torch.float32)  # scale: yt ~ log ratio
        xc, xn, yy = xc.to(DEV), xn.to(DEV), yy.to(DEV)
        outs = [np.zeros(len(Z)) for Z in Xs]
        for s in seeds:
            torch.manual_seed(s); net = Net(xn.shape[1], hid, drop).to(DEV)
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
            steps = epochs * ((len(yy) + bs - 1) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
            net.train()
            for ep in range(epochs):
                perm = torch.randperm(len(yy), device=DEV)
                for i in range(0, len(yy), bs):
                    idx = perm[i:i + bs]
                    loss = ((net(xc[idx], xn[idx]) - yy[idx]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            net.eval()
            with torch.no_grad():
                for j, Z in enumerate(Xs):
                    zc, zn = enc(Z); outs[j] += net(zc.to(DEV), zn.to(DEV)).cpu().numpy() / 100 / len(seeds)
        return outs
    return fp

if __name__ == '__main__':
    for kw in [dict(epochs=40), dict(epochs=80), dict(epochs=60, hid=512, drop=0.2), dict(epochs=60, wd=1e-2)]:
        t = time.time(); oofs, r = dry_oof(nn_fp(**kw))
        print(kw, 'dry RMSE', round(r, 4), f'({time.time()-t:.0f}s)', flush=True)
