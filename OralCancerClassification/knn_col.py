"""Colour-stats-graph kNN smoothing (k=100, lam=0.5, 10 rounds, on logits) of a test CSV -> <csv>_knncol.csv.   python knn_col.py FILE.csv"""
import os, sys
import numpy as np, pandas as pd, torch
f = sys.argv[1]; D = '.'; dv = 'cuda'
cs = np.load('colour_stats.npz'); df = pd.read_csv(f)
assert (df.Name.values == cs['test_names']).all()
p = torch.tensor(df.Diagnosis.values, device=dv).clamp(1e-6, 1 - 1e-6); s0 = (p / (1 - p)).log()
S = cs['train_stats'].astype('float32'); Z = torch.tensor((cs['test_stats'].astype('float32') - S.mean(0)) / S.std(0), device=dv); Z = Z / Z.norm(dim=1, keepdim=True)
nn_ = []
for i in range(0, len(Z), 4096):
    sm = Z[i:i + 4096] @ Z.T; r = torch.arange(len(sm), device=dv); sm[r, r + i] = -9; nn_.append(sm.topk(100, 1).indices)
nn_ = torch.cat(nn_); s = s0.clone()
for _ in range(10): s = .5 * s0 + .5 * s[nn_].mean(1)
o = f.replace('.csv', '_knncol.csv'); pd.DataFrame({'Name': df.Name, 'Diagnosis': s.sigmoid().cpu().numpy()}).to_csv(o, index=False); print('wrote', o)
