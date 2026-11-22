"""Pod: frozen DINOv3 of any size -> same features as mt.py teacher (per-image std BF/FL, CLS + mean-patch per modality),
balanced L2 LR (C=0.05) on all 12 labelled patients -> test CSV in runs/big/.  (kNN smoothing: knn_col.py)
    python big_probe.py dinov3_vitb16 dinov3_vitb16.pth
"""
import os, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, 'dinov3'); import dinov3.hub.backbones as hb
FN, W = sys.argv[1], sys.argv[2]; R = 'data'; dv = 'cuda'; out = 'runs/big'; os.makedirs(out, exist_ok=True); t0 = time.time()


class DS(Dataset):
    def __init__(s, names, split): s.n, s.sp = list(names), split
    def __len__(s): return len(s.n)
    def __getitem__(s, i):
        return torch.stack([torch.from_numpy(np.asarray(Image.open(f'{R}/{m}/{s.sp}/{s.n[i]}').convert('RGB')).copy()).permute(2, 0, 1) for m in ('BF', 'FL')])


bb = getattr(hb, FN)(weights=W).to(dv).eval()


@torch.no_grad()
def feats(names, split):
    o = []
    for x in DataLoader(DS(names, split), batch_size=256, num_workers=8):
        x = x.to(dv).float() / 255; B = len(x); x = (x - x.mean((-2, -1), keepdim=True)) / (x.std((-2, -1), keepdim=True) + 1e-3)
        with torch.autocast('cuda', dtype=torch.bfloat16): t = bb.forward_features(x.flatten(0, 1))
        o.append(torch.cat([t['x_norm_clstoken'], t['x_norm_patchtokens'].mean(1)], 1).float().reshape(B, -1))
    return torch.cat(o)


tr = pd.read_csv(f'{R}/train.csv'); te = pd.read_csv(f'{R}/sampleSubmission.csv')
X = feats(tr.Name, 'train'); print(f'train feats {tuple(X.shape)} {time.time() - t0:.0f}s', flush=True)
Xt = feats(te.Name, 'test'); print(f'test feats {time.time() - t0:.0f}s', flush=True)
mu, sd = X.mean(0), X.std(0) + 1e-6; X, Xt = (X - mu) / sd, (Xt - mu) / sd
y = torch.tensor(tr.Diagnosis.values, dtype=torch.float32, device=dv); n = len(y); sw = torch.where(y == 1, n / (2 * y.sum()), n / (2 * (n - y.sum())))
w = torch.zeros(X.shape[1], device=dv, requires_grad=True); b = torch.zeros(1, device=dv, requires_grad=True)
lb = torch.optim.LBFGS([w, b], lr=1, max_iter=200, tolerance_grad=1e-5, line_search_fn='strong_wolfe')
def cl():
    lb.zero_grad(); l = .05 * (sw * F.binary_cross_entropy_with_logits(X @ w + b, y, reduction='none')).sum() + .5 * (w ** 2).sum(); l.backward(); return l
lb.step(cl)
with torch.no_grad(): p = (Xt @ w + b).sigmoid().cpu().numpy(); ptr = (X @ w + b).cpu().numpy()
print('train per-patient mean logit', pd.Series(ptr).groupby(tr.Name.str[:6].values).mean().round(2).to_dict(), flush=True)
pd.DataFrame({'Name': te.Name, 'Diagnosis': p}).to_csv(f'{out}/frozen_{FN}.csv', index=False); print('wrote', f'{out}/frozen_{FN}.csv', f'{time.time() - t0:.0f}s', flush=True)
