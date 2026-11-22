"""CPU: which frozen DINOv3 features give the best linear probe on unseen patients? (emb_layers_train.npz from embed_layers.py)
Feature sets from layers 5/8/11 x (cls, mean-patch, std-patch) x (BF, FL); balanced L2 LR (C=0.05 unless noted).
Scored on split A, split B (all cells of held-out patients) and 40 random 2+3 splits (1000 cells/patient, same draws as cpu_tests.py).
'tstd' = standardise held-out features with held-out statistics (feature-level AdaBN: transductive batch-shift fix).
    python layer_tests.py
"""
import itertools, os, time, warnings
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); torch.set_num_threads(8); t0 = time.time()
D = '.'; R = 'data'
d = np.load(f'{D}/emb_layers_train.npz'); names = d['names']; X = d['feat'].astype('float32')
lab = pd.read_csv(f'{R}/train.csv').set_index('Name').Diagnosis.loc[names].values; pat = np.array([n[:6] for n in names])


def cols(layers, kinds):  # feat order: modality(bf, fl) x layer(5, 8, 11) x kind(cls, mean, std), 384 each
    return np.concatenate([np.arange(384) + 384 * (9 * m + 3 * (5, 8, 11).index(l) + ('cls', 'mean', 'std').index(k))
                           for m in (0, 1) for l in layers for k in kinds])


SETS = {'L11 cls+mean (current)': (cols([11], ['cls', 'mean']), .05, 0), 'L11 +std': (cols([11], ['cls', 'mean', 'std']), .05, 0),
        'L8 cls+mean': (cols([8], ['cls', 'mean']), .05, 0), 'L5 cls+mean': (cols([5], ['cls', 'mean']), .05, 0),
        'L8+L11 cls+mean': (cols([8, 11], ['cls', 'mean']), .05, 0), 'all layers+std': (cols([5, 8, 11], ['cls', 'mean', 'std']), .05, 0),
        'all layers+std C=.01': (cols([5, 8, 11], ['cls', 'mean', 'std']), .01, 0), 'L11 cls+mean tstd': (cols([11], ['cls', 'mean']), .05, 1),
        'L8+L11 tstd': (cols([8, 11], ['cls', 'mean']), .05, 1)}


def lr_fit(X, y, C):  # balanced L2 logistic regression, same objective as mt.py teacher
    dv = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'; X, y = torch.tensor(X, device=dv), torch.tensor(y, dtype=torch.float32, device=dv); n = len(y); sw = torch.where(y == 1, n / (2 * y.sum()), n / (2 * (n - y.sum())))
    w = torch.zeros(X.shape[1], device=dv, requires_grad=True); b = torch.zeros(1, device=dv, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=200, tolerance_grad=1e-5, line_search_fn='strong_wolfe')
    def cl():
        opt.zero_grad(); l = C * (sw * F.binary_cross_entropy_with_logits(X @ w + b, y, reduction='none')).sum() + .5 * (w ** 2).sum(); l.backward(); return l
    opt.step(cl); return w.detach().cpu().numpy(), b.item()


def score(tr, te):
    out = {}
    for nm, (c, C, ts) in SETS.items():
        A, Bt = X[tr][:, c], X[te][:, c]; mu, sd = A.mean(0), A.std(0) + 1e-6
        w, b = lr_fit((A - mu) / sd, lab[tr], C)
        Z = (Bt - Bt.mean(0)) / (Bt.std(0) + 1e-6) if ts else (Bt - mu) / sd
        s = Z @ w + b; y = lab[te]
        out[nm] = (roc_auc_score(y, s), roc_auc_score(y, pd.Series(s).groupby(pat[te]).transform('mean').values))
    return out


rows = []
for nm, ho in (('split A', 'pat_03,pat_16,pat_09,pat_10,pat_15'), ('split B', 'pat_16,pat_17,pat_09,pat_11,pat_14')):
    te = np.isin(pat, ho.split(',')); r = score(np.where(~te)[0], np.where(te)[0])
    print(f'\n{nm} (all cells)  pooled / patient-mean AUC', flush=True)
    for k, (p, q) in r.items(): print(f'  {k:26s} {p:.4f} / {q:.3f}', flush=True)

rng = np.random.default_rng(0)  # same draws as cpu_tests.py / explore_mil.py
sub = np.concatenate([rng.choice(np.where(pat == q)[0], min(1000, (pat == q).sum()), replace=False) for q in sorted(set(pat))])
pos, neg = sorted(set(pat[sub][lab[sub] == 1])), sorted(set(pat[sub][lab[sub] == 0]))
splits = [(a, b) for a in itertools.combinations(pos, 2) for b in itertools.combinations(neg, 3)]
splits = [splits[i] for i in rng.choice(len(splits), 40, replace=False)]
for i, (cp, hp) in enumerate(splits):
    te = np.isin(pat[sub], cp + hp); r = score(sub[~te], sub[te])
    rows.append({**{k: v[0] for k, v in r.items()}, **{k + ' |pm': v[1] for k, v in r.items()}})
    if i % 10 == 9: print(f'split {i + 1}/40 {time.time() - t0:.0f}s', flush=True)
res = pd.DataFrame(rows); res.to_csv(f'{D}/out/layer_tests.csv', index=False)
base = res['L11 cls+mean (current)']; hard = res['L11 cls+mean (current) |pm'] < 1
print(f'\n40 splits: pooled AUC mean | mean on {hard.sum()} hard splits (baseline patient-mean AUC < 1) | wins vs current', flush=True)
for k in SETS: print(f'  {k:26s} {res[k].mean():.4f} | {res[k][hard].mean():.4f} | {(res[k] > base).mean():.2f}', flush=True)
print('LAYER_TESTS_DONE', f'{time.time() - t0:.0f}s', flush=True)
