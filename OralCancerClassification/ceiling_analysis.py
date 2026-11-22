"""Why do we plateau at ~0.76? Analysis on split B held-out patients (pat_16, pat_17 cancer; pat_09, pat_11, pat_14 healthy)
with the frozen probe and Mean Teacher r4 EMA ep3 predictions, plus colour-stats structure of the real test set.
    python ceiling_analysis.py      (needs colour_stats.npz, ana/{frozen,mt}.npy, data/train.csv)
"""
import numpy as np, pandas as pd, torch
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
tr = pd.read_csv('data/train.csv'); tr['pat'] = tr.Name.str[:6]
cs = np.load('colour_stats.npz'); Str, Ste = cs['train_stats'].astype('float32'), cs['test_stats'].astype('float32')
HO = ['pat_16', 'pat_17', 'pat_09', 'pat_11', 'pat_14']; m = tr.pat.isin(HO).values
y, pat = tr.Diagnosis.values[m], np.array(tr.pat.tolist())[m]; Z = (Str[m] - Str[~m].mean(0)) / Str[~m].std(0)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
dv = 'cuda' if torch.cuda.is_available() else 'cpu'


def knn(Z, k=100):
    Zt = torch.tensor(Z / np.linalg.norm(Z, axis=1, keepdims=True), device=dv); out = []
    for i in range(0, len(Zt), 4096):
        sm = Zt[i:i + 4096] @ Zt.T; r = torch.arange(len(sm), device=dv); sm[r, r + i] = -9; out.append(sm.topk(k, 1).indices)
    return torch.cat(out).cpu().numpy()


def smooth(s0, nn_, lam, rounds=10):
    s = s0.copy()
    for _ in range(rounds): s = (1 - lam) * s0 + lam * s[nn_].mean(1)
    return s


nn_ = knn(Z)
print(f'held-out cells {len(y)}; per patient', pd.Series(pat).value_counts().to_dict())
print(f'colour-graph neighbours from the SAME patient: {(pat[nn_] == pat[:, None]).mean():.3f} (chance {(pd.Series(pat).value_counts(normalize=True) ** 2).sum():.3f})\n')
for nm in ('frozen', 'mt'):
    s = lg(np.load(f'ana/{nm}.npy')); pm = pd.Series(s).groupby(pat).transform('mean').values
    print(f'=== {nm}: pooled AUC {roc_auc_score(y, s):.4f} | if every cell had its patient mean (ceiling) {roc_auc_score(y, pm):.4f}')
    cp, hp = [q for q in HO if y[pat == q][0] == 1], [q for q in HO if y[pat == q][0] == 0]
    M = pd.DataFrame([[roc_auc_score(np.r_[np.ones((pat == c).sum()), np.zeros((pat == h).sum())], np.r_[s[pat == c], s[pat == h]]) for h in hp] for c in cp], cp, hp)
    print('pairwise cell AUC (cancer rows x healthy cols):\n' + M.round(3).to_string())
    q95 = np.quantile(s[y == 0], .95)
    print('share of cells above healthy 95th percentile:', {q: round((s[pat == q] > q95).mean(), 3) for q in HO})
    pi = (s[y == 1] > q95).mean() - .05
    print(f'-> estimated share of clearly abnormal cells in cancer patients ~{pi / .95:.2f}; implied cell-AUC ceiling ~0.5+pi/2 = {.5 + pi / .95 / 2:.3f}')
    print('kNN smoothing strength -> pooled AUC:', {f'lam{l}': round(roc_auc_score(y, smooth(s, nn_, l)), 4) for l in (0, .5, .8, .9, .95)})
    for K in (5, 10, 20):
        lab = KMeans(K, n_init=4, random_state=0).fit_predict(Z); pur = pd.crosstab(lab, pat).max(1).sum() / len(lab)
        cm = pd.Series(s).groupby(lab).transform('mean').values
        print(f'  k-means K={K} on colour stats: purity {pur:.3f} | cluster-mean AUC {roc_auc_score(y, cm):.4f} | 0.5 cell + 0.5 cluster {roc_auc_score(y, .5 * s + .5 * cm):.4f}')
    print()

Zt = (Ste - Str.mean(0)) / Str.std(0); Zh = (Str[m] - Str.mean(0)) / Str.std(0)
for nm, A, K in (('held-out 5 patients', Zh, 5), ('real test (~7 patients)', Zt, 7)):
    km = KMeans(K, n_init=4, random_state=0).fit(A)
    from sklearn.metrics import silhouette_score
    idx = np.random.default_rng(0).choice(len(A), 5000, replace=False)
    print(f'{nm}: K={K} cluster sizes {sorted(np.bincount(km.labels_).tolist(), reverse=True)} | silhouette {silhouette_score(A[idx], km.labels_[idx]):.3f}')
