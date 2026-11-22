"""Multi-layer variant: layers 5, 8, 11 (0-indexed), per modality cls, mean-patch, std-patch -> emb_layers_{split}.npz.
    python embed_layers.py train|test [N]   (N = first N cells, smoke test; reads data/, dinov3/, dinov3_vits16.pth)
"""
import csv, os, sys, time
import numpy as np, torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = 'data'
split, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
sys.path.insert(0, 'dinov3')
from dinov3.hub.backbones import dinov3_vits16


class DS(Dataset):
    def __init__(s, names): s.n = names
    def __len__(s): return len(s.n)
    def __getitem__(s, i):
        return torch.stack([torch.from_numpy(np.asarray(Image.open(f'{ROOT}/{m}/{split}/{s.n[i]}').convert('RGB')).copy()).permute(2, 0, 1) for m in ('BF', 'FL')])


if __name__ == '__main__':
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    bb = dinov3_vits16(weights='dinov3_vits16.pth').to(dev).eval()
    names = [r['Name'] for r in csv.DictReader(open(f'{ROOT}/{"train.csv" if split == "train" else "sampleSubmission.csv"}'))]
    if N: names = names[:N]
    dl = DataLoader(DS(names), batch_size=256, num_workers=6)
    L = (5, 8, 11); out = {'stats': [], 'feat': []}; t0 = time.time()
    with torch.no_grad():
        for b, x in enumerate(dl):
            x = x.to(dev).float() / 255
            bf, fl = x[:, 0], x[:, 1]
            out['stats'].append(torch.cat([bf.mean((2, 3)), fl.mean((2, 3)), bf.std((2, 3)), fl.std((2, 3))], 1).cpu().numpy())
            fs = []
            for v in (bf, fl):
                for P, c in bb.get_intermediate_layers((v - v.mean((2, 3), keepdim=True)) / (v.std((2, 3), keepdim=True) + 1e-3), n=L, return_class_token=True, norm=True):
                    fs += [c, P.mean(1), P.std(1)]
            out['feat'].append(torch.cat(fs, 1).cpu().numpy())  # order: modality(bf,fl) x layer(5,8,11) x (cls, mean, std), 384 each
            if b % 20 == 0: print(f'{split} batch {b}/{len(dl)} {time.time() - t0:.0f}s', flush=True)
    np.savez((f'emb_layers_{split}{"_smoke" if N else ""}.npz'), names=np.array(names),
             **{k: np.concatenate(v).astype('float16' if k != 'stats' else 'float32') for k, v in out.items()})
    print('saved', split, len(names), f'{time.time() - t0:.0f}s', flush=True)
