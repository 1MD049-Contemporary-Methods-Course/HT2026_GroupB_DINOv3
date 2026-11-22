"""Colour-robust frozen DINOv3 embeddings (run on the pod, GPU). Several input variants per cell, one pass each:
bf_raw (as before), bf_norm (per-image per-channel standardised), bf_gray (per-image standardised luminance),
fl_str (as before: 99.5th-percentile stretch), fl_norm (per-image per-channel standardised).

    python embed_robust.py train|test      # -> emb2_{split}.npz   (env: ROOT, DINOV3, WEIGHTS)
"""
import csv, os, sys, time
import numpy as np, torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = os.environ.get('ROOT', 'data')
DINOV3 = os.environ.get('DINOV3', 'dinov3')
WEIGHTS = os.environ.get('WEIGHTS', 'dinov3_vits16.pth')
split = sys.argv[1]
V = ('bf_raw', 'bf_norm', 'bf_gray', 'fl_str', 'fl_norm')


class DS(Dataset):
    def __init__(s, names): s.n = names
    def __len__(s): return len(s.n)
    def __getitem__(s, i):
        return torch.stack([torch.from_numpy(np.asarray(Image.open(f'{ROOT}/{m}/{split}/{s.n[i]}').convert('RGB')).copy()).permute(2, 0, 1) for m in ('BF', 'FL')])


def std(x):  # per image, per channel: zero mean, unit std
    return (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-3)


def main():
    sys.path.insert(0, DINOV3)
    from dinov3.hub.backbones import dinov3_vits16
    dev = 'cuda'
    bb = dinov3_vits16(weights=WEIGHTS).to(dev).eval()
    mu = torch.tensor([.485, .456, .406], device=dev).view(1, 3, 1, 1); sd = torch.tensor([.229, .224, .225], device=dev).view(1, 3, 1, 1)
    src = 'train.csv' if split == 'train' else 'sampleSubmission.csv'
    names = [r['Name'] for r in csv.DictReader(open(f'{ROOT}/{src}'))]
    dl = DataLoader(DS(names), batch_size=256, num_workers=min(16, os.cpu_count()), pin_memory=True)
    out = {f'{v}_{k}': [] for v in V for k in ('cls', 'mp')}; out['stats'] = []
    t0 = time.time()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for b, x in enumerate(dl):
            x = x.to(dev).float() / 255
            bf, fl = x[:, 0], x[:, 1]
            out['stats'].append(torch.cat([bf.mean((2, 3)), fl.mean((2, 3)), bf.std((2, 3)), fl.std((2, 3))], 1).cpu().numpy())
            hi = torch.quantile(fl.flatten(1), .995, dim=1).clamp(min=.03).view(-1, 1, 1, 1)
            g = bf.mean(1, keepdim=True).expand(-1, 3, -1, -1)
            inp = {'bf_raw': (bf - mu) / sd, 'bf_norm': std(bf), 'bf_gray': std(g), 'fl_str': ((fl / hi).clamp(0, 1) - mu) / sd, 'fl_norm': std(fl)}
            for v in V:
                t = bb.forward_features(inp[v])
                out[f'{v}_cls'].append(t['x_norm_clstoken'].float().cpu().numpy()); out[f'{v}_mp'].append(t['x_norm_patchtokens'].float().mean(1).cpu().numpy())
            if b % 20 == 0: print(f'batch {b}/{len(dl)}  {time.time() - t0:.0f}s', flush=True)
    np.savez(f'emb2_{split}.npz', names=np.array(names), **{k: np.concatenate(v).astype('float16') if k != 'stats' else np.concatenate(v) for k, v in out.items()})
    print('saved', split, time.time() - t0, 's', flush=True)


if __name__ == '__main__':
    main()
