"""On-disk cache of patch features, one file per (dataset, model, grid, layer, sequence).

Features are stored as fp16. Freshly computed features are rounded to the cache dtype
*before* use, so a run gives identical results whether its features come from the cache or
not. Default location is outside OneDrive (%LOCALAPPDATA%/dino-vos-cache), override with the
DINO_VOS_CACHE env var.
"""
import os
import torch

DEFAULT_ROOT = os.environ.get(
    "DINO_VOS_CACHE",
    os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "dino-vos-cache"))


class SequenceFeatureCache:
    def __init__(self, root, dataset, model, layer, seq, n_frames, grid, dtype=torch.float16):
        d = os.path.join(root, dataset, model, f"{grid[0]}x{grid[1]}_L{layer}")
        os.makedirs(d, exist_ok=True)
        self.path = os.path.join(d, f"{seq}.pt")
        self.dtype = dtype
        self.feats, self.hit, self.new = None, False, None
        if os.path.exists(self.path):
            f = torch.load(self.path, map_location="cpu")
            if f.shape[0] == n_frames and tuple(f.shape[1:3]) == tuple(grid):
                self.feats, self.hit = f, True
        if self.feats is None:
            self.new = [None] * n_frames

    def __call__(self, i, compute):
        if self.feats is not None:
            return self.feats[i]
        f = compute().to(self.dtype)
        self.new[i] = f.cpu()
        return f

    def finalize(self):
        if self.new is not None and all(x is not None for x in self.new):
            tmp = self.path + ".tmp"
            torch.save(torch.stack(self.new), tmp)
            os.replace(tmp, self.path)
            self.new = None
