"""Load every model, check output shape at the 1x/2x token grid, prefix-token count,
and that --layer <depth> reproduces the default (last layer) features."""
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vos.features import FeatureExtractor, MODELS  # noqa: E402

names = sys.argv[1:] or list(MODELS)
torch.manual_seed(0)
for name in names:
    ext = FeatureExtractor(name)
    p = ext.patch
    cfg = getattr(ext.model, "config", None)
    info = ""
    if cfg is not None:
        info = (f"regs={getattr(cfg, 'num_register_tokens', 0)} "
                f"pos={'rope' if 'dinov3' in cfg.model_type else 'learned'} "
                f"type={cfg.model_type}")
    for grid in [(30, 52), (60, 104)]:
        img = torch.rand(3, grid[0] * p, grid[1] * p)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        f = ext.extract(img)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        assert f.shape[:2] == grid, f.shape
        n = f.norm(dim=-1)
        print(f"{name:11s} grid {grid} in {tuple(img.shape[1:])} -> {tuple(f.shape)} "
              f"|f|={n.min():.3f}..{n.max():.3f} {dt*1000:.0f} ms "
              f"peak {torch.cuda.max_memory_allocated()/2**20:.0f} MB")
    # last-layer consistency
    img = torch.rand(3, 30 * p, 52 * p)
    a = ext.extract(img)
    ext.layer = ext.depth
    b = ext.extract(img)
    ext.layer = None
    print(f"{name:11s} depth {ext.depth}, prefix {ext.n_prefix}, {info}, "
          f"mean/std {ext.mean.flatten().tolist()}/{[round(x, 3) for x in ext.std.flatten().tolist()]}, "
          f"layer=depth vs default max diff {(a - b).abs().max().item():.1e}")
    del ext
    torch.cuda.empty_cache()
