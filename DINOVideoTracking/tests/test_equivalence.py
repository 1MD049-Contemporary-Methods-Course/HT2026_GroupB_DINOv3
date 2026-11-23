"""Check that the chunked label_propagation equals the original dense implementation."""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vos.propagate import label_propagation  # noqa: E402


def original(feat_tar, feats, segs, h, w, radius, topk):
    """Verbatim logic of dino/eval_video_segmentation.py label_propagation."""
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * radius + 1):
                for q in range(2 * radius + 1):
                    if i - radius + p < 0 or i - radius + p >= h:
                        continue
                    if j - radius + q < 0 or j - radius + q >= w:
                        continue
                    mask[i, j, i - radius + p, j - radius + q] = 1
    mask = mask.reshape(h * w, h * w).cuda()
    n = len(feats)
    fs = torch.stack([f.T for f in feats])
    ft = feat_tar.unsqueeze(0).repeat(n, 1, 1)
    aff = torch.exp(torch.bmm(ft, fs) / 0.1)
    aff *= mask.unsqueeze(0).repeat(n, 1, 1)
    aff = aff.transpose(2, 1).reshape(-1, h * w)
    tk, _ = torch.topk(aff, dim=0, k=topk)
    tkm, _ = torch.min(tk, dim=0)
    aff[aff < tkm] = 0
    aff = aff / torch.sum(aff, keepdim=True, axis=0)
    s = torch.cat(segs)
    nc, C, _, _ = s.shape
    s = s.reshape(nc, C, -1).transpose(2, 1).reshape(-1, C).T
    return torch.mm(s, aff).reshape(1, C, h, w)


if __name__ == "__main__":
    torch.manual_seed(0)
    h, w, C, L, n = 12, 20, 64, 3, 8
    norm = lambda x: F.normalize(x, dim=-1)
    ft = norm(torch.randn(h * w, C, device="cuda"))
    feats = [norm(torch.randn(h * w, C, device="cuda")) for _ in range(n)]
    segs = [torch.rand(1, L, h, w, device="cuda") for _ in range(n)]
    for radius in (0, 3, 6):
        a = original(ft, feats, segs, h, w, radius, 5) if radius else None
        b = label_propagation(ft, feats, segs, h, w, radius, 5, 0.1, chunk=37)
        if a is not None:
            print(f"radius {radius}: max abs diff {(a - b).abs().max().item():.2e}")
