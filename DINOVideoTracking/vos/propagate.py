"""Label propagation, ported from facebookresearch/dino eval_video_segmentation.py.

Defaults match the original script: 7 preceding frames + frame 0 as context, top-5
neighbours, temperature 0.1, local window radius 12 patches (at the 1x grid).

Differences from the original (all numerically equivalent):
- the local-window mask is built from grid offsets per query chunk instead of a dense
  (ncontext, hw, hw) tensor, and queries are processed in chunks (top-k is per query,
  so chunking does not change the result). Needed to fit 1.5x / 2x grids in 8 GB.
- the token grid and the label-upsampling target size are explicit arguments so that
  every model / patch size / scale uses the same token grid and the same post-processing.
"""
import os
import copy
import glob
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEFAULTS = dict(n_last_frames=7, topk=5, temperature=0.1, radius=12)


def davis_palette(n=256):
    """Standard DAVIS / PASCAL VOC palette (index 0 black, 1 red-ish, ...)."""
    pal = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        pal[i] = (r, g, b)
    return pal


PALETTE = davis_palette()


def base_grid(ori_h, ori_w, short=480, patch=16):
    """Token grid the original script produces for a frame (short side 480, long side
    floored to a multiple of 64, then divided by patch 16). 480x854 -> (30, 52)."""
    if ori_h > ori_w:
        tw = short
        th = int(((tw * ori_h) / ori_w // 64) * 64)
    else:
        th = short
        tw = int(((th * ori_w) / ori_h // 64) * 64)
    return th // patch, tw // patch


def scaled_grid(grid, scale):
    return int(round(grid[0] * scale)), int(round(grid[1] * scale))


def read_frame(path, size_hw):
    """cv2 read, bilinear resize to (H, W), BGR->RGB, [0,1], (3,H,W). Same as original
    minus the colour normalisation, which the feature extractor applies."""
    img = cv2.imread(path)
    ori_h, ori_w, _ = img.shape
    img = cv2.resize(img, (size_hw[1], size_hw[0]))
    img = img.astype(np.float32) / 255.0
    img = img[:, :, ::-1]
    img = torch.from_numpy(np.transpose(img.copy(), (2, 0, 1))).float()
    return img, ori_h, ori_w


def read_seg(path, grid):
    """Returns one-hot (1, C, gh, gw) at the token grid (nearest downsample) and the
    original-resolution index array."""
    seg = Image.open(path)
    seg_ori = np.asarray(seg)
    small = np.array(seg.resize((grid[1], grid[0]), Image.NEAREST))
    n_cls = int(seg_ori.max()) + 1
    small = torch.from_numpy(small.astype(np.int64))
    onehot = F.one_hot(small, n_cls).permute(2, 0, 1).float().unsqueeze(0)
    return onehot, seg_ori


def imwrite_indexed(path, array, palette=PALETTE):
    im = Image.fromarray(array.astype(np.uint8), mode="P")
    im.putpalette(palette.ravel().tolist())
    im.save(path, format="PNG")


def norm_mask(mask):
    """Per-channel min-max normalisation (as in the original)."""
    for c in range(mask.shape[0]):
        m = mask[c]
        if m.max() > 0:
            m = m - m.min()
            mask[c] = m / m.max()
    return mask


@torch.no_grad()
def label_propagation(feat_tar, ctx_feats, ctx_segs, h, w, radius, topk, temperature,
                      chunk=1024):
    """
    feat_tar: (hw, C) L2-normalised target features.
    ctx_feats: list of (hw, C) L2-normalised context features.
    ctx_segs: list of (1, L, h, w) soft label maps.
    Returns (1, L, h, w) propagated soft labels.
    """
    hw = h * w
    n = len(ctx_feats)
    src = torch.stack(ctx_feats)                         # n x hw x C
    segs = torch.cat([s.to(feat_tar.device) for s in ctx_segs])  # n x L x h x w
    L = segs.shape[1]
    segs = segs.reshape(n, L, hw).transpose(2, 1).reshape(-1, L).T  # L x n*hw (context-major)

    idx = torch.arange(hw, device=feat_tar.device)
    rows, cols = idx // w, idx % w
    out = torch.empty(L, hw, device=feat_tar.device)
    for s in range(0, hw, chunk):
        q = feat_tar[s:s + chunk]                        # Q x C
        aff = torch.exp(torch.matmul(src, q.T) / temperature)  # n x hw(src) x Q
        if radius > 0:
            rq, cq = rows[s:s + chunk], cols[s:s + chunk]
            mask = ((rows[:, None] - rq[None, :]).abs() <= radius) & \
                   ((cols[:, None] - cq[None, :]).abs() <= radius)  # hw x Q
            aff = aff * mask.unsqueeze(0)
        aff = aff.reshape(n * hw, -1)                    # n*hw (keys) x Q
        tk_val, _ = torch.topk(aff, dim=0, k=topk)
        tk_min, _ = torch.min(tk_val, dim=0)
        aff[aff < tk_min] = 0
        aff = aff / aff.sum(dim=0, keepdim=True)
        out[:, s:s + chunk] = segs @ aff
    return out.reshape(1, L, h, w)


@torch.no_grad()
def track_sequence(extractor, frame_paths, first_seg_path, out_dir, scale=1.0,
                   n_last_frames=7, topk=5, temperature=0.1, radius=12,
                   feature_cache=None, save=True):
    """Propagate the frame-0 mask through the clip. Returns a stats dict.

    radius is given at the 1x grid and scaled with the grid (radius * scale).
    feature_cache: optional factory(grid) -> callable(i, compute_fn) -> (h, w, C) tensor,
    with .hit and .finalize().
    """
    dev = extractor.device
    ori = cv2.imread(frame_paths[0]).shape[:2]
    g0 = base_grid(*ori)
    grid = scaled_grid(g0, scale)
    if feature_cache is not None:
        feature_cache = feature_cache(grid)
    h, w = grid
    in_hw = (h * extractor.patch, w * extractor.patch)
    up_hw = (g0[0] * 16, g0[1] * 16)   # original script's processing size, e.g. 480x832
    r = int(round(radius * scale))

    def feats(i):
        def compute():
            img, _, _ = read_frame(frame_paths[i], in_hw)
            return extractor.extract(img)
        f = feature_cache(i, compute) if feature_cache is not None else compute()
        return f.to(dev).float().reshape(h * w, -1)

    if save:
        os.makedirs(out_dir, exist_ok=True)
    first_seg, seg_ori = read_seg(first_seg_path, grid)
    first_seg = first_seg.to(dev)
    ori_h, ori_w = seg_ori.shape

    cuda = torch.cuda.is_available() and str(dev).startswith("cuda")
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    f0 = feats(0)
    if save:
        imwrite_indexed(os.path.join(out_dir, "00000.png"), seg_ori)
    que = deque(maxlen=n_last_frames)
    preds = []
    for i in range(1, len(frame_paths)):
        ft = feats(i)
        used_feats = [f0] + [p[0] for p in que]
        used_segs = [first_seg] + [p[1] for p in que]
        soft = label_propagation(ft, used_feats, used_segs, h, w, r, topk, temperature)
        que.append([ft, copy.deepcopy(soft)])

        up = F.interpolate(soft, size=up_hw, mode="bilinear", align_corners=False)[0]
        up = norm_mask(up)
        lab = torch.argmax(up, dim=0).cpu().numpy().astype(np.uint8)
        lab = np.array(Image.fromarray(lab).resize((ori_w, ori_h), Image.NEAREST))
        if save:
            name = os.path.basename(frame_paths[i]).replace(".jpg", ".png")
            imwrite_indexed(os.path.join(out_dir, name), lab)
        preds.append(lab)

    if cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    cached = bool(feature_cache is not None and feature_cache.hit)
    if feature_cache is not None:
        feature_cache.finalize()
    return dict(cached=cached, n_frames=len(frame_paths), seconds=dt, sec_per_frame=dt / len(frame_paths),
                peak_mem_mb=torch.cuda.max_memory_allocated() / 2**20 if cuda else float("nan"),
                grid=f"{h}x{w}", tokens=h * w, input=f"{in_hw[0]}x{in_hw[1]}", radius=r)


def list_frames(seq_dir):
    return sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
