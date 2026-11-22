"""Mean Teacher on DINOv3 ViT-S/16 + LoRA, starting from the frozen linear-probe teacher (public 0.7163).

Step 0: frozen features (CLS + mean patch of per-image standardised BF and FL) -> balanced L2 logistic head on
all 12 patients. That model is the teacher; its test predictions are written (and submitted with --submit).
Student = same net with LoRA on the last N blocks (LoRA B starts at 0, so student == teacher at step 0). It trains on
  labelled cells (true patient label; patient- and class-balanced sampling) + test cells (teacher soft labels),
both under strong structural augmentation. Teacher = EMA of the student (--ema 1 keeps the frozen teacher fixed).
Every epoch: test predictions of student, EMA teacher and student+teacher rank average, diagnostics, optional submit.

    python3 mt.py --epochs 0 --size 128 --submit       # teacher only (should reproduce ~0.716)
    python3 mt.py --epochs 3 --ema 0.999 --submit      # Mean Teacher
    python3 mt.py --smoke 640 --epochs 1               # quick code check, no submit
"""
import argparse, copy, math, os, subprocess, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from peft import LoraConfig, get_peft_model

p = argparse.ArgumentParser()
p.add_argument('--root', default='data')
p.add_argument('--dinov3', default='dinov3')
p.add_argument('--weights', default='dinov3_vits16.pth')
p.add_argument('--size', type=int, default=128)     # 128 = native (8x8 patches); 224 upsamples (14x14)
p.add_argument('--epochs', type=int, default=3)     # passes over the test cells
p.add_argument('--bs', type=int, default=64)        # per half: bs labelled + bs unlabelled cells per step
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--rank', type=int, default=4)
p.add_argument('--blocks', type=int, default=6)     # LoRA on the last N of the 12 blocks
p.add_argument('--ema', type=float, default=0.999)  # teacher = EMA of student; 1 = keep the frozen teacher fixed
p.add_argument('--wu', type=float, default=1.0)     # weight of the pseudo-label loss
p.add_argument('--mdrop', type=float, default=0.15)
p.add_argument('--droppath', type=float, default=0.1)
p.add_argument('--C', type=float, default=0.05)     # logistic-regression strength (same as analyse3.py)
p.add_argument('--out', default='runs/mt')
p.add_argument('--submit', action='store_true')
p.add_argument('--smoke', type=int, default=0)      # >0: only this many cells per split, never submits
p.add_argument('--mask', type=float, default=0.)    # masked consistency (MIC): extra student view of test cells with this share of patch tokens masked
p.add_argument('--wm', type=float, default=1.)      # weight of the masked-consistency loss
p.add_argument('--seed', type=int, default=0)
p.add_argument('--load', default='')                # comma-separated LoRA+head checkpoints: predict test with each (+TTA), save .npy next to it, exit
p.add_argument('--tta', type=int, default=1)        # test-time views: 1 = none, 4 = rot90s, 8 = rot90s x flip
p.add_argument('--holdout', default='')            # e.g. pat_16,pat_17,pat_09,pat_11,pat_14: these labelled patients play the unlabelled test set (diagnostic, no submit)
p.add_argument('--lora_blocks', default='')        # e.g. 0-3: surgical fine-tuning of early blocks (overrides --blocks = last N)
p.add_argument('--attnpool', action='store_true')  # learned attention pooling over patch tokens (zero-init = mean pooling, so starts identical)
p.add_argument('--wise', type=float, default=1.)   # with --load: WiSE-FT, scale LoRA delta and interpolate head toward the linear probe by this alpha
p.add_argument('--ptune', default='lora', choices=['lora', 'norm', 'bitfit'])  # what to tune in the chosen blocks: LoRA, LayerNorm affine only, or biases only (BitFit)
p.add_argument('--layer', type=int, default=11)  # read features after this block (0-indexed, 11 = last); frozen probe: layer 5 >> 11 on splits A/B
p.add_argument('--init', default='')  # round-2 self-training: start student AND EMA teacher from this checkpoint, WiSE-interpolated by --wise
p.add_argument('--ssl', action='store_true')  # label-free adaptation: iBOT-style self-distillation (EMA teacher) on test+train images, then refit the probe
p.add_argument('--fuse', action='store_true')  # token-level fusion: BF+FL patch tokens in ONE sequence (shared positions + learned modality embedding)
p.add_argument('--mlp', type=int, default=0)       # >0: extra residual MLP head of this width on top of the linear head (last layer zero-init -> starts identical)
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
os.makedirs(a.out, exist_ok=True)
dev, COMP = 'cuda', 'multimodal-cancer-classification-challenge-2026'
log = lambda *s: print(*s, flush=True)
sys.path.insert(0, a.dinov3)
from dinov3.hub.backbones import dinov3_vits16  # direct import: hubconf.py drags in extra deps


class DS(Dataset):
    def __init__(s, names, split): s.n, s.split = list(names), split
    def __len__(s): return len(s.n)
    def __getitem__(s, i):
        return i, torch.stack([torch.from_numpy(np.asarray(Image.open(f'{a.root}/{m}/{s.split}/{s.n[i]}').convert('RGB')).copy()).permute(2, 0, 1) for m in ('BF', 'FL')])


def loader(names, split, bs=256, **kw):
    return DataLoader(DS(names, split), batch_size=bs, num_workers=5, pin_memory=True, **kw)


def strong(x):  # x: B,2,3,H,W in [0,1]; one geometry per cell, shared by BF and FL
    B, M, C, H, W = x.shape; d = x.device
    x = x.clamp(1e-4, 1) ** torch.empty(B, M, 1, 1, 1, device=d).uniform_(.7, 1.5)  # gamma: survives standardisation
    s = torch.empty(B, device=d).uniform_(.6, 1).sqrt()                               # crop area 60-100%
    t = torch.rand(B, device=d) * 2 * math.pi                                         # any rotation angle
    f = torch.where(torch.rand(B, device=d) < .5, -1., 1.)                            # flip
    tx, ty = [(torch.rand(B, device=d) * 2 - 1) * (1 - s) for _ in range(2)]
    th = torch.stack([torch.stack([s * t.cos() * f, -s * t.sin(), tx], 1), torch.stack([s * t.sin() * f, s * t.cos(), ty], 1)], 1)
    x = x.reshape(B, M * C, H, W)
    x = F.grid_sample(x, F.affine_grid(th, list(x.shape), align_corners=False), padding_mode='reflection', align_corners=False)
    if np.random.rand() < .5:  # ponytail: one blur sigma per batch on half the batches; per-cell sigma if blur turns out to matter
        g = torch.arange(-3, 4, device=d).float(); g = (-g ** 2 / (2 * np.random.uniform(.1, 1.5) ** 2)).exp(); g = g / g.sum()
        x = F.conv2d(F.pad(x, (3, 3, 0, 0), mode='reflect'), g.view(1, 1, 1, 7).repeat(M * C, 1, 1, 1), groups=M * C)
        x = F.conv2d(F.pad(x, (0, 0, 3, 3), mode='reflect'), g.view(1, 1, 7, 1).repeat(M * C, 1, 1, 1), groups=M * C)
    return x.reshape(B, M, C, H, W)


def post(x):  # after standardisation: additive noise + erase one 16-32 px square per cell (same square in BF and FL)
    B, M, C, H, W = x.shape; d = x.device
    x = x + .1 * torch.randn_like(x)
    r = torch.randint(8, 17, (B, 1, 1), device=d); cy, cx = [torch.randint(0, H, (B, 1, 1), device=d) for _ in range(2)]
    m = ((torch.arange(H, device=d).view(1, H, 1) - cy).abs() < r) & ((torch.arange(W, device=d).view(1, 1, W) - cx).abs() < r)
    return x.masked_fill(m.view(B, 1, 1, H, W), 0)


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        bb = dinov3_vits16(weights=a.weights)
        for blk in bb.blocks: blk.sample_drop_ratio = a.droppath  # stochastic depth (train mode only)
        lo, hi = map(int, a.lora_blocks.split('-')) if a.lora_blocks else (12 - a.blocks, 11)
        k = '|'.join(str(i) for i in range(lo, hi + 1))
        s.bb = get_peft_model(bb, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=.1, target_modules=rf'.*blocks\.({k})\.attn\.(qkv|proj)'))
        if a.ptune != 'lora':  # LoRA_B stays zero (frozen) -> backbone tuned only through norms / biases of blocks lo..hi
            for n_, q in s.bb.named_parameters():
                q.requires_grad_(any(f'blocks.{i}.' in n_ for i in range(lo, hi + 1)) and 'lora' not in n_
                                 and (('norm' in n_) if a.ptune == 'norm' else n_.endswith('bias')))
        s.head = nn.Linear(1536, 1)
        s.att = nn.Linear(384, 1) if a.attnpool else None
        if a.attnpool: nn.init.zeros_(s.att.weight); nn.init.zeros_(s.att.bias)
        s.extra = nn.Sequential(nn.Linear(1536, a.mlp), nn.GELU(), nn.Dropout(.2), nn.Linear(a.mlp, 1)) if a.mlp else None
        if a.mlp: nn.init.zeros_(s.extra[-1].weight); nn.init.zeros_(s.extra[-1].bias)
        s.mod = nn.Parameter(torch.zeros(2, 1, 384)) if a.fuse else None  # zero-init modality embedding for BF / FL patch tokens
        s.register_buffer('fm', torch.zeros(1536)); s.register_buffer('fs', torch.ones(1536))  # feature standardisation of the LR head

    def feats(s, x, aug=False, mask=0.):  # uint8 B,2,3,H,W -> B,1536 = [BF cls, BF mean-patch, FL cls, FL mean-patch] (as embed_robust norm BF+FL)
        B = x.shape[0]; m = None
        if mask:  # same random patch-token mask for BF and FL of a cell; DINOv3 replaces them with its pretrained (iBOT) mask token
            N = (a.size // 16) ** 2
            m = (torch.rand(B, N, device=x.device).argsort(1) < int(mask * N)).repeat_interleave(2, 0)
        x = x.float() / 255
        if aug: x = strong(x)
        x = (x - x.mean((-2, -1), keepdim=True)) / (x.std((-2, -1), keepdim=True) + 1e-3)
        if aug: x = post(x)
        x = x.flatten(0, 1)
        if a.size != x.shape[-1]: x = F.interpolate(x, size=a.size, mode='bilinear', align_corners=False)
        if a.fuse or a.layer != 11: return s.tokens(x, m, B)
        t = s.bb.forward_features(x, masks=m)
        P = t['x_norm_patchtokens']
        pool = (P * s.att(P).float().softmax(1).to(P.dtype)).sum(1) if s.att is not None else P.mean(1)
        return torch.cat([t['x_norm_clstoken'], pool], 1).reshape(B, 1536)

    def tokens(s, x, m, B):  # manual forward up to block a.layer; --fuse: ONE sequence [cls, registers, BF patches, FL patches] with RoPE positions repeated
        vt = s.bb.base_model.model; pre = vt.n_storage_tokens + 1
        tok, (H, W) = vt.prepare_tokens_with_masks(x, m); n = H * W
        if a.fuse:
            tok = tok.view(B, 2, tok.shape[1], -1); x = torch.cat([tok[:, 0, :pre], tok[:, 0, pre:] + s.mod[0], tok[:, 1, pre:] + s.mod[1]], 1)
        else: x = tok
        for blk in vt.blocks[:a.layer + 1]:
            sin, cos = vt.rope_embed(H=H, W=W)
            if a.fuse: sin, cos = torch.cat([sin, sin], -2), torch.cat([cos, cos], -2)
            x = blk([x], [(sin, cos)])[0]
        if vt.untie_cls_and_patch_norms: c, P = vt.cls_norm(x[:, :pre])[:, 0], vt.norm(x[:, pre:])
        else: xn = vt.norm(x); c, P = xn[:, 0], xn[:, pre:]
        pl = (lambda Q: (Q * s.att(Q).float().softmax(1).to(Q.dtype)).sum(1)) if s.att is not None else (lambda Q: Q.mean(1))
        if a.fuse: return torch.cat([c, pl(P[:, :n]), c, pl(P[:, n:])], 1)  # same 1536 layout as late fusion (shared cls twice)
        return torch.cat([c, pl(P)], 1).reshape(B, 1536)

    def toks(s, x, mask=0.):  # x float B,2,3,H,W in [0,1], augmented outside -> cls (2B,384), patch tokens (2B,N,384), token mask (2B,N) or None
        B = x.shape[0]; N = (a.size // 16) ** 2; m = None
        if mask: m = (torch.rand(B, N, device=x.device).argsort(1) < int(mask * N)).repeat_interleave(2, 0)
        x = (x - x.mean((-2, -1), keepdim=True)) / (x.std((-2, -1), keepdim=True) + 1e-3)
        t = s.bb.forward_features(x.flatten(0, 1), masks=m)
        return t['x_norm_clstoken'], t['x_norm_patchtokens'], m

    def forward(s, x, aug=False, mask=0.):
        z = (s.feats(x, aug, mask).float() - s.fm) / s.fs
        if aug and a.mdrop > 0:  # modality dropout: replace BF or FL features by their mean for some cells
            r = torch.rand(len(z), 1, device=z.device); h = a.mdrop / 2
            z = torch.cat([z[:, :768] * (r >= h), z[:, 768:] * ~((r >= h) & (r < a.mdrop))], 1)
        return (s.head(z) + (s.extra(z) if s.extra is not None else 0)).squeeze(1)


@torch.no_grad()
def run(model, names, split, fn):
    model.eval(); out = []
    for _, x in loader(names, split):
        with torch.autocast('cuda', dtype=torch.bfloat16): out.append(fn(model, x.to(dev, non_blocking=True)).float())
    return torch.cat(out)


tr = pd.read_csv(f'{a.root}/train.csv'); tr['pat'] = tr.Name.str[:6]
te = pd.read_csv(f'{a.root}/sampleSubmission.csv')
if a.smoke: tr, te = tr.groupby('pat').head(a.smoke // 12).reset_index(drop=True), te.head(a.smoke); a.submit = False
TS = 'test'
if a.holdout:  # simulated test: held-out labelled patients become the unlabelled set; their labels are used only for evaluation
    ho = tr.pat.isin(a.holdout.split(','))
    tr, te, TS, a.submit = tr[~ho].reset_index(drop=True), tr[ho].reset_index(drop=True), 'train', False
dv = tr.groupby('pat').head(200)  # fixed labelled subset for the train-AUC / per-patient diagnostic
views = lambda x: [torch.rot90(x.flip(-1) if f else x, k, (-2, -1)) for k in range(4) for f in (0, 1)][:a.tta]  # same view for BF and FL
predict = lambda m: run(m, te.Name, TS, lambda m, x: torch.stack([m(v).float().sigmoid() for v in views(x)]).mean(0)).cpu().numpy()
rank = lambda v: v.argsort().argsort() / (len(v) - 1)


def diag(m):
    pr = run(m, dv.Name, 'train', lambda m, x: m(x)).cpu().numpy()
    return roc_auc_score(dv.Diagnosis, pr), pd.Series(pr).groupby(dv.pat.values).mean().round(2).to_dict()


def save_sub(pred, name, msg):
    path = f'{a.out}/{name}.csv'
    pd.DataFrame({'Name': te.Name, 'Diagnosis': pred}).to_csv(path, index=False); log('wrote', path)
    if a.submit:
        r = subprocess.run(['kaggle', 'competitions', 'submit', '-c', COMP, '-f', path, '-m', msg], capture_output=True, text=True)
        log('submit:', (r.stdout + r.stderr).strip()[-200:])


net = Net().to(dev); t0 = time.time()
# ---- step 0: teacher = frozen backbone + balanced L2 logistic head on all labelled cells ----
y = torch.tensor(tr.Diagnosis.values, dtype=torch.float32, device=dev)


def fit_lp(m):  # balanced L2 logistic regression on m's (current) features of all labelled cells -> m.head, m.fm, m.fs
    X = run(m, tr.Name, 'train', lambda m, x: m.feats(x)); mu, sd = X.mean(0), X.std(0) + 1e-6; X = (X - mu) / sd
    n = len(y); sw = torch.where(y == 1, n / (2 * y.sum()), n / (2 * (n - y.sum())))
    w = torch.zeros(1536, device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
    lb = torch.optim.LBFGS([w, b], lr=1, max_iter=200, tolerance_grad=1e-5, line_search_fn='strong_wolfe')
    def closure():
        lb.zero_grad(); l = a.C * (sw * F.binary_cross_entropy_with_logits(X @ w + b, y, reduction='none')).sum() + .5 * (w ** 2).sum(); l.backward(); return l
    lb.step(closure)
    with torch.no_grad(): m.fm.copy_(mu); m.fs.copy_(sd); m.head.weight.copy_(w.view(1, -1)); m.head.bias.copy_(b)


fit_lp(net)
if a.load:  # prediction only (after the LP fit, which WiSE-FT needs): checkpoints hold LoRA + heads + feature standardisation
    lp = {k: v.clone() for k, v in net.state_dict().items() if k.startswith('head')}
    for f in a.load.split(','):
        ck = torch.load(f)
        for k in ck:
            if 'lora_B' in k: ck[k] = ck[k] * a.wise
            if k in lp: ck[k] = lp[k] + a.wise * (ck[k] - lp[k])
        net.load_state_dict(ck, strict=False)
        pr = predict(net); np.save(f.replace('.pt', f'_tta{a.tta}_wise{a.wise}.npy'), pr)
        log(f'{f}: tta{a.tta} wise{a.wise} mean {pr.mean():.3f} {time.time() - t0:.0f}s')
        if a.holdout:
            y, pg = te.Diagnosis.values, pd.Series(pr).groupby(te.pat.values)
            log(f'  HOLDOUT loaded: pooled AUC {roc_auc_score(y, pr):.4f} | patient-mean AUC {roc_auc_score(y, pg.transform("mean").values):.3f}')
    sys.exit()
if a.init:
    lp = {k: v.clone() for k, v in net.state_dict().items() if k.startswith('head')}; ck = torch.load(a.init)
    for k in ck:
        if 'lora_B' in k: ck[k] = ck[k] * a.wise
        if k in lp: ck[k] = lp[k] + a.wise * (ck[k] - lp[k])
    net.load_state_dict(ck, strict=False); log(f'init from {a.init} with wise {a.wise}')
p0 = predict(net); np.save(f'{a.out}/teacher_size{a.size}.npy', p0)
log(f'teacher ready {time.time() - t0:.0f}s | test mean {p0.mean():.3f} | train AUC, per-patient mean {diag(net)}')
def heval(pred, nm):  # holdout only: pooled cell AUC (the Kaggle metric), AUC of true-patient means, per-patient mean prob
    if not a.holdout: return
    y, pg = te.Diagnosis.values, pd.Series(pred).groupby(te.pat.values)
    log(f'  HOLDOUT {nm}: pooled AUC {roc_auc_score(y, pred):.4f} | patient-mean AUC {roc_auc_score(y, pg.transform("mean").values):.3f} | '
        f'per-patient mean {pg.mean().round(3).to_dict()} | label {te.groupby("pat").Diagnosis.first().to_dict()}')
heval(p0, 'frozen teacher')
if a.epochs == 0:
    save_sub(p0, f'teacher_size{a.size}', f'Frozen DINOv3 ViT-S, per-image std BF+FL, CLS+mean-patch, balanced LR, size {a.size} (teacher)')
    sys.exit()

# ---- student / EMA teacher ----
teacher = copy.deepcopy(net).eval()
for q in teacher.parameters(): q.requires_grad_(False)
net.bb.print_trainable_parameters()
opt = torch.optim.AdamW([q for q in net.parameters() if q.requires_grad], lr=a.lr, weight_decay=.05)
spe = len(te) // a.bs; total = a.epochs * spe
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 100) * .5 * (1 + math.cos(math.pi * i / total)))
lw = 1 / (tr.groupby('pat').Name.transform('size') * tr.groupby('Diagnosis').pat.transform('nunique')).values  # each patient equal, classes 50/50
ll = iter(loader(tr.Name, 'train', bs=a.bs, sampler=WeightedRandomSampler(lw, total * a.bs), drop_last=True))
yl, T = y, torch.tensor(p0, device=dev)
if a.ssl:  # labels never enter training: student (masked view 1) matches EMA teacher on cls of view 2 and on masked patch tokens of view 1
    for ep in range(a.epochs):
        net.train(); t0 = time.time(); sc = sp = 0.
        for i, (iu, xu) in enumerate(loader(te.Name, TS, bs=a.bs, shuffle=True, drop_last=True)):
            il, xl = next(ll)
            x = torch.cat([xl, xu]).to(dev, non_blocking=True).float() / 255; v1, v2 = strong(x), strong(x)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad(): _, P1, _ = teacher.toks(v1); c2, _, _ = teacher.toks(v2)
                cs, Ps, m = net.toks(v1, mask=a.mask or .5)
            lc = 1 - F.cosine_similarity(cs.float(), c2.float(), -1).mean(); lp_ = 1 - F.cosine_similarity(Ps.float()[m], P1.float()[m], -1).mean()
            opt.zero_grad(set_to_none=True); (lc + lp_).backward(); opt.step(); sched.step()
            with torch.no_grad():
                for pt, ps in zip(teacher.parameters(), net.parameters()):
                    if ps.requires_grad: pt.lerp_(ps, 1 - a.ema)
            sc += lc.item(); sp += lp_.item()
            if i % 100 == 0: log(f'  ssl ep{ep} step {i}/{spe} cls {sc / (i + 1):.4f} patch {sp / (i + 1):.4f} | teacher cls batch-std {c2.float().std(0).mean():.3f} {time.time() - t0:.0f}s')
        fit_lp(teacher); pt_ = predict(teacher); auc, per = diag(teacher); g = f'ssl_r{a.rank}_mask{a.mask or .5}_seed{a.seed}_ep{ep + 1}' + (f'_ho{len(a.holdout.split(","))}' if a.holdout else '')
        log(f'ssl epoch {ep + 1} {time.time() - t0:.0f}s | spearman ema-probe~frozen probe {np.corrcoef(rank(pt_), rank(p0))[0, 1]:.3f} | labelled train AUC {auc:.3f} | {per}')
        torch.save({k: v for k, v in teacher.state_dict().items() if 'lora' in k or k.startswith(('head', 'fm', 'fs'))}, f'{a.out}/emateacher_{g}.pt')
        np.save(f'{a.out}/emateacher_{g}.npy', pt_); heval(pt_, f'SSL EMA ep{ep + 1}')
        pd.DataFrame({'Name': te.Name, 'Diagnosis': pt_}).to_csv(f'{a.out}/emateacher_{g}.csv', index=False); log('wrote', g)
    sys.exit()
tag = (f'size{a.size}_r{a.rank}b{a.lora_blocks or a.blocks}_ema{a.ema}_seed{a.seed}' + (f'_mask{a.mask}' if a.mask else '') + (f'_mlp{a.mlp}' if a.mlp else '') + ('_fuse' if a.fuse else '') + (f'_L{a.layer}' if a.layer != 11 else '') + (f'_{a.ptune}' if a.ptune != 'lora' else '')
       + ('_attn' if a.attnpool else '') + ('_init' if a.init else '') + (f'_wu{a.wu}' if a.wu != 1 else '') + (f'_ho{len(a.holdout.split(","))}' if a.holdout else ''))

for ep in range(a.epochs):
    net.train(); t0 = time.time(); ss = su = sm = 0.
    for i, (iu, xu) in enumerate(loader(te.Name, TS, bs=a.bs, shuffle=True, drop_last=True)):
        il, xl = next(ll)
        xu, xl = xu.to(dev, non_blocking=True), xl.to(dev, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if a.ema < 1:
                with torch.no_grad(): tgt = teacher(xu).float().sigmoid()
            else: tgt = T[iu.to(dev)]
            zl, zu = net(torch.cat([xl, xu]), aug=True).float().split([len(xl), len(xu)])
            zm = net(xu, aug=True, mask=a.mask).float() if a.mask else None
        ls = F.binary_cross_entropy_with_logits(zl, yl[il.to(dev)]); lu = F.binary_cross_entropy_with_logits(zu, tgt)
        lm = F.binary_cross_entropy_with_logits(zm, tgt) if a.mask else torch.zeros((), device=dev)
        opt.zero_grad(set_to_none=True); (ls + a.wu * lu + a.wm * lm).backward(); opt.step(); sched.step()
        if a.ema < 1:
            with torch.no_grad():
                for pt, ps in zip(teacher.parameters(), net.parameters()):
                    if ps.requires_grad: pt.lerp_(ps, 1 - a.ema)
        ss += ls.item(); su += lu.item(); sm += lm.item()
        if i % 100 == 0: log(f'  ep{ep} step {i}/{spe} sup {ss / (i + 1):.4f} pseudo {su / (i + 1):.4f} masked {sm / (i + 1):.4f} {time.time() - t0:.0f}s')
    ps_ = predict(net); pt_ = predict(teacher) if a.ema < 1 else p0
    auc, per = diag(net)
    log(f'epoch {ep + 1} {time.time() - t0:.0f}s | spearman student~frozen teacher {np.corrcoef(rank(ps_), rank(p0))[0, 1]:.3f}'
        f' ema~frozen {np.corrcoef(rank(pt_), rank(p0))[0, 1]:.3f} | test mean student {ps_.mean():.3f} ema {pt_.mean():.3f} frozen {p0.mean():.3f}'
        f' | labelled train AUC {auc:.3f}\n  per-patient mean {per}')
    g = f'{tag}_ep{ep + 1}'
    np.save(f'{a.out}/student_{g}.npy', ps_)
    for nm, m in (('student', net), ('emateacher', teacher)):
        torch.save({k: v for k, v in m.state_dict().items() if 'lora' in k or (a.ptune != 'lora' and 'blocks' in k) or k.startswith(('head', 'extra', 'att', 'mod', 'fm', 'fs'))}, f'{a.out}/{nm}_{g}.pt')
    np.save(f'{a.out}/emateacher_{g}.npy', pt_)
    heval(ps_, f'student ep{ep + 1}'); heval(pt_, f'EMA ep{ep + 1}')
    save_sub(ps_, f'student_{g}', f'Mean Teacher student {g}')
    if a.ema < 1: save_sub(pt_, f'emateacher_{g}', f'Mean Teacher EMA teacher {g}')
    # ponytail: student+frozen rank average dropped (worse than either Mean Teacher model in every epoch on 2026-11-15)
