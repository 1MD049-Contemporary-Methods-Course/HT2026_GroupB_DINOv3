"""Model-agnostic patch-feature extraction.

`FeatureExtractor.extract(img)` takes a (3, H, W) float tensor in [0, 1], RGB, already
resized to (grid_h * patch, grid_w * patch), and returns (grid_h, grid_w, C) L2-normalised
patch features with class and register tokens removed.
"""
import os
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A stale user-level HF_TOKEN on this machine overrides the valid `hf auth login` token and
# blocks the gated DINOv3 repos. Drop it unless explicitly asked to keep it.
if "HF_TOKEN" in os.environ and not os.environ.get("DINO_VOS_KEEP_HF_TOKEN"):
    os.environ.pop("HF_TOKEN")
    print("[features] ignoring HF_TOKEN env var, using stored HF login "
          "(set DINO_VOS_KEEP_HF_TOKEN=1 to keep it)")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
DINO_REPO = os.path.join(ROOT, "third_party", "dino")

# Normalisation. The original eval_video_segmentation.py uses std 0.228 for the red channel
# (inherited from UVC) instead of the ImageNet 0.229. We keep it for v1 so step 1 is a
# faithful reproduction; v2 / v3 use their own processors' ImageNet statistics.
V1_SCRIPT_MEAN, V1_SCRIPT_STD = (0.485, 0.456, 0.406), (0.228, 0.224, 0.225)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

MODELS = {
    # name: (family, id, patch)
    "v1-s16": ("v1", "dino_vits16", 16),
    "v1-s8": ("v1", "dino_vits8", 8),
    "v2r-s14": ("hf", "facebook/dinov2-with-registers-small", 14),
    "v3-s16": ("hf", "facebook/dinov3-vits16-pretrain-lvd1689m", 16),
    "v3-s16plus": ("hf", "facebook/dinov3-vits16plus-pretrain-lvd1689m", 16),
    "v3-b16": ("hf", "facebook/dinov3-vitb16-pretrain-lvd1689m", 16),
    "v3-l16": ("hf", "facebook/dinov3-vitl16-pretrain-lvd1689m", 16),
}


class FeatureExtractor:
    def __init__(self, name, layer=None, device="cuda", pad=0):
        """layer: None = last block; otherwise 1-based block index (1..depth).
        pad: reflect-pad the input by this many patches on every side and crop the features
        back to the original grid (ablation for image-border effects)."""
        family, model_id, patch = MODELS[name]
        self.name, self.family, self.patch, self.device = name, family, patch, device
        self.layer = layer
        self.pad = pad
        if family == "v1":
            if os.path.isdir(DINO_REPO):
                self.model = torch.hub.load(DINO_REPO, model_id, source="local")
            else:  # no local clone: fetch the hub entry point from GitHub
                self.model = torch.hub.load("facebookresearch/dino:main", model_id)
            self.depth = len(self.model.blocks)
            mean, std = V1_SCRIPT_MEAN, V1_SCRIPT_STD
            self.n_prefix = 1
        else:
            from transformers import AutoModel, AutoConfig
            cfg = AutoConfig.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id, dtype=torch.float32)
            self.depth = cfg.num_hidden_layers
            self.n_prefix = 1 + getattr(cfg, "num_register_tokens", 0)
            mean, std = IMAGENET_MEAN, IMAGENET_STD
            try:
                from transformers import AutoImageProcessor
                proc = AutoImageProcessor.from_pretrained(model_id)
                mean, std = tuple(proc.image_mean), tuple(proc.image_std)
            except Exception:
                pass
        self.mean = torch.tensor(mean, device=device).view(3, 1, 1)
        self.std = torch.tensor(std, device=device).view(3, 1, 1)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        if layer is not None and not (1 <= layer <= self.depth):
            raise ValueError(f"layer must be in 1..{self.depth}")

    def _final_norm(self):
        for attr in ("layernorm", "norm"):
            m = getattr(self.model, attr, None)
            if isinstance(m, torch.nn.Module):
                return m
        raise AttributeError("could not find final norm layer")

    @torch.no_grad()
    def extract(self, img):
        _, H, W = img.shape
        P = self.pad
        h, w = H // self.patch + 2 * P, W // self.patch + 2 * P
        x = ((img.to(self.device) - self.mean) / self.std).unsqueeze(0)
        if P:
            q = P * self.patch
            x = F.pad(x, (q, q, q, q), mode="reflect")
        if self.family == "v1":
            # get_intermediate_layers returns norm(output) of the last n blocks
            n = 1 if self.layer is None else self.depth - self.layer + 1
            out = self.model.get_intermediate_layers(x, n=n)[0]
        else:
            if self.layer is None:
                out = self.model(pixel_values=x).last_hidden_state
            else:
                hs = self.model(pixel_values=x, output_hidden_states=True).hidden_states
                # hidden_states[0] = embeddings, [k] = output of block k; apply final norm
                # so intermediate layers are treated like v1's get_intermediate_layers.
                out = self._final_norm()(hs[self.layer])
        out = out[0, self.n_prefix:, :]
        assert out.shape[0] == h * w, (out.shape, h, w, self.n_prefix)
        out = F.normalize(out, dim=-1, p=2).reshape(h, w, -1)
        if P:
            out = out[P:h - P, P:w - P]
        return out
