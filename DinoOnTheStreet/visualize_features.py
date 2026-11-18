import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from torchvision.transforms import functional as TF


parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--weights", type=Path, required=True)
parser.add_argument("--image", type=Path, required=True)
parser.add_argument("--out", type=Path, default=Path("outputs"))
parser.add_argument("--long-side", type=int, default=768)
args = parser.parse_args()

args.out.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load(
    str(args.repo.resolve()),
    "dinov3_vits16",
    source="local",
    weights=str(args.weights.resolve()),
)
model = model.to(device).eval()
model.requires_grad_(False)

image = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")

# Resize while approximately preserving aspect ratio.
width, height = image.size
scale = args.long_side / max(width, height)
new_width = max(16, round(width * scale / 16) * 16)
new_height = max(16, round(height * scale / 16) * 16)

resized = image.resize(
    (new_width, new_height),
    Image.Resampling.BICUBIC,
)

x = TF.to_tensor(resized)
x = TF.normalize(
    x,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)
x = x.unsqueeze(0).to(device)

with torch.inference_mode():
    features = model.get_intermediate_layers(
        x,
        n=1,
        reshape=True,
        norm=True,
    )[0]

features = features[0].float().cpu()
channels, grid_h, grid_w = features.shape

patch_vectors = features.permute(1, 2, 0).reshape(-1, channels).numpy()

pca = PCA(n_components=3, random_state=0)
colors = pca.fit_transform(patch_vectors)

low = np.percentile(colors, 1, axis=0)
high = np.percentile(colors, 99, axis=0)
colors = np.clip((colors - low) / np.maximum(high - low, 1e-8), 0, 1)

colors = colors.reshape(grid_h, grid_w, 3)
color_tensor = torch.from_numpy(colors).permute(2, 0, 1).unsqueeze(0)
color_image = F.interpolate(
    color_tensor,
    size=(new_height, new_width),
    mode="nearest",
)[0].permute(1, 2, 0).numpy()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(resized)
axes[0].set_title("Camera image")

axes[1].imshow(color_image)
axes[1].set_title("DINOv3 patch features: PCA")

axes[2].imshow(resized)
axes[2].imshow(color_image, alpha=0.5)
axes[2].set_title("Overlay")

for ax in axes:
    ax.axis("off")

fig.tight_layout()
fig.savefig(args.out / "feature_pca.png", dpi=160)
plt.close(fig)

torch.save(
    {
        "features": features,
        "original_size_wh": image.size,
        "input_size_wh": resized.size,
        "image_path": str(args.image.resolve()),
    },
    args.out / "features.pt",
)

print(f"Device: {device}")
print(f"Input shape: {tuple(x.shape)}")
print(f"Feature shape: {tuple(features.shape)}")
print(f"PCA explained variance: {pca.explained_variance_ratio_}")
print(f"Saved results to: {args.out.resolve()}")
