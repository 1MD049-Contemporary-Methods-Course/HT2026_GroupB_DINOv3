import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


COLORS = {
    "vehicle": "lime",
    "pedestrian": "cyan",
    "bicycle": "yellow",
    "traffic_cone": "orange",
    "barrier": "magenta",
    "czone_sign": "white",
    "generic_object": "red",
}


def resolve_image(root, raw_path):
    path = Path(raw_path)
    candidates = [
        path,
        root / path,
        root / "nuplan_data" / path,
    ]

    text = str(raw_path).replace("\\", "/")
    marker = "nuplan-v1.1/"
    if marker in text:
        suffix = text.split(marker, 1)[1]
        candidates.append(
            root / "nuplan_data" / "nuplan-v1.1" / suffix
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Cannot resolve image: {raw_path}")


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path("."))
parser.add_argument("--ann", type=Path, required=True)
parser.add_argument("--camera", default="CAM_F0")
parser.add_argument("--indices", type=int, nargs="+", default=[0, 200, 500])
parser.add_argument(
    "--classes",
    nargs="+",
    default=["vehicle", "pedestrian", "bicycle"],
)
parser.add_argument("--out", type=Path, default=Path("outputs/projections"))
args = parser.parse_args()

root = args.root.resolve()
args.out.mkdir(parents=True, exist_ok=True)

# Load only trusted pickle files.
with args.ann.open("rb") as handle:
    infos = pickle.load(handle)["infos"]

for index in args.indices:
    if not 0 <= index < len(infos):
        raise IndexError(f"Record {index}; dataset has {len(infos)} records")

    info = infos[index]
    camera = info["cams"][args.camera]
    image_path = resolve_image(root, camera["data_path"])

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    width, height = image.size

    boxes = np.asarray(info["gt_boxes"], dtype=np.float64)
    names = np.asarray(info["gt_names"]).astype(str)
    centers = boxes[:, :3]

    R = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
    t = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
    K = np.asarray(camera["cam_intrinsic"], dtype=np.float64)

    if not np.allclose(R.T @ R, np.eye(3), atol=1e-4):
        raise ValueError("Rotation matrix is not approximately orthonormal")

    # Row-vector form of R.T @ (point - t).
    camera_points = (centers - t) @ R
    depths = camera_points[:, 2]

    pixels = np.full((len(centers), 2), np.nan)
    in_front = depths > 0.5
    homogeneous = camera_points[in_front] @ K.T
    pixels[in_front] = homogeneous[:, :2] / homogeneous[:, 2:3]

    selected = (
        in_front
        & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
        & np.isin(names, args.classes)
    )

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(image)

    for j in np.flatnonzero(selected):
        u, v = pixels[j]
        color = COLORS.get(names[j], "white")

        ax.scatter(
            u, v, s=45, c=color,
            edgecolors="black", linewidths=0.7,
        )
        ax.text(
            u + 5,
            v - 5,
            f"{j}: {names[j]} z={depths[j]:.1f}",
            color=color,
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
        )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title(
        f"{args.camera} | record {index} | "
        f"{selected.sum()} projected centers\n"
        "Diagnostic projection: coordinate convention not yet validated"
    )
    ax.axis("off")
    fig.tight_layout()

    output = args.out / f"{args.camera}_{index:04d}_centers.jpg"
    fig.savefig(output, dpi=150)
    plt.close(fig)

    print(f"\nRecord {index}")
    print(f"  Image: {image_path}")
    print(f"  Image dimensions: {width} x {height}")
    print(f"  Objects in front: {in_front.sum()}")
    print(f"  Selected centers inside image: {selected.sum()}")
    print(f"  Saved: {output}")
