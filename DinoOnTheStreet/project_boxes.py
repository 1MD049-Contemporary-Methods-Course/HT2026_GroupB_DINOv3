import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


COLORS = {
    "vehicle": "lime",
    "pedestrian": "cyan",
    "bicycle": "yellow",
}

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def resolve_image(root, raw):
    path = Path(raw)
    candidates = [path, root / path, root / "nuplan_data" / path]

    text = str(raw).replace("\\", "/")
    marker = "nuplan-v1.1/"
    if marker in text:
        suffix = text.split(marker, 1)[1]
        candidates.append(root / "nuplan_data" / "nuplan-v1.1" / suffix)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(raw)


def box_corners(box):
    x, y, z, width, length, height, stored_yaw = box

    corners = np.array([
        [length,  width,  height],
        [length, -width,  height],
        [-length, -width, height],
        [-length, width,  height],
        [length,  width, -height],
        [length, -width, -height],
        [-length, -width, -height],
        [-length, width, -height],
    ], dtype=np.float64) / 2.0

    angle = -stored_yaw
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])

    return corners @ rotation.T + np.array([x, y, z])


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path("."))
parser.add_argument("--ann", type=Path, required=True)
parser.add_argument("--camera", default="CAM_F0")
parser.add_argument("--indices", type=int, nargs="+", default=[0, 200, 500])
parser.add_argument("--classes", nargs="+", default=["vehicle", "pedestrian"])
parser.add_argument("--min-size", type=int, default=24)
parser.add_argument("--max-depth", type=float, default=60.0)
parser.add_argument("--out", type=Path, default=Path("outputs/boxes"))
args = parser.parse_args()

root = args.root.resolve()
args.out.mkdir(parents=True, exist_ok=True)

# Load only trusted pickle files.
with args.ann.open("rb") as handle:
    infos = pickle.load(handle)["infos"]

for index in args.indices:
    info = infos[index]
    camera = info["cams"][args.camera]

    # Converter positions are ego-frame coordinates.
    # This diagnostic uses sensor2lidar only when ego and lidar coincide.
    lidar_translation = np.asarray(info["lidar2ego_translation"])
    lidar_quaternion = np.asarray(info["lidar2ego_rotation"])
    identity_rotation = (
        np.allclose(lidar_quaternion, [1, 0, 0, 0], atol=1e-6)
        or np.allclose(lidar_quaternion, [-1, 0, 0, 0], atol=1e-6)
    )
    if not (
        np.allclose(lidar_translation, 0, atol=1e-6)
        and identity_rotation
    ):
        raise ValueError(
            f"Record {index}: ego and lidar frames differ. "
            "Use an explicit ego-to-camera transform for this record."
        )

    image_path = resolve_image(root, camera["data_path"])
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    image_width, image_height = image.size
    R = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
    t = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
    K = np.asarray(camera["cam_intrinsic"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(image)
    candidates = []
    drawn = 0

    for object_index, (box, category) in enumerate(
        zip(info["gt_boxes"], info["gt_names"])
    ):
        category = str(category)
        if category not in args.classes:
            continue

        box = np.asarray(box, dtype=np.float64)
        if not np.isfinite(box).all() or np.any(box[3:6] <= 0):
            continue

        corners = box_corners(box)
        camera_corners = (corners - t) @ R
        camera_center = (box[:3] - t) @ R
        depth = float(camera_center[2])

        if depth > args.max_depth or np.any(camera_corners[:, 2] <= 0.5):
            continue

        homogeneous = camera_corners @ K.T
        uv = homogeneous[:, :2] / homogeneous[:, 2:3]

        lower = uv.min(axis=0)
        upper = uv.max(axis=0)
        clipped_lower = np.maximum(lower, [0, 0])
        clipped_upper = np.minimum(upper, [image_width, image_height])
        clipped_size = clipped_upper - clipped_lower
        projected_size = upper - lower

        if np.any(projected_size <= 0):
            continue

        if clipped_size[0] < 40 or clipped_size[1] < 64:
            continue

        projected_area = float(np.prod(projected_size))
        retained_area = float(np.prod(np.maximum(clipped_size, 0)))
        retained_fraction = retained_area / projected_area

        if retained_fraction < 0.90:
            continue

        color = COLORS.get(category, "orange")

        for start, end in EDGES:
            ax.plot(
                uv[[start, end], 0],
                uv[[start, end], 1],
                color=color,
                linewidth=1.1,
            )

        x0, y0 = clipped_lower
        x1, y1 = clipped_upper
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor=color, linestyle="--", linewidth=1,
        ))
        ax.text(
            x0, max(12, y0 - 4),
            f"{object_index}: {category} z={depth:.1f}",
            fontsize=8, color=color,
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
        )
        drawn += 1

        crop_bounds = (
            int(np.floor(x0)), int(np.floor(y0)),
            int(np.ceil(x1)), int(np.ceil(y1)),
        )
        crop = image.crop(crop_bounds)
        candidates.append((depth, object_index, category, crop))

    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)
    ax.set_title(
        f"{args.camera} | record {index} | {drawn} candidate boxes\n"
        "Original image; no distortion correction"
    )
    ax.axis("off")
    fig.tight_layout()

    prefix = f"{args.camera}_{index:04d}"
    fig.savefig(args.out / f"{prefix}_boxes.jpg", dpi=150)
    plt.close(fig)

    preview = sorted(candidates, key=lambda item: item[0])[:16]
    if preview:
        rows = (len(preview) + 3) // 4
        fig, axes = plt.subplots(
            rows, 4, figsize=(14, 3.5 * rows), squeeze=False
        )
        for axis in axes.flat:
            axis.axis("off")

        for axis, (depth, object_index, category, crop) in zip(
            axes.flat, preview
        ):
            axis.imshow(crop)
            axis.set_title(
                f"{object_index}: {category}\nz={depth:.1f}",
                fontsize=10,
            )

        fig.suptitle("Candidate crops—not visibility-validated")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(args.out / f"{prefix}_crops.jpg", dpi=150)
        plt.close(fig)

    print(
        f"Record {index}: {drawn} candidate boxes; "
        f"image size={image_width}x{image_height}"
    )
