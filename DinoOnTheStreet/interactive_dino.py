"""Small utilities shared by the interactive DINOv3 viewer."""
from pathlib import Path

import cv2
import numpy as np


COLORS = {
    "vehicle": (0, 255, 0),
    "pedestrian": (255, 255, 0),
    "bicycle": (0, 255, 255),
}


def resolve_image(root, raw_path):
    """Resolve NuPlan image paths despite converter machine prefixes."""
    root = Path(root)
    path = Path(raw_path)
    candidates = [path, root / path, root / "nuplan_data" / path]

    text = str(raw_path).replace("\\", "/")
    marker = "nuplan-v1.1/"
    if marker in text:
        suffix = text.split(marker, 1)[1]
        candidates.append(root / "nuplan_data" / "nuplan-v1.1" / suffix)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(raw_path)


def _box_corners(box):
    x, y, z, width, length, height, yaw = box[:7]
    corners = np.array([
        [length, width, height], [length, -width, height],
        [-length, -width, height], [-length, width, height],
        [length, width, -height], [length, -width, -height],
        [-length, -width, -height], [-length, width, -height],
    ], dtype=np.float64) / 2
    angle = -yaw
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return corners @ rotation.T + np.array([x, y, z])


def draw_boxes(image, info, camera_name, original_size):
    """Draw projected NuPlan 3D boxes on the resized BGR camera frame."""
    camera = info["cams"][camera_name]
    rotation = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
    translation = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
    intrinsic = np.asarray(camera["cam_intrinsic"], dtype=np.float64)
    original_width, original_height = original_size
    scale_x = image.shape[1] / original_width
    scale_y = image.shape[0] / original_height
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))

    for box, category in zip(info.get("gt_boxes", []), info.get("gt_names", [])):
        category = str(category)
        box = np.asarray(box, dtype=np.float64)
        if box.size < 7 or not np.isfinite(box[:7]).all() or np.any(box[3:6] <= 0):
            continue
        camera_corners = (_box_corners(box) - translation) @ rotation
        if np.any(camera_corners[:, 2] <= 0.5):
            continue
        projected = camera_corners @ intrinsic.T
        pixels = projected[:, :2] / projected[:, 2:3]
        pixels[:, 0] *= scale_x
        pixels[:, 1] *= scale_y
        points = np.round(pixels).astype(np.int32)
        color = COLORS.get(category, (0, 165, 255))
        for start, end in edges:
            cv2.line(image, tuple(points[start]), tuple(points[end]), color, 1, cv2.LINE_AA)
    return image
