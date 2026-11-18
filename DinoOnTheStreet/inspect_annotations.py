import pickle
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(".").resolve()
ANN = (
    ROOT
    / "annotations"
    / "2021.05.12.22.00.38_veh-35_01008_01518_infos.pkl"
)


def describe(value):
    if isinstance(value, np.ndarray):
        text = f"ndarray shape={value.shape}, dtype={value.dtype}"
        if value.size <= 20:
            text += f"\n{value}"
        return text

    if isinstance(value, dict):
        return f"dict keys={list(value)}"

    if isinstance(value, (list, tuple)):
        if len(value) <= 16:
            return repr(value)
        return f"{type(value).__name__}, length={len(value)}"

    return repr(value)


def path_candidates(raw):
    path = Path(str(raw))
    options = [
        path,
        ROOT / path,
        ROOT / "nuplan_data" / path,
    ]

    # Handle converter paths containing an obsolete machine-specific prefix.
    text = str(raw).replace("\\", "/")
    marker = "nuplan-v1.1/"
    if marker in text:
        suffix = text.split(marker, 1)[1]
        options.append(ROOT / "nuplan_data" / "nuplan-v1.1" / suffix)

    return list(dict.fromkeys(options))


# Only load trusted pickle files.
with ANN.open("rb") as handle:
    data = pickle.load(handle)

infos = data["infos"]
first = infos[0]

print("ALL FIRST-RECORD KEYS")
for key, value in first.items():
    print(f"{key}: {describe(value)}")

print("\nFIRST-RECORD CAMERA METADATA")
for camera_name, camera in first["cams"].items():
    print(f"\n{camera_name}")

    if not isinstance(camera, dict):
        print(describe(camera))
        continue

    for key, value in camera.items():
        print(f"  {key}: {describe(value)}")

        if isinstance(value, str) and (
            "path" in key.lower()
            or value.lower().endswith((".jpg", ".jpeg", ".png"))
        ):
            existing = [p for p in path_candidates(value) if p.is_file()]
            print(f"    Resolved files: {[str(p) for p in existing]}")

classes = Counter()
valid_classes = Counter()
scenes = Counter()
box_counts = []
timestamps = []

for info in infos:
    names = np.asarray(info.get("gt_names", [])).reshape(-1)
    classes.update(str(name) for name in names)
    box_counts.append(len(names))
    scenes[str(info.get("scene_token", "<missing>"))] += 1

    if "timestamp" in info:
        timestamps.append(int(info["timestamp"]))

    flags = np.asarray(info.get("valid_flag", []), dtype=bool).reshape(-1)
    if len(flags) == len(names):
        valid_classes.update(str(name) for name in names[flags])

print("\nDATASET SUMMARY")
print(f"Annotation records: {len(infos)}")
print(f"Distinct scene_token values: {len(scenes)}")
print(f"Records per scene: {scenes.most_common(15)}")

print("\nClass counts: annotated appearances, not unique objects")
for name, count in classes.most_common():
    print(f"  {name}: {count}")

print("\nClass counts after stored valid_flag filter")
for name, count in valid_classes.most_common():
    print(f"  {name}: {count}")

print("\nObjects per annotation record")
print(f"  min={min(box_counts)}")
print(f"  median={np.median(box_counts):.1f}")
print(f"  max={max(box_counts)}")

if timestamps:
    unique_times = np.unique(np.asarray(timestamps, dtype=np.int64))
    print("\nTimestamp diagnostics: raw units")
    print(f"  first={unique_times[0]}")
    print(f"  last={unique_times[-1]}")
    print(f"  duplicate timestamps={len(timestamps) - len(unique_times)}")

    if len(unique_times) > 1:
        deltas = np.diff(unique_times)
        print(f"  median positive interval={np.median(deltas)}")
        print(f"  largest interval={deltas.max()}")

print("\nFIRST FIVE BOXES")
print(np.asarray(first.get("gt_boxes", []))[:5])

print("\nFIRST FIVE LABELS")
print(np.asarray(first.get("gt_names", []))[:5])
