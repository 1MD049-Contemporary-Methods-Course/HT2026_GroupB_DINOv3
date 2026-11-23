"""Overlay videos: one or more runs side by side on the same clip (main slide material).

  # one run, DAVIS
  python make_video.py --seq lab-coat --runs v3-s16_1.5x v3-s16_2x --ref
  # custom clip, three generations + SAM 2 pseudo-GT panel
  python make_video.py --data data/custom --results results_custom --seq twin-mugs \
      --runs v1-s16_1x v2r-s14_1x v3-s16_1x --ref sam2

Panels are laid out in one row (up to 4) or a grid. Output: videos/<seq>__<runs>.mp4.
--blur_faces blurs detected faces with OpenCV's YuNet detector; it needs
checkpoints/face_detection_yunet_2023mar.onnx (not downloaded by default). Clips with people
must not go on shared slides without it.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from vos.propagate import PALETTE  # noqa: E402

YUNET = os.path.join(ROOT, "checkpoints", "face_detection_yunet_2023mar.onnx")


def overlay(img, lab, alpha=0.5):
    vis = img.copy()
    for k in np.unique(lab):
        if k in (0, 255):
            continue
        col = PALETTE[k][::-1].astype(np.float32)
        m = lab == k
        vis[m] = ((1 - alpha) * vis[m] + alpha * col).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, tuple(int(c) for c in col), 2)
    return vis


def title(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class FaceBlur:
    def __init__(self, shape, hold=8):
        if not os.path.exists(YUNET):
            sys.exit(f"--blur_faces needs {YUNET} (OpenCV zoo YuNet model); not found")
        h, w = shape
        self.det = cv2.FaceDetectorYN.create(YUNET, "", (w, h), score_threshold=0.6)
        self.last, self.age, self.hold = [], 0, hold

    def __call__(self, img):
        _, faces = self.det.detect(img)
        boxes = [] if faces is None else [f[:4].astype(int) for f in faces]
        if boxes:
            self.last, self.age = boxes, 0
        elif self.age < self.hold:  # keep the previous box for a few frames (missed detections)
            boxes, self.age = self.last, self.age + 1
        out = img.copy()
        for x, y, w, h in boxes:
            x0, y0 = max(0, x - w // 3), max(0, y - h // 3)
            x1, y1 = min(img.shape[1], x + w + w // 3), min(img.shape[0], y + h + h // 3)
            if x1 > x0 and y1 > y0:
                out[y0:y1, x0:x1] = cv2.GaussianBlur(out[y0:y1, x0:x1], (0, 0), 25)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "DAVIS"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--seq", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", help="panel titles (default: run names)")
    ap.add_argument("--ref", nargs="?", const="gt", choices=["gt", "sam2"],
                    help="add a reference panel: gt = Annotations/, sam2 = Annotations_sam2/")
    ap.add_argument("--fps", type=float, default=24)
    ap.add_argument("--blur_faces", action="store_true")
    ap.add_argument("--cols", type=int, help="panels per row (default: all in one row up to 4)")
    ap.add_argument("--out")
    a = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(a.data, "JPEGImages", "480p", a.seq, "*.jpg")))
    panels = [(lab, os.path.join(a.results, r, a.seq))
              for r, lab in zip(a.runs, a.labels or a.runs)]
    if a.ref:
        sub = "Annotations" if a.ref == "gt" else "Annotations_sam2"
        panels.append(("reference" if a.ref == "gt" else "SAM 2 (pseudo-GT)",
                       os.path.join(a.data, sub, "480p", a.seq)))
    for _, d in panels:
        if not os.path.isdir(d):
            sys.exit(f"missing {d}")

    h0, w0 = cv2.imread(frames[0]).shape[:2]
    n = len(panels)
    cols = a.cols or (n if n <= 4 else int(np.ceil(np.sqrt(n))))
    rows = int(np.ceil(n / cols))
    W, H = cols * w0, rows * h0
    W, H = W + W % 2, H + H % 2
    out = a.out or os.path.join(ROOT, "videos", f"{a.seq}__{'+'.join(a.runs)}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    import imageio_ffmpeg
    wr = imageio_ffmpeg.write_frames(out, (W, H), fps=a.fps, codec="libx264", quality=8,
                                     macro_block_size=1)
    wr.send(None)
    blur = FaceBlur((h0, w0)) if a.blur_faces else None
    for f in frames:
        img = cv2.imread(f)
        if blur:
            img = blur(img)
        canvas = np.zeros((H, W, 3), np.uint8)
        for i, (lab, d) in enumerate(panels):
            p = os.path.join(d, os.path.basename(f).replace(".jpg", ".png"))
            tile = overlay(img, np.array(Image.open(p))) if os.path.exists(p) else img.copy()
            tile = title(tile, f"{lab}  #{os.path.basename(f)[:-4]}")
            r, c = divmod(i, cols)
            canvas[r * h0:(r + 1) * h0, c * w0:(c + 1) * w0] = tile
        wr.send(np.ascontiguousarray(canvas[:, :, ::-1]))
    wr.close()
    print(f"{out}  ({len(frames)} frames, {W}x{H})")


if __name__ == "__main__":
    main()
