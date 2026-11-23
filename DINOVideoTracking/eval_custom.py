"""Score runs on the custom clips against whatever reference masks exist.

Reference = hand masks in Annotations/ (frames other than 0) if any, else SAM 2 pseudo ground
truth in Annotations_sam2/ (every --every-th frame). Frame 0 is never scored.

Per run and sequence: mean IoU per object over reference frames, and the identity-swap rate:
on frames where both reference objects i and j are present, prediction i is "swapped" if it
overlaps reference j more than reference i. Swap rate = swapped (frame, object) pairs / pairs.

The swap rate misses "merges", where one label covers both twins (prediction j covers reference
i and j, so neither prediction overlaps the wrong object more than its own). Identity confusion
catches merges: the mean, over the same (frame, object) pairs, of the share of reference
object i's pixels that are predicted as another object.

  python eval_custom.py                      # all runs in results_custom/
  python eval_custom.py --runs v3-s16_1x v1-s16_1x --timeline twin-mugs
"""
import argparse
import csv
import glob
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(p):
    return np.array(Image.open(p))


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.nan if u == 0 else np.logical_and(a, b).sum() / u


def reference_frames(root, seq, every):
    hand = sorted(glob.glob(os.path.join(root, "Annotations", "480p", seq, "*.png")))
    hand = [p for p in hand if not p.endswith("00000.png")]
    if hand:
        return "hand", hand
    sam = sorted(glob.glob(os.path.join(root, "Annotations_sam2", "480p", seq, "*.png")))
    sam = [p for p in sam if not p.endswith("00000.png")]
    return "sam2", sam[every - 1::every] if every > 1 else sam


def score(run_dir, root, seq, every):
    kind, refs = reference_frames(root, seq, every)
    if not refs:
        return None
    ids = [int(k) for k in np.unique(load(os.path.join(root, "Annotations", "480p", seq, "00000.png")))
           if k not in (0, 255)]
    ious = {k: [] for k in ids}
    swaps, pairs, timeline = 0, 0, []
    conf = []   # share of reference object i's pixels predicted as another object id
    for rp in refs:
        pp = os.path.join(run_dir, seq, os.path.basename(rp))
        if not os.path.exists(pp):
            continue
        ref, pred = load(rp), load(pp)
        row = {"frame": int(os.path.basename(rp)[:-4])}
        for k in ids:
            if (ref == k).any():
                ious[k].append(iou(pred == k, ref == k))
        present = [k for k in ids if (ref == k).any()]
        for i in present:
            others = [j for j in present if j != i]
            if not others:
                continue
            ri = ref == i
            c = np.isin(pred[ri], others).mean()
            conf.append(c)
            row[f"obj{i}_confused"] = round(float(c), 3)
            own = iou(pred == i, ref == i)
            best_other = max(iou(pred == i, ref == j) for j in others)
            sw = bool(np.nan_to_num(best_other) > np.nan_to_num(own))
            swaps += sw
            pairs += 1
            row[f"obj{i}_swapped"] = int(sw)
        timeline.append(row)
    per_obj = {k: float(np.nanmean(v)) if v else np.nan for k, v in ious.items()}
    return dict(reference=kind, n_ref=len(refs), mIoU=float(np.nanmean(list(per_obj.values()))),
                per_obj=per_obj, swap_rate=(swaps / pairs if pairs else np.nan), pairs=pairs,
                id_confusion=(float(np.mean(conf)) if conf else np.nan), timeline=timeline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results_custom"))
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "custom"))
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--every", type=int, default=1, help="use every k-th SAM 2 frame as reference")
    ap.add_argument("--timeline", help="print the per-frame swap timeline for this sequence")
    a = ap.parse_args()
    runs = a.runs or sorted(d for d in os.listdir(a.results) if os.path.isdir(os.path.join(a.results, d)))
    seqs = [s.strip() for s in open(os.path.join(a.root, "ImageSets", "custom.txt")) if s.strip()]
    rows = []
    for r in runs:
        for s in seqs:
            if not os.path.isdir(os.path.join(a.results, r, s)):
                continue
            res = score(os.path.join(a.results, r), a.root, s, a.every)
            if res is None:
                continue
            rows.append(dict(run=r, sequence=s, reference=res["reference"], n_ref=res["n_ref"],
                             mIoU=round(100 * res["mIoU"], 2),
                             swap_rate=("" if np.isnan(res["swap_rate"]) else round(100 * res["swap_rate"], 1)),
                             id_confusion=("" if np.isnan(res["id_confusion"])
                                           else round(100 * res["id_confusion"], 1)),
                             **{f"IoU_obj{k}": round(100 * v, 2) for k, v in res["per_obj"].items()}))
            if a.timeline == s:
                print(f"{r} {s} swap timeline (1 = swapped):")
                for t in res["timeline"]:
                    print("  ", t)
    out = os.path.join(a.results, "custom_scores.csv")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        print(f"{row['run']:22s} {row['sequence']:16s} mIoU {row['mIoU']:6.2f}  "
              f"swap {row['swap_rate']!s:>5}  ({row['reference']}, {row['n_ref']} frames)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
