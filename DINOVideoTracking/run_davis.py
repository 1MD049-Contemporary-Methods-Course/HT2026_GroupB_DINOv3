"""Run label propagation on DAVIS 2017 val (or a DAVIS-layout custom root) and evaluate.

Example:
  python run_davis.py --model v1-s16 --scale 1
  python run_davis.py --model v3-s16 --scale 1 2 --seqs bike-packing dog
"""
import argparse
import csv
import datetime
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from vos.features import FeatureExtractor, MODELS  # noqa: E402
from vos.propagate import track_sequence, list_frames, DEFAULTS  # noqa: E402
from vos.cache import SequenceFeatureCache, DEFAULT_ROOT as DEFAULT_CACHE_ROOT  # noqa: E402


def scale_tag(s):
    return f"{s:g}x"


def run_name(model, scale, layer):
    return f"{model}_{scale_tag(scale)}" + (f"_L{layer}" if layer else "")


def evaluate(davis_root, res_dir, seqs):
    from davis2017.evaluation import DAVISEvaluation
    ev = DAVISEvaluation(davis_root=davis_root, task="semi-supervised", gt_set="val",
                         sequences=seqs if seqs else "all")
    m = ev.evaluate(res_dir)
    J, F = m["J"], m["F"]
    jm, fm = float(np.mean(J["M"])), float(np.mean(F["M"]))
    g = {"J&F-Mean": (jm + fm) / 2, "J-Mean": jm, "J-Recall": float(np.mean(J["R"])),
         "J-Decay": float(np.mean(J["D"])), "F-Mean": fm, "F-Recall": float(np.mean(F["R"])),
         "F-Decay": float(np.mean(F["D"]))}
    per = [(k, J["M_per_object"][k], F["M_per_object"][k]) for k in J["M_per_object"]]
    # same files as the toolkit's evaluation_method.py
    tag = "val" if not seqs else "subset"
    with open(os.path.join(res_dir, f"global_results-{tag}.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(g.keys()); wr.writerow([f"{v:.3f}" for v in g.values()])
    with open(os.path.join(res_dir, f"per-sequence_results-{tag}.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["Sequence", "J-Mean", "F-Mean"])
        for k, j, fv in per:
            wr.writerow([k, f"{j:.3f}", f"{fv:.3f}"])
    return g, per


def append_csv(path, row):
    """Append a row; if the row has columns the file lacks, rewrite with the union."""
    rows, fields = [], list(row.keys())
    if os.path.exists(path):
        with open(path, newline="") as f:
            rd = csv.DictReader(f)
            old = rd.fieldnames or []
            if set(fields) <= set(old):
                fields = old
            else:
                rows = list(rd)
                fields = old + [k for k in fields if k not in old]
    if rows or not os.path.exists(path):
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields, restval="")
            wr.writeheader()
            wr.writerows(rows)
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields, restval="").writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--scale", type=float, nargs="+", default=[1.0])
    ap.add_argument("--layer", type=int, default=None, help="1-based block; default last")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "DAVIS"))
    ap.add_argument("--dataset", default="davis17-val", help="label written to the CSVs")
    ap.add_argument("--seq_list", default=None, help="default <data>/ImageSets/2017/val.txt")
    ap.add_argument("--seqs", nargs="*", default=None, help="subset of sequences")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--no_cache", action="store_true", help="do not read/write the feature cache")
    ap.add_argument("--pad", type=int, default=0,
                    help="reflect-pad input by this many patches per side (ablation; disables cache)")
    ap.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--cache_scales", type=float, nargs="*", default=[1.0],
                    help="scales whose features are cached (1x only by default: ~2.3 GB per "
                         "384-dim model on DAVIS val; 2x is 4x that)")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()

    seq_list = args.seq_list or os.path.join(args.data, "ImageSets", "2017", "val.txt")
    seqs = [s.strip() for s in open(seq_list) if s.strip()]
    if args.seqs:
        seqs = [s for s in seqs if s in args.seqs]
    os.makedirs(args.results, exist_ok=True)

    ext = FeatureExtractor(args.model, layer=args.layer, pad=args.pad)
    if args.pad:
        args.no_cache = True  # cache keys do not include padding
    print(f"{args.model}: patch {ext.patch}, depth {ext.depth}, prefix tokens {ext.n_prefix}")

    for scale in args.scale:
        name = run_name(args.model, scale, args.layer)
        res_dir = os.path.join(args.results, name)
        stats = []
        for i, seq in enumerate(seqs):
            frames = list_frames(os.path.join(args.data, "JPEGImages", "480p", seq))
            ann = os.path.join(args.data, "Annotations", "480p", seq, "00000.png")
            cache = None
            if not args.no_cache and scale in args.cache_scales:
                cache = (lambda grid, seq=seq, n=len(frames): SequenceFeatureCache(
                    args.cache_root, args.dataset, args.model, args.layer or ext.depth,
                    seq, n, grid))
            try:
                st = track_sequence(ext, frames, ann, os.path.join(res_dir, seq), scale=scale,
                                    n_last_frames=args.n_last_frames, topk=args.topk,
                                    temperature=args.temperature, radius=args.radius,
                                    feature_cache=cache)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM on {seq} at {scale_tag(scale)}; skipping this cell")
                torch.cuda.empty_cache()
                stats = None
                break
            stats.append(st)
            print(f"[{name}] {i + 1}/{len(seqs)} {seq}: {st['n_frames']} frames, "
                  f"{st['sec_per_frame']:.3f} s/frame, peak {st['peak_mem_mb']:.0f} MB, grid {st['grid']}")
        if stats is None or args.no_eval:
            continue
        g, per = evaluate(args.data, res_dir, args.seqs)
        n_frames = sum(s["n_frames"] for s in stats)
        row = dict(timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
                   dataset=args.dataset + ("" if not args.seqs else f"[{len(seqs)} seqs]"),
                   model=args.model, scale=scale_tag(scale),
                   layer=args.layer or ext.depth, JF=round(g["J&F-Mean"] * 100, 2),
                   J=round(g["J-Mean"] * 100, 2), F=round(g["F-Mean"] * 100, 2),
                   sec_per_frame=round(sum(s["seconds"] for s in stats) / n_frames, 4),
                   tokens=stats[0]["tokens"], grid=stats[0]["grid"], input=stats[0]["input"],
                   radius=stats[0]["radius"],
                   peak_mem_mb=round(max(s["peak_mem_mb"] for s in stats)),
                   n_frames=n_frames,
                   features=("cache" if all(s["cached"] for s in stats) else
                             "computed" if not any(s["cached"] for s in stats) else "mixed")
                            + ("-fp16" if (not args.no_cache and scale in args.cache_scales)
                               else "-fp32"),
                   torch=torch.__version__)
        append_csv(os.path.join(args.results, "summary.csv"), row)
        for k, j, fv in per:
            append_csv(os.path.join(args.results, "per_sequence.csv"),
                       dict(run=name, dataset=row["dataset"], sequence=k,
                            J=round(j * 100, 2), F=round(fv * 100, 2)))
        print(f"== {name}: J&F {row['JF']:.2f}  J {row['J']:.2f}  F {row['F']:.2f}  "
              f"({row['sec_per_frame']} s/frame, peak {row['peak_mem_mb']} MB)")


if __name__ == "__main__":
    main()
