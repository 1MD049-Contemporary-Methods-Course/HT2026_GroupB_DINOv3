"""Print the J&F table (model x scale) from results/summary.csv, latest full-val row per cell.

  python make_table.py            # J&F
  python make_table.py --metric J
"""
import argparse
import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
ORDER = ["v1-s16", "v1-s8", "v2r-s14", "v3-s16", "v3-s16plus", "v3-b16", "v3-l16"]

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=os.path.join(ROOT, "results", "summary.csv"))
ap.add_argument("--metric", default="JF", choices=["JF", "J", "F", "sec_per_frame", "peak_mem_mb"])
ap.add_argument("--dataset", default="davis17-val")
args = ap.parse_args()

df = pd.read_csv(args.csv)
df = df[df["dataset"] == args.dataset]
if "layer" in df:  # last-layer rows only (layer sweep is reported separately)
    last = df.groupby("model")["layer"].transform("max")
    df = df[df["layer"] == last]
df = df.drop_duplicates(["model", "scale"], keep="last")
t = df.pivot(index="model", columns="scale", values=args.metric)
t = t.reindex([m for m in ORDER if m in t.index])
t = t[[c for c in ["1x", "1.5x", "2x"] if c in t.columns]]
print(f"{args.metric} on {args.dataset}")
print(t.to_string(float_format=lambda v: f"{v:.2f}"))
