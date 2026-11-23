"""Turn phone videos into DAVIS-layout sequences and annotate them with SAM 2.

  # 1. frames: short side 480, numbered from 00000.jpg, optional trim / frame-rate reduction
  python clip_tool.py extract data/raw_videos/twins.mp4 --name twins --start 2 --duration 6 --fps 15

  # 2. first-frame mask with SAM 2 clicks (also --frames 0 10 20 or --every 10 for reference masks)
  python clip_tool.py annotate twins

  # 3. optional: SAM 2 video propagation -> Annotations_sam2/ (pseudo ground truth, kept separate)
  python clip_tool.py propagate twins

  # SAM 2 as a "specialist" reference on DAVIS val, written as a results folder:
  python clip_tool.py propagate --root data/DAVIS --all --out results/sam2.1-s_1x

Annotate window: left click = positive point, right click = negative point,
  n = next object, u = undo last point, r = clear current object, s = save, q / Esc = quit
  ([ and ] = previous / next frame when annotating several frames).
Masks are saved as indexed palette PNGs (DAVIS palette): 0 = background, 1..N = objects.
Later objects overwrite earlier ones where they overlap.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from vos.propagate import PALETTE, imwrite_indexed  # noqa: E402

CUSTOM = os.path.join(ROOT, "data", "custom")
CKPT = os.path.join(ROOT, "checkpoints", "sam2.1_hiera_small.pt")
CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SEQ_LIST = os.path.join("ImageSets", "custom.txt")


def frames_dir(root, seq):
    return os.path.join(root, "JPEGImages", "480p", seq)


def ann_dir(root, seq, kind="Annotations"):
    return os.path.join(root, kind, "480p", seq)


# --------------------------------------------------------------------------- extract
def probe_size(ffmpeg, video):
    """Displayed (w, h) of the first frame, after the phone's rotation metadata is applied."""
    out = subprocess.run([ffmpeg, "-v", "error", "-i", video, "-frames:v", "1", "-f", "image2pipe",
                          "-vcodec", "png", "-"], capture_output=True)
    img = cv2.imdecode(np.frombuffer(out.stdout, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"could not decode {video}:\n{out.stderr.decode(errors='ignore')[-500:]}")
    return img.shape[1], img.shape[0]


def cmd_extract(a):
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    name = a.name or os.path.splitext(os.path.basename(a.video))[0]
    out = frames_dir(a.root, name)
    if os.path.exists(out) and os.listdir(out):
        if not a.overwrite:
            sys.exit(f"{out} already has frames; pass --overwrite to replace them")
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    w, h = probe_size(ffmpeg, a.video)
    if h > w:
        print(f"WARNING: portrait video ({w}x{h}). DAVIS is landscape; the clip will be "
              f"{a.short}px wide and taller than DAVIS frames. Consider re-filming in landscape.")
    scale = f"scale=-2:{a.short}" if w >= h else f"scale={a.short}:-2"
    vf = (f"fps={a.fps}," if a.fps else "") + scale
    cmd = [ffmpeg, "-v", "error", "-y"]
    if a.start:
        cmd += ["-ss", str(a.start)]
    cmd += ["-i", a.video]
    if a.duration:
        cmd += ["-t", str(a.duration)]
    if not a.fps:
        # some phone clips have a 120 fps timebase; without passthrough ffmpeg duplicates frames
        cmd += ["-fps_mode", "passthrough"]
    cmd += ["-vf", vf, "-an", "-q:v", "2", "-start_number", "0", os.path.join(out, "%05d.jpg")]
    subprocess.run(cmd, check=True)

    n = len(glob.glob(os.path.join(out, "*.jpg")))
    fw, fh = Image.open(os.path.join(out, "00000.jpg")).size
    print(f"{name}: {n} frames of {fw}x{fh} -> {out}")
    if not 70 <= n <= 150:
        print(f"note: {n} frames; aim for roughly 70-150 (use --start/--duration/--fps)")
    lst = os.path.join(a.root, SEQ_LIST)
    os.makedirs(os.path.dirname(lst), exist_ok=True)
    seqs = [s.strip() for s in open(lst)] if os.path.exists(lst) else []
    if name not in seqs:
        with open(lst, "a") as f:
            f.write(name + "\n")


# --------------------------------------------------------------------------- annotate
def load_image_predictor():
    """SAM 2 image predictor. Set SAM2_DEVICE=cpu to keep the GPU free (e.g. while a timed
    DAVIS run is going; CPU takes a few seconds per click)."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not os.path.exists(CKPT):
        sys.exit(f"missing SAM 2 checkpoint {CKPT}")
    dev = os.environ.get("SAM2_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    return SAM2ImagePredictor(build_sam2(CFG, CKPT, device=dev))


def colorize(img_bgr, label, objs_pts, cur):
    vis = img_bgr.copy()
    for k in np.unique(label):
        if k == 0:
            continue
        col = PALETTE[k][::-1].astype(np.float32)  # RGB -> BGR
        m = label == k
        vis[m] = (0.5 * vis[m] + 0.5 * col).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, tuple(int(c) for c in col), 2)
    for k, pts in objs_pts.items():
        for (x, y, lab) in pts:
            c = (0, 255, 0) if lab == 1 else (0, 0, 255)
            cv2.circle(vis, (x, y), 5, c, -1)
            cv2.circle(vis, (x, y), 5, (255, 255, 255), 1)
    return vis


def annotate_frame(pred, img_path, out_path, win, n_prev_objects=0, save_on_quit=False):
    """Returns 'next', 'prev', 'quit' after the user leaves the frame.
    save_on_quit: save unsaved clicks when the user quits instead of discarding them."""
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    masks = {}       # obj id -> bool mask
    pts = {}         # obj id -> [(x, y, label)]
    if os.path.exists(out_path):
        prev = np.array(Image.open(out_path))
        for k in np.unique(prev):
            if k:
                masks[int(k)], pts[int(k)] = prev == k, []
    cur = [max(masks, default=0) + 1 if not masks else 1]
    state = {"dirty": False}

    def label_map():
        lab = np.zeros((H, W), np.uint8)
        for k in sorted(masks):
            lab[masks[k]] = k
        return lab

    def run_sam(k):
        p = pts.get(k, [])
        if not p:
            masks.pop(k, None)
            return
        xy = np.array([[x, y] for x, y, _ in p], np.float32)
        lb = np.array([l for _, _, l in p], np.int32)
        multi = len(p) == 1
        m, s, _ = pred.predict(point_coords=xy, point_labels=lb, multimask_output=multi)
        masks[k] = m[int(np.argmax(s))] > 0 if multi else m[0] > 0

    def on_mouse(ev, x, y, flags, _):
        if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            k = cur[0]
            pts.setdefault(k, []).append((x, y, 1 if ev == cv2.EVENT_LBUTTONDOWN else 0))
            run_sam(k)
            state["dirty"] = True

    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = colorize(img, label_map(), pts, cur[0])
        txt = (f"{os.path.basename(img_path)}  object {cur[0]}  "
               f"[L+ R- n:next obj u:undo r:clear s:save [ ]:frame q:quit]"
               + ("  *unsaved*" if state["dirty"] else ""))
        cv2.rectangle(vis, (0, 0), (W, 22), (0, 0, 0), -1)
        col = tuple(int(c) for c in PALETTE[cur[0]][::-1])
        cv2.putText(vis, txt, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col if any(col) else (255, 255, 255), 1)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            key = ord("q")
        if key == ord("n"):
            cur[0] = max(list(masks) + [cur[0]]) + 1
        elif key == ord("u") and pts.get(cur[0]):
            pts[cur[0]].pop(); run_sam(cur[0]); state["dirty"] = True
        elif key == ord("r"):
            pts[cur[0]] = []; masks.pop(cur[0], None); state["dirty"] = True
        elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6"), ord("7"), ord("8"), ord("9")):
            cur[0] = key - ord("0")
        elif key == ord("s"):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            imwrite_indexed(out_path, label_map())
            state["dirty"] = False
            print(f"saved {out_path} with objects {sorted(k for k in masks if masks[k].any())}")
        elif key in (ord("]"), ord("[")):
            if state["dirty"]:
                print("unsaved changes: press s first (or r to clear)")
                continue
            return "next" if key == ord("]") else "prev"
        elif key in (ord("q"), 27):
            if state["dirty"] and save_on_quit and any(m.any() for m in masks.values()):
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                imwrite_indexed(out_path, label_map())
                print(f"saved {out_path}")
            elif state["dirty"]:
                print("unsaved changes discarded")
            return "quit"


def cmd_annotate(a):
    fdir = frames_dir(a.root, a.seq)
    frames = sorted(glob.glob(os.path.join(fdir, "*.jpg")))
    if not frames:
        sys.exit(f"no frames in {fdir}; run extract first")
    idx = a.frames if a.frames else (list(range(0, len(frames), a.every)) if a.every else [0])
    pred = load_image_predictor()
    win = f"annotate {a.seq}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    i = 0
    while 0 <= i < len(idx):
        f = frames[idx[i]]
        out = os.path.join(ann_dir(a.root, a.seq), os.path.basename(f).replace(".jpg", ".png"))
        r = annotate_frame(pred, f, out, win)
        if r == "quit":
            break
        i += 1 if r == "next" else -1
        i = max(i, 0)
    cv2.destroyAllWindows()


def cmd_points(a):
    """Non-interactive annotation: --obj "+x,y +x,y -x,y" once per object (+ positive, - negative).
    Same SAM 2 predictor as `annotate`; writes the mask and a preview overlay."""
    fdir = frames_dir(a.root, a.seq)
    f = sorted(glob.glob(os.path.join(fdir, "*.jpg")))[a.frame]
    img = cv2.imread(f)
    pred = load_image_predictor()
    pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    lab = np.zeros(img.shape[:2], np.uint8)
    all_pts = {}
    for k, spec in enumerate(a.obj, start=1):
        p = []
        for tok in spec.split():
            sign, (x, y) = tok[0], tok[1:].split(",")
            p.append((int(float(x)), int(float(y)), 0 if sign == "-" else 1))
        xy = np.array([[x, y] for x, y, _ in p], np.float32)
        lb = np.array([l for _, _, l in p], np.int32)
        multi = len(p) == 1
        m, s, _ = pred.predict(point_coords=xy, point_labels=lb, multimask_output=multi)
        mk = m[int(np.argmax(s))] > 0 if multi else m[0] > 0
        lab[mk] = k
        all_pts[k] = p
        print(f"object {k}: {mk.sum()} px, score {float(np.max(s)):.3f}")
    out = os.path.join(ann_dir(a.root, a.seq), os.path.basename(f).replace(".jpg", ".png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imwrite_indexed(out, lab)
    prev = a.preview or os.path.join(a.root, "previews", f"{a.seq}_{a.frame:05d}.jpg")
    os.makedirs(os.path.dirname(prev), exist_ok=True)
    cv2.imwrite(prev, colorize(img, lab, all_pts, 1))
    print(f"saved {out}; preview {prev}")


# --------------------------------------------------------------------------- propagate
def parse_points(s):
    """'+x,y -x,y' -> ([[x, y], ...], [1, 0, ...])"""
    xy, lb = [], []
    for tok in s.split():
        x, y = tok[1:].split(",")
        xy.append([float(x), float(y)])
        lb.append(0 if tok[0] == "-" else 1)
    return xy, lb


def cmd_propagate(a):
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    if a.all:
        lst = a.seq_list or os.path.join(a.root, "ImageSets", "2017", "val.txt")
        seqs = [s.strip() for s in open(lst) if s.strip()]
    else:
        seqs = [a.seq]
    pred = build_sam2_video_predictor(CFG, CKPT, device="cuda")
    for seq in seqs:
        fdir = frames_dir(a.root, seq)
        first = os.path.join(ann_dir(a.root, seq), "00000.png")
        if not os.path.exists(first):
            print(f"{seq}: no {first}, skipping"); continue
        out = os.path.join(a.out, seq) if a.out else ann_dir(a.root, seq, "Annotations_sam2")
        os.makedirs(out, exist_ok=True)
        seg0 = np.array(Image.open(first))
        ids = [int(k) for k in np.unique(seg0) if k not in (0, 255)]
        n = len(glob.glob(os.path.join(fdir, "*.jpg")))
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
        with torch.inference_mode():
            st = pred.init_state(video_path=fdir, offload_video_to_cpu=True)
            for k in ids:
                pred.add_new_mask(st, frame_idx=0, obj_id=k, mask=seg0 == k)
            for fx in a.fix or []:
                # correction clicks on a later frame: "FRAME:OBJ:+x,y -x,y ..."
                fr, obj, pts = fx.split(":", 2)
                xy, lb = parse_points(pts)
                pred.add_new_points_or_box(st, frame_idx=int(fr), obj_id=int(obj),
                                           points=np.array(xy, np.float32),
                                           labels=np.array(lb, np.int32))
                print(f"{seq}: correction on frame {fr}, object {obj}: {pts}")
            for fi, obj_ids, logits in pred.propagate_in_video(st):
                lg = logits[:, 0].float().cpu().numpy()          # N x H x W
                lab = np.zeros(lg.shape[1:], np.uint8)
                best = lg.argmax(0)
                fg = lg.max(0) > 0
                lab[fg] = np.array(obj_ids, np.uint8)[best[fg]]
                if fi == 0:
                    lab = seg0.astype(np.uint8)
                imwrite_indexed(os.path.join(out, f"{fi:05d}.png"), lab)
            pred.reset_state(st)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        print(f"{seq}: {n} frames, {dt / n:.3f} s/frame, peak "
              f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MB -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("video")
    e.add_argument("--name", help="sequence name (default: video file name)")
    e.add_argument("--start", type=float, help="trim start, seconds")
    e.add_argument("--duration", type=float, help="trim length, seconds")
    e.add_argument("--fps", type=float, help="reduce frame rate, e.g. 15")
    e.add_argument("--short", type=int, default=480, help="short side in px")
    e.add_argument("--overwrite", action="store_true")
    n = sub.add_parser("annotate")
    n.add_argument("seq")
    n.add_argument("--frames", type=int, nargs="*", help="frame indices to annotate (default 0)")
    n.add_argument("--every", type=int, help="annotate every k-th frame")
    pt = sub.add_parser("points", help="annotate from given click coordinates (no window)")
    pt.add_argument("seq")
    pt.add_argument("--obj", action="append", required=True,
                    help='one per object, e.g. --obj "+420,200 +430,260 -500,90"')
    pt.add_argument("--frame", type=int, default=0)
    pt.add_argument("--preview", help="preview image path (default <root>/previews/)")
    p = sub.add_parser("propagate")
    p.add_argument("seq", nargs="?")
    p.add_argument("--all", action="store_true", help="all sequences in the list")
    p.add_argument("--seq_list")
    p.add_argument("--out", help="write here (results format) instead of Annotations_sam2/")
    p.add_argument("--fix", action="append",
                   help='correction clicks on a later frame, "FRAME:OBJ:+x,y -x,y" (repeatable)')
    for s in (e, n, pt, p):
        s.add_argument("--root", default=CUSTOM)
    a = ap.parse_args()
    {"extract": cmd_extract, "annotate": cmd_annotate, "points": cmd_points,
     "propagate": cmd_propagate}[a.cmd](a)


if __name__ == "__main__":
    main()
