"""Track objects in your own video with frozen DINO features. No training.

  python track.py                       # pick a video in a file dialog
  python track.py my_clip.mp4 --model v3-s16 --scale 1.5

Steps:
  1. pick a video (file dialog, or pass the path)
  2. the first frame opens: click on the objects you want to track (SAM 2 turns clicks into masks)
       left click = this is the object      right click = this is NOT the object
       n = next object    u = undo click    r = clear object    q = done (saves)
  3. check the first-frame masks: Enter / y = continue,  e = edit again,  q = quit
  4. check them on the whole video: SAM 2 tracks your masks through the clip and a player opens
       space = play / pause,  a / d or arrow keys = step,  slider = jump
       Enter / y = run DINO,  e = back to the clicks,  q = quit
  5. DINO propagates the masks through the video; you get an MP4 and a GIF in outputs/<name>/

Options worth knowing:
  --model    v3-s16 (default), v3-b16, v3-l16, v2r-s14, v1-s16 ...  (see vos/features.py)
  --scale    input resolution: 1 = 480 px short side, 1.5 = 720 px, 2 = 960 px. More pixels
             helps more than a bigger model, but costs time and memory.
  --start / --duration / --fps   trim the video before tracking (seconds / frame rate)
  --mask     skip clicking and use an existing first-frame mask (indexed PNG, 0 = background)
  --no_preview   skip step 4 (saves ~0.5 s per frame)
  --yes      do not ask for confirmation (with --mask: fully non-interactive)
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from vos.features import MODELS  # noqa: E402

VIDEO_TYPES = [("Videos", "*.mp4 *.mov *.avi *.mkv *.m4v *.webm"), ("All files", "*.*")]


# --------------------------------------------------------------------------- 1. video
def pick_video():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title="Pick a video to track", filetypes=VIDEO_TYPES,
                                          initialdir=os.path.join(ROOT, "data", "raw_videos")
                                          if os.path.isdir(os.path.join(ROOT, "data", "raw_videos"))
                                          else os.path.expanduser("~"))
        root.destroy()
    except Exception:
        path = input("Path to the video: ").strip().strip('"')
    if not path:
        sys.exit("no video chosen")
    return path


def clear_dir(d, tries=10):
    """Empty a folder but keep it. Deleting the folder itself fails on Windows while OneDrive or
    Explorer has it open, so remove the files and retry briefly if one is locked."""
    import time
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        p = os.path.join(d, f)
        for i in range(tries):
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
                break
            except PermissionError:
                if i == tries - 1:
                    sys.exit(f"{p} is locked (OneDrive syncing or open in another program); "
                             "close it or use --name to write to a new folder")
                time.sleep(0.5)


def extract_frames(video, out_dir, short=480, start=None, duration=None, fps=None):
    """Frames with the short side at `short` px, numbered 00000.jpg, ... Returns (paths, fps)."""
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    reader = imageio_ffmpeg.read_frames(video)
    meta = next(reader)
    reader.close()
    src_fps = fps or meta.get("fps") or 30
    # displayed size after rotation metadata: decode one frame
    probe = subprocess.run([ffmpeg, "-v", "error", "-i", video, "-frames:v", "1", "-f", "image2pipe",
                            "-vcodec", "png", "-"], capture_output=True)
    first = cv2.imdecode(np.frombuffer(probe.stdout, np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        sys.exit(f"could not read {video}")
    h, w = first.shape[:2]
    scale = f"scale=-2:{short}" if w >= h else f"scale={short}:-2"
    clear_dir(out_dir)
    cmd = [ffmpeg, "-v", "error", "-y"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", video]
    if duration:
        cmd += ["-t", str(duration)]
    if not fps:
        cmd += ["-fps_mode", "passthrough"]  # phone clips with a 120 fps timebase duplicate frames otherwise
    cmd += ["-vf", (f"fps={fps}," if fps else "") + scale, "-an", "-q:v", "2", "-start_number", "0",
            os.path.join(out_dir, "%05d.jpg")]
    subprocess.run(cmd, check=True)
    frames = sorted(glob.glob(os.path.join(out_dir, "*.jpg")))
    if len(frames) < 2:
        sys.exit("the video gave fewer than 2 frames")
    return frames, float(src_fps)


# --------------------------------------------------------------------------- 2-3. masks
def click_masks(first_frame, mask_path):
    """Opens the SAM 2 click window on the first frame (same tool as clip_tool.py annotate)."""
    from clip_tool import load_image_predictor, annotate_frame
    print("\nLoading SAM 2 ...")
    pred = load_image_predictor()
    win = "Click the objects  (L = object, R = not object, n = next object, s = save, q = done)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    annotate_frame(pred, first_frame, mask_path, win, save_on_quit=True)
    cv2.destroyAllWindows()
    del pred
    _free_gpu()


def confirm_masks(first_frame, mask_path):
    """Shows the saved masks. Returns 'run', 'edit' or 'quit'."""
    from make_video import overlay
    if not os.path.exists(mask_path):
        print("no mask saved (press s in the click window to save)")
        return "edit"
    lab = np.array(Image.open(mask_path))
    ids = [int(k) for k in np.unique(lab) if k not in (0, 255)]
    if not ids:
        print("the mask is empty")
        return "edit"
    vis = overlay(cv2.imread(first_frame), lab)
    H, W = vis.shape[:2]
    cv2.rectangle(vis, (0, 0), (W, 46), (0, 0, 0), -1)
    cv2.putText(vis, f"{len(ids)} object(s). Enter/y = continue", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, "e = edit again   q = quit", (6, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    win = "Check the masks"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, vis)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            key = ord("q")
        if key in (13, 10, ord("y")):
            ans = "run"
        elif key == ord("e"):
            ans = "edit"
        elif key in (ord("q"), 27):
            ans = "quit"
        else:
            continue
        cv2.destroyAllWindows()
        return ans


def sam2_preview(frames, mask_path, out_dir):
    """Track the first-frame masks through the whole clip with SAM 2, as a quick check of what
    the clicks selected. Writes one indexed PNG per frame to out_dir."""
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    from clip_tool import CFG, CKPT
    from vos.propagate import imwrite_indexed
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nSAM 2 preview on {dev}: tracking {len(frames)} frames "
          f"({'about 0.5 s' if dev == 'cuda' else 'several seconds'} per frame) ...")
    clear_dir(out_dir)
    seg0 = np.array(Image.open(mask_path))
    ids = [int(k) for k in np.unique(seg0) if k not in (0, 255)]
    pred = build_sam2_video_predictor(CFG, CKPT, device=dev)
    with torch.inference_mode():
        st = pred.init_state(video_path=os.path.dirname(frames[0]), offload_video_to_cpu=True)
        for k in ids:
            pred.add_new_mask(st, frame_idx=0, obj_id=k, mask=seg0 == k)
        for fi, obj_ids, logits in pred.propagate_in_video(st):
            lg = logits[:, 0].float().cpu().numpy()
            lab = np.zeros(lg.shape[1:], np.uint8)
            fg = lg.max(0) > 0
            lab[fg] = np.array(obj_ids, np.uint8)[lg.argmax(0)[fg]]
            imwrite_indexed(os.path.join(out_dir, f"{fi:05d}.png"), seg0.astype(np.uint8) if fi == 0 else lab)
    del pred, st
    _free_gpu()


def review_video(frames, mask_dir, fps, title_text="SAM 2 preview"):
    """Player for checking masks on every frame. Returns 'run', 'edit' or 'quit'.
    space = play / pause, a d or arrow keys = step, slider = jump."""
    from make_video import overlay, title
    win = f"{title_text} - check the masks on the whole video"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    pos, playing, n = [0], [True], len(frames)
    cv2.createTrackbar("frame", win, 0, n - 1, lambda v: pos.__setitem__(0, v))
    delay = max(1, int(1000 / min(fps, 30)))
    LEFT, RIGHT = (2424832, 65361, 63234), (2555904, 65363, 63235)  # Windows / Linux / macOS
    shown = -1
    while True:
        i = pos[0]
        if i != shown:
            img = cv2.imread(frames[i])
            p = os.path.join(mask_dir, os.path.basename(frames[i]).replace(".jpg", ".png"))
            vis = overlay(img, np.array(Image.open(p))) if os.path.exists(p) else img
            vis = title(vis, f"{title_text}  #{i:05d} / {n - 1}{'' if playing[0] else '  (paused)'}")
            H, W = vis.shape[:2]
            cv2.rectangle(vis, (0, H - 44), (W, H), (0, 0, 0), -1)
            cv2.putText(vis, "Enter/y = run DINO   e = edit clicks   q = quit", (6, H - 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, "space = play/pause   a/d or arrows = step", (6, H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, vis)
            if cv2.getTrackbarPos("frame", win) != i:
                cv2.setTrackbarPos("frame", win, i)
            shown = i
        key = cv2.waitKeyEx(delay if playing[0] else 30)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            key = ord("q")
        k = key & 0xFF if 0 <= key < 256 else key
        if k in (13, 10, ord("y")):
            ans = "run"
        elif k == ord("e"):
            ans = "edit"
        elif k in (ord("q"), 27):
            ans = "quit"
        else:
            if k == ord(" "):
                playing[0] = not playing[0]
                shown = -1
            elif k in (ord("a"),) + LEFT:
                playing[0], pos[0] = False, max(0, i - 1)
            elif k in (ord("d"),) + RIGHT:
                playing[0], pos[0] = False, min(n - 1, i + 1)
            elif playing[0]:
                pos[0] = (i + 1) % n
            continue
        cv2.destroyAllWindows()
        return ans


# --------------------------------------------------------------------------- 4. DINO + outputs
def _free_gpu():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_dino(frames, mask_path, out_dir, model, scale):
    import torch
    from vos.features import FeatureExtractor
    from vos.propagate import track_sequence
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {model} on {dev} ...")
    try:
        ext = FeatureExtractor(model, device=dev)
    except Exception as e:
        msg = str(e)
        if model.startswith("v3") and ("gated" in msg.lower() or "401" in msg or "403" in msg):
            sys.exit("DINOv3 weights are gated on Hugging Face: accept the licence on the model page, "
                     "run `hf auth login`, or use --model v2r-s14 (no login needed).")
        raise
    print(f"Tracking {len(frames)} frames at {scale:g}x (this is the slow part) ...")
    st = track_sequence(ext, frames, mask_path, out_dir, scale=scale)
    print(f"done: {st['sec_per_frame']:.3f} s/frame, token grid {st['grid']}")
    del ext
    _free_gpu()


def write_outputs(frames, mask_dir, base, fps, label):
    """MP4 (full size) and GIF (smaller) of the masks drawn over the video."""
    import imageio_ffmpeg
    from make_video import overlay, title
    fps = fps if fps <= 60 else 30  # slow-motion phone clips play back at 30 fps
    h0, w0 = cv2.imread(frames[0]).shape[:2]
    W, H = w0 + w0 % 2, h0 + h0 % 2
    mp4 = base + ".mp4"
    wr = imageio_ffmpeg.write_frames(mp4, (W, H), fps=fps, codec="libx264", quality=8,
                                     macro_block_size=1)
    wr.send(None)
    for f in frames:
        img = cv2.imread(f)
        p = os.path.join(mask_dir, os.path.basename(f).replace(".jpg", ".png"))
        tile = overlay(img, np.array(Image.open(p))) if os.path.exists(p) else img
        tile = title(tile, f"{label}  #{os.path.basename(f)[:-4]}")
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:h0, :w0] = tile
        wr.send(np.ascontiguousarray(canvas[:, :, ::-1]))
    wr.close()

    gif = base + ".gif"
    gif_w = min(w0, 480)
    vf = (f"fps={min(fps, 12)},scale={gif_w}:-2:flags=lanczos,split[a][b];"
          "[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", mp4, "-vf", vf, gif],
                   check=True)
    return mp4, gif


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="video file (default: pick one in a dialog)")
    ap.add_argument("--model", default="v3-s16", choices=list(MODELS))
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--name", help="output folder name (default: the video's file name)")
    ap.add_argument("--start", type=float, help="trim: start, seconds")
    ap.add_argument("--duration", type=float, help="trim: length, seconds")
    ap.add_argument("--fps", type=float, help="reduce the frame rate, e.g. 15")
    ap.add_argument("--mask", help="first-frame mask PNG to use instead of clicking")
    ap.add_argument("--yes", action="store_true", help="skip all checks (first frame and preview)")
    ap.add_argument("--no_preview", action="store_true",
                    help="skip the SAM 2 full-video preview (it takes ~0.5 s per frame on a GPU)")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--no_open", action="store_true", help="do not open the result when done")
    a = ap.parse_args()

    video = a.video or pick_video()
    if not os.path.isfile(video):
        sys.exit(f"not a file: {video}")
    name = a.name or "".join(c if c.isalnum() or c in "-_" else "_"
                             for c in os.path.splitext(os.path.basename(video))[0])[:60]
    work = os.path.join(a.out, name)
    print(f"Video: {video}\nOutput folder: {work}")

    frames, fps = extract_frames(video, os.path.join(work, "frames"), start=a.start,
                                 duration=a.duration, fps=a.fps)
    print(f"{len(frames)} frames of {Image.open(frames[0]).size[0]}x{Image.open(frames[0]).size[1]}")
    if len(frames) > 300:
        print("note: long clip; tracking quality drops over time. Consider --start/--duration.")

    mask_path = os.path.join(work, "first_frame_mask.png")
    if a.mask:
        shutil.copy(a.mask, mask_path)
    elif os.path.exists(mask_path):
        os.remove(mask_path)  # frames were re-extracted; start the clicks fresh
    preview_dir = os.path.join(work, "masks_sam2_preview")
    while True:
        if not a.mask:
            click_masks(frames[0], mask_path)
        ans = "run" if a.yes and os.path.exists(mask_path) else confirm_masks(frames[0], mask_path)
        if ans == "run" and not (a.yes or a.no_preview):
            sam2_preview(frames, mask_path, preview_dir)
            ans = review_video(frames, preview_dir, fps)
        if ans == "run":
            break
        if ans == "quit":
            sys.exit("stopped; nothing was tracked")
        a.mask = None  # 'edit': back to the click window

    mask_dir = os.path.join(work, f"masks_{a.model}_{a.scale:g}x")
    run_dino(frames, mask_path, mask_dir, a.model, a.scale)
    label = f"{a.model} {a.scale:g}x"
    mp4, gif = write_outputs(frames, mask_dir, os.path.join(work, f"{name}_{a.model}_{a.scale:g}x"),
                             fps, label)
    print(f"\nVideo: {mp4}\nGIF:   {gif}")
    if not a.no_open and sys.platform == "win32":
        os.startfile(mp4)


if __name__ == "__main__":
    main()
