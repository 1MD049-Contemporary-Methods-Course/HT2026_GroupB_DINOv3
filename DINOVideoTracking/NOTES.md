# dino-vos: working notes

## Where we are (night of 16 Nov)

**Done:** step 1 (validation), step 2 (all 7 models load), feature cache, all of P0, and P1 for
v3-S and v3-S+.

**Stopped for the night** (PC too loud). v3-B/16 1x finished and is saved (J&F 69.72). 1.5x was
stopped part-way, so `results/v3-b16_1.5x/` is incomplete and has no summary row. Resume with
`.venv\Scripts\python run_davis.py --model v3-b16 --scale 1.5 2`, which overwrites the partial
folder.

DAVIS 2017 val J&F so far. The model card column is DINOv3's reported value at 960 px.

| Model | 1x | 1.5x | 2x | Model card |
|---|---|---|---|---|
| v1-S/16 | 61.78 | | 68.24 | |
| v2r-S/14 | 61.40 | | 71.61 | |
| v3-S/16 | 67.93 | 72.82 | 71.83 | 72.7 |
| v3-S+/16 | 68.94 | 74.31 | 74.59 | 75.5 |
| v3-B/16 | 69.72 | 75.60 | 76.34 | 77.2 |
| v3-L/16 | 72.02 | not run (time) | not run (time) | 79.9 |

**Decision, 17 Nov:** v3-L at 1.5x / 2x on DAVIS is dropped because it would take too long (about
1 h). None of the take-home messages depend on it: the equal-compute comparison only needs L at 1x,
and on our own clips L is complete at 1x and 2x. If it is ever wanted:
`.venv\Scripts\python run_davis.py --model v3-l16 --scale 1.5 2`.

Findings so far:
- At 1x, v3 is about 6 J&F ahead of v1 and v2. At 2x, v2 almost catches up (71.6 against 71.8).
- v3-S peaks at 1.5x. The 2x mean drops because a few objects collapse: lab-coat_1 −40,
  shooting_1 −34, motocross-jump_1 −14, breakdance_1 −13. Most objects still improve (30 of 61).
  The same sequences fail for S+ (lab-coat_1 −38, motocross-jump_1 −24). These are good
  failure-case overlay videos.
- v1 at 2x needs 3.7 GB and 0.76 s/frame. v2/v3-S at 2x need about 1.1 GB and 0.54 s/frame (SDPA).

**Next (in order):**
1. Finish v3-B at 1.5x and 2x (command above, about 30–40 min), then run v3-L:
   `.venv\Scripts\python run_davis.py --model v3-l16 --scale 1 1.5 2`. This will take about an hour.
   If 2x runs out of memory, the cell is skipped automatically.
2. `python make_table.py` prints the final table.
3. Overlay video and side-by-side tool. Start with lab-coat, shooting and motocross-jump at 1.5x
   against 2x.
4. Step 4: video prep and SAM 2 annotation tool for our own clips, then the twin test.
5. P2: layer sweep (uses the 1x cache), v1-S/8, and a SAM 2 reference.

Hard deadline 18 Nov; experiments frozen by midday 17 Nov. The group meeting to approve the
project is on 16 Nov, and step 1 serves as its demo.

## Step 1: validation (done 16 Nov)

DINO v1 ViT-S/16 at 1x on DAVIS 2017 val (semi-supervised, 30 seqs), torch 2.10 + cu130, fp32:
**J&F 61.78, J 60.18, F 63.38.** Published: 61.8 / 60.2 / 63.4. 0.067 s/frame, peak 366 MB.

Settings are the original script's defaults (read from `third_party/dino/eval_video_segmentation.py`):
frame 0 + 7 previous frames as context, top-5, temperature 0.1, window radius 12 patches,
short side 480 with the long side floored to a multiple of 64. Base grid for 480x854 is 30x52; for
the 480x910 sequences it is 30x56.

## Differences from the original script

- Propagation processes queries in chunks and builds the window mask from grid offsets. The original
  uses a dense (ncontext, hw, hw) mask. Checked equal to 2e-7 in `tests/test_equivalence.py`.
- The number of label channels comes from the full-resolution frame-0 mask instead of the
  downsampled one. The result is the same, because a missing channel is never predicted either way.
- v1 keeps the script's normalisation std (0.228, 0.224, 0.225). The ImageNet value is 0.229. v2/v3
  use their own processors' mean/std.
- At every model and scale, soft labels are upsampled bilinearly to the script's processing size
  (base grid x 16, e.g. 480x832) and then resized nearest to the original frame. For v1 at 1x this
  is exactly the original.
- The window radius scales with the grid: 12 / 18 / 24 at 1x / 1.5x / 2x.

## Patches to the DAVIS toolkit (third_party/davis2017-evaluation)

- `davis2017/utils.py`: `np.int` changed to `np.int32` (NumPy 2).
- `davis2017/davis.py`: frame IDs used `path.split('/')`. On Windows this made the evaluator score GT
  against GT (J&F = 100). Now uses `os.path.basename`. **Any other group member on Windows using the
  unpatched toolkit will get wrong numbers.**

## Custom clips (17 Nov)

Raw phone videos are in `data/raw_videos/`. `clip_tool.py extract` wrote ten clips to `data/custom/`,
listed in `ImageSets/custom.txt`. `1000135581` was skipped because it is only 1.2 s.

| Clip | Source | Frames | Orientation | Objects | Notes |
|---|---|---|---|---|---|
| hat-cup | 565 | 170 | landscape | hat, cup | cup passes behind the hat (full occlusion) |
| cup-slide | 566 | 97 | landscape | cup | simple translation |
| cup-bottle | 576 | 99 | landscape | cup, bottle | bottle passes in front of the cup |
| twin-mugs | 580 | 135 | landscape | mug L, mug R | **twin test**, 3.0–7.5 s; full crossing around frame 60 |
| person-dance | 577 | 182 | portrait | person | **face visible** |
| person-walk | 578 | 156 | portrait | person | walks into the camera; **face visible** |
| person-approach | 579 | 143 | portrait | person | close-up, turns away; **face visible** |
| mug-swing | 582 | 189 | portrait | mug | hanging, camera moves; first 6.3 s |
| spoon-drag | 583 | 162 | portrait | spoon | dragged on string |
| spoon-long | 584 | 181 | portrait | spoon | 10 fps; hand shadows, nearly leaves the frame |

- Seven clips are portrait because of the phone's rotation metadata. They are kept as portrait:
  480x854, base grid 52x30, the same token count as DAVIS.
- Two videos had a 120 fps timebase, and ffmpeg duplicated frames. Fixed with `-fps_mode passthrough`.
- Claude placed the first-frame masks with scripted SAM 2 points (`clip_tool.py points`, run on CPU)
  and checked each one. Previews are in `data/custom/previews/`. Fix any of them with
  `clip_tool.py annotate <seq>`.
- SAM 2.1-S pseudo ground truth for every frame is in `Annotations_sam2/`. It was spot-checked:
  identities are correct in twin-mugs (red = left mug, green = right mug), and occlusions in hat-cup
  and cup-bottle are handled. SAM 2 runs at about 0.35–0.55 s/frame and below 1 GB.
- **The twin mugs are not perfect twins in the image.** The right mug is closer, so it looks bigger,
  and the handles face different ways. Appearance can therefore partly separate them. Mention this
  in the talk.
- **Faces:** the three person clips must not go into shared slides unless the faces are blurred.
  `make_video.py --blur_faces` does this with YuNet, but needs
  `checkpoints/face_detection_yunet_2023mar.onnx` from the OpenCV zoo (not downloaded yet; needs
  the user's OK).

### Custom-clip results (17 Nov). **Priority: custom clips first, DAVIS second (user decision).**

Scored with `eval_custom.py` against SAM 2 pseudo-GT on every frame; frame 0 is excluded. Output:
`results_custom/custom_scores.csv`. Mean mIoU over the 10 clips:

| Model | 1x | 2x |
|---|---|---|
| v1-S/16 | 75.1 | 82.4 |
| v2r-S/14 | 69.1 | 80.5 |
| v3-S/16 | 75.6 | 82.6 |

On our clips the generations are much closer than on DAVIS. 2x helps every model.

**Full table over all 11 clips, monstera included** (mean mIoU, SAM 2 reference):

| Model | 1x | 2x |
|---|---|---|
| v1-S/16 | 72.4 | 79.8 |
| v2r-S/14 | 67.8 | 78.3 |
| v3-S/16 | 73.9 | 81.0 |
| v3-S+/16 | 74.7 | 82.3 |
| v3-B/16 | 74.7 | 81.8 |
| v3-L/16 | 74.7 | **82.8** |

(Updated after the monstera reference fix, see below; before it, every value was 0.0–0.2 lower.)

- **On our clips, resolution matters much more than model size.** 1x → 2x gives +7 to +10 for
  every model. S → L gives only +0.8 at 1x and +1.8 at 2x. On DAVIS, S → L gives +4.1 at 1x.
- **Model size does not fix the twin test.** twin-mugs id_confusion is 28–29% for every v3 size at
  both scales.
- **Equal compute, from the DAVIS timings:** v3-S at 1.5x (0.18 s/frame) scores 72.8 J&F. v3-L at
  1x (0.31 s/frame) scores 72.0. So spending compute on tokens beats spending it on parameters.
- Spoon clips gain most from 2x (spoon-drag 58 → 85 for v3-S). They have thin handles and wires,
  where 16 px patches are too coarse.

**Twin test (twin-mugs).** All models are perfect until the crossing. The rear mug is fully hidden
for about 3 frames (57–59). **After that, every DINO model labels the reappearing rear mug as the
front mug: 90–100% of its pixels get the front mug's label.** Its own label survives only on scraps
of background. v3-S also spreads both labels along the string tied to the mugs. SAM 2 keeps both
identities.
- The swap metric from the handoff misses this "merge", because one label covers both mugs.
  `eval_custom.py` therefore also reports `id_confusion`: the share of each object's pixels
  predicted as the other object, on frames where both are visible. On twin-mugs:
  v1 25–27%, v2 28–29%, v3 28–29%, at both 1x and 2x.
- **Ablation: no search window (`--radius 0`, global matching, 1x)** in `results_custom_r0/`.
  twin-mugs is unchanged: identical id_confusion for every model. So the failure is not caused by the
  window. The rear mug's features match the front mug's recent-frame features better than its own
  frame-0 features. Last-layer features from all generations, including RoPE v3, do not separate
  identical objects by position.
- The same ablation on hat-cup: the cup reappears on the far side of the hat, outside the window.
  Global matching fixes this (mIoU 63 → 73 / 74 / **80** for v1 / v2 / v3, confusion 19 → 4 for
  v3). v3 gains most. Video: `videos/custom_hat-cup__v3S_window_vs_global.mp4`.
- This is an ablation applied identically to all models, not a per-model tuning. The main table
  keeps the original settings.

Videos in `videos/`:
- `custom_<clip>__generations_1x.mp4`: v1 / v2 / v3-S / SAM 2, one per clip.
- `twin-mugs__generations_1x.mp4` and `twin-mugs__v3_1x_vs_2x.mp4`.
- DAVIS clips are in `davis_*.mp4`.

**The "smeared red mug" in v3-S 1x on twin-mugs is a disocclusion trail** (diagnosed 17 Nov,
outputs in `results_ablate/`):
- It is a band at the mug's height along the path the mug has travelled, from its frame-0 position
  at the left edge to wherever the mug is now. Stray foreground is about 45k px/frame, against
  0.6–1.4k for v1, v2 and v3-S 2x. It starts at frame 8, behind the moving mug, and only grows.
- Mechanism: patches at the mug's rear edge mix mug and wall (16 px patches at 1x) and get a partial
  mug label. Once the mug has moved on, each patch is pure static wall, but it matches its own copy
  from the previous frames best and inherits that copy's label. The 7-frame memory keeps the error
  alive.
- Ablations on v3-S 1x, twin-mugs:
  - **Reflect padding by 2 patches** (`--pad 2`): stray 45k → 38k, mIoU 42.1 → 43.3. The image
    border is not the cause.
  - **No memory** (`--n_last_frames 0`): stray falls to 2.3k (−95%), which confirms the
    self-reinforcement. But tracking collapses elsewhere (cup-slide 82 → 42), so this is not a fix.
  - **2x resolution** removes it (644 px/frame). This fits the main take-home: resolution matters
    most.
- Earlier in these notes I blamed the string. That was wrong for the red trail. The green streak on
  the right does follow the string.
- `--pad` stays in the code as an ablation option (default 0; it disables the cache).

**Monstera (second batch, 17 Nov).** The clip is 136 frames; the plant is turned about half a
revolution. The source video is only 744x418 because the online cutter downscaled it, so the frames
are upscaled a little. Next time trim on the phone or with `clip_tool.py --start/--duration`.
Three leaves are tracked: 1 = big dark leaf top-left, 2 = young split leaf, 3 = big dark leaf
top-right. Leaves 1 and 3 are near-twins. The SAM 2 reference is plausible throughout (consistent
with the rotation) but least certain in frames 60–90, where all three leaves overlap.

**Reference fix (17 Nov evening).** The user spotted that SAM 2 dropped part of the red leaf (1) at
the end. From frame ~105 SAM 2 labelled only the outer half of leaf 1; the inner half, next to the
young leaf, was background (up to ~1.8k px/frame). DINO covered it. Fixed by re-running SAM 2 with
correction clicks on leaf 1 at frames 105/115/125/135 (`clip_tool.py propagate monstera --fix
"105:1:+558,109 +642,108 -484,116" --fix "115:1:+586,121 +696,134 -494,117" --fix
"125:1:+587,136 +694,149 -505,121" --fix "135:1:+584,133 +691,144 -504,120"`). The old reference is
in `data/custom/Annotations_sam2_orig/`, the old scores in
`results_custom/custom_scores_before_monstera_fix.csv`. Videos and GIFs were re-rendered.
Checked by eye: the big light-green leaf with slits at the end is the **young leaf (2)**, not leaf 1.
In frames 44–53 it is visible opening behind the dark leaf 1, which has a single oval hole. So SAM 2
labels it correctly, and the DINO models are wrong to call it leaf 1.

| Run | mIoU | Leaf 1 | Leaf 2 | Leaf 3 | id_confusion |
|---|---|---|---|---|---|
| v1-S 1x / 2x | 44.7 / 53.4 | 37.2 / 41.4 | 36.6 / 48.1 | 60.3 / 70.7 | 29.5 / 28.0 |
| v2r-S 1x / 2x | 54.7 / 57.1 | 53.2 / 56.0 | 44.7 / 36.6 | 66.1 / 78.8 | 12.9 / 18.6 |
| v3-S 1x / 2x | 57.4 / **65.2** | 54.6 / 60.4 | 39.5 / 46.6 | 78.1 / **88.5** | 17.0 / 15.7 |
| v3-L 2x | 64.7 | 61.4 | 44.6 | 88.3 | 15.9 |

The fix changes the 11-clip means by only +0.1 (v3-L 2x: 82.8).

- On this clip the generations line up as on DAVIS: v1 < v2 < v3, and 2x helps.
- **Near-twin leaves 1 and 3:** in frames 60–135, v1 labels 56% of leaf 1 as leaf 3 (70% at 2x). v2
  does this for 9% (3% at 2x) and v3 for **0%**. Newer features keep look-alike *but not
  identical* parts apart. This does not contradict the mugs: the leaves differ in shape and pose,
  while the mugs are identical.
- **Young leaf 2:** every model hands 52–86% of it to leaf 1 in frames 60–135. This is a real DINO
  error, confirmed by eye: as the young leaf unfolds from behind leaf 1, its new parts match leaf 1
  (the neighbour it was hidden behind) instead of its own curled frame-0 appearance.
- Videos: `custom_monstera__generations_1x.mp4` (2x2 grid with SAM 2) and
  `custom_monstera__v3S_1x_vs_2x.mp4`.
- The monstera timings were measured while the GPU was shared with the v3-L custom run, so they are
  not valid for speed comparisons.

**person-approach-boomerang (17 Nov).** Frames 0–71 (0–2.4 s) of person-approach are played forward,
then in reverse back to frame 0, giving 143 frames. It imitates a zoom in and out. The frame order
is in `data/custom/person-approach-boomerang_source_frames.txt`. The first-frame mask is copied from
person-approach. SAM 2 is the reference and is itself symmetric (99.8). Runs: v1, v2, v3-S at 1x and
v3 S, B, L at 2x, with no S+ (user decision).

| Run | IoU zoom-in | IoU zoom-out | In/out agreement on identical frames |
|---|---|---|---|
| v1-S 1x | 93.2 | 91.7 | 92.6 |
| v2r-S 1x | 92.6 | 90.6 | 93.9 |
| v3-S 1x | 94.4 | 91.8 | 93.7 |
| v3-S 2x | 96.5 | **78.8** | 80.1 |
| v3-B 2x | 96.5 | 88.2 | 89.8 |
| v3-L 2x | **96.9** | **94.1** | **95.3** |

- On the way out, every error is *extra* mask, i.e. spill onto background. Wall that the close-up
  covered keeps the person label as the person shrinks. This is the same uncovered-background effect
  as the twin-mugs trail.
- At 2x the effect shrinks with model size: S drops to 67 IoU around frames 110–125, B to 80,
  L stays at 86 or above. The 1x models drift only 2–3 points.
- So here, unlike in the size sweep, **model size matters**. The larger model recovers from the
  spill, and S at 2x is worse than S at 1x on the way out.
- Videos: `custom_person-approach-boomerang__generations_1x.mp4` and `__v3_sizes_2x.mp4`.
  Frame 118 of the sizes video shows the spill clearly.

Face blur: **skipped by user decision.** The person clips show faces, so the group decides whether
they go on slides.

## Step 2: feature extraction (16 Nov)

All 7 models load and pass `tests/check_models.py`. Each one gives a 30x52 / 60x104 grid at 1x / 2x.
Features are unit-norm, CLS + 4 registers are dropped for v2/v3, and `--layer <depth>` gives the same
features as the default. Peak memory for the forward pass at 2x: v1-S 1.9 GB (no SDPA), v3-B 0.6 GB,
v3-L 1.5 GB. The stale user-level `HF_TOKEN` env var is ignored in `vos/features.py`. The valid login
from `hf auth login` is used instead.

## To do later

- **ViT-L (v3-l16):** at first postponed to the custom-video stage. Then, because the weights had
  already downloaded, it was moved back into the P1 DAVIS sweep (1x / 1.5x / 2x). Still to do when
  the videos arrive: run it on the custom clips (step 4 / twin test).

## Feature cache

`vos/cache.py` caches features as fp16 under `%LOCALAPPDATA%\dino-vos-cache`, which is outside
OneDrive. By default only 1x is cached (`--cache_scales`), because the layer sweep and propagation
variants run at 1x. v1-S at 1x takes 2.3 GB, and caching all of P0+P1 would take about 140 GB.
Computed features are rounded to fp16 before use, so cached and uncached runs give identical
results. v1-S/16 1x with fp16 features scores 61.78 / 60.18 / 63.38, exactly as fp32 did.
The `features` column in `summary.csv` records whether a row's features were computed or loaded from
the cache. **Only use `sec_per_frame` from rows marked computed.**

## Layout

- `vos/features.py`: `FeatureExtractor(name, layer=None).extract(img) -> (h, w, C)`, L2-normalised,
  CLS and register tokens dropped
- `vos/propagate.py`: propagation and I/O
- `run_davis.py --model <name> --scale 1 1.5 2 [--layer k] [--seqs ...]` writes
  `results/<model>_<scale>/`, `results/summary.csv` and `results/per_sequence.csv`
- `.venv` reuses the system torch (`--system-site-packages`)
