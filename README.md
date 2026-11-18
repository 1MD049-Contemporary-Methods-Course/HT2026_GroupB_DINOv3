# HT2026 Group B — DINOv3

This repository contains the code contributions for our group assignment.

## Emmanouil Vettas — DINOv3 feature similarity on NuPlan street scenes

**Goal.** Explore how pretrained DINOv3 visual features respond to cars and other content in NuPlan street-camera footage. This was intentionally scoped as a small, exploratory task that could be run on a very low-end computer.

**Method.** The interactive viewer in `dino_player.py` loads a pretrained DINOv3 ViT-S/16 model, resizes each camera frame to a modest width, and extracts normalized spatial patch features. The user clicks a patch in a frame to select its feature vector. For each patch in the current frame, the viewer computes the dot product with the selected, normalized feature vector, which is cosine similarity. It visualizes the resulting similarity map across frames and camera streams. This can be used to inspect whether visual features similar to a selected car or street region appear elsewhere.

**Classifier clarification.** No separate trained classifier or classification head was used. The apparent “classification” was based on DINOv3’s extracted features and cosine similarity to the selected query patch. It is therefore a feature-similarity visualization / exploratory classification idea, rather than a conventional classifier that predicts named classes. This distinction explains the confusion during the presentation.

**Low-resource scope.** The task was designed to stay small enough for a very low-end computer: it uses the small ViT-S/16 backbone, modest image resolution, inference without gradients, and a bounded frame cache. Processing is kept interactive at a low frame rate; a GPU is optional, and CPU execution is supported.

**Relevant files.**

- `dino_player.py` — interactive video playback, DINOv3 feature extraction, query selection, cosine-similarity map.
- `interactive_dino.py` — NuPlan image path resolution and annotation-box drawing helpers used by the player.
- `dinov3/` — the upstream DINOv3 source needed by local `torch.hub` loading; see its `LICENSE.md`.
- `visualize_features.py` — standalone PCA visualization of DINOv3 patch features.
- `project_boxes.py`, `project_centers.py` — NuPlan annotation projection diagnostics.
- `inspect_annotations.py` — annotation structure and class-count inspection.
- `outputs/` — example images from tests and visual checks made during implementation, including projected annotation boxes and centers, box crops, and a PCA visualization of DINOv3 features. These are illustrative outputs from the experiments.

### Running the viewer

Install the dependencies listed in `requirements.txt` and provide the NuPlan camera files and annotation pickle locally. The pretrained `dinov3_vits16.pth` ViT-S/16 checkpoint is included in this repository. `.gitignore` excludes local dataset folders, annotation pickles, generated outputs, and Python cache files from Git; it does not ignore the model weights. The annotation pickle must be from a trusted source because Python pickle loading can execute code.

**Dataset source.** The NuPlan mini sample used for this implementation is available from [Hugging Face: runfuture/nuplan-mini-sample](https://huggingface.co/datasets/runfuture/nuplan-mini-sample). Its dataset card describes the NuPlan mini log, camera data, and converted annotation artifacts, along with provenance and license notes. Follow the NuPlan/Motional terms that apply to the data.

**Model source.** The DINOv3 implementation and pretrained model weights come from Meta's [official DINOv3 repository](https://github.com/facebookresearch/dinov3). The included `dinov3_vits16.pth` is the ViT-S/16 pretrained checkpoint; the viewer uses it by default, or you can pass another checkpoint path with `--weights`. The checkpoint originates from Meta’s official DINOv3 release. Follow the model's license and access terms.

From this directory, run:

```bash
python dino_player.py \
  --repo dinov3 \
  --weights dinov3_vits16.pth \
  --ann /path/to/2021.05.12.22.00.38_veh-35_01008_01518_infos.pkl \
  --root /path/to/nuplan-data-root
```

`--root` should be arranged so the NuPlan camera image paths in the annotation file resolve under it (the helper also handles paths containing `nuplan-v1.1/`). The viewer defaults to CPU when CUDA is unavailable. On a low-end machine, keep the default width and low playback frame rate.

## Member 2 — To be completed

_Add implementation idea, method, and relevant files._

## Member 3 — To be completed

_Add implementation idea, method, and relevant files._

## Member 4 — To be completed

_Add implementation idea, method, and relevant files._

## Member 5 — To be completed

_Add implementation idea, method, and relevant files._
