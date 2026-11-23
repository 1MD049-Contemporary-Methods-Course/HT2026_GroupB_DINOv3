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

## Olivia Buhr — DINO for Object Tracking​

Use a frozen DINO model as backbone for **semi-supervised video object
segmentation** on DAVIS 2017: given the ground-truth mask of the first frame, follow the
object(s) through the rest of the video. No training is done.

proof of concept: trying DINOv2 - ViT-B/14 on 10 videos. 

for fun: Comparison with YOLO on the same dataset. It only works for 80 coco classes it has been trained on but it performed better than DINO on this dataset. It does however missclassify camel as a horse, but the segmentation is still good for that example. 

### Method: label propagation
1. **Features:** each frame is resized to height 448 and passed through DINOv2 (ViT-B/14).
   Every 14×14 patch gets a feature vector (L2-normalized). Inputs use ImageNet
   mean/std normalization.
2. **First frame:** the ground-truth mask is downsampled to the patch grid, giving each
   patch a soft label (share of background / each object).
3. **Propagation:** for each patch in frame *t*, compute cosine similarity to all patches
   in frame 0 and the previous 5 frames, keep the top-10 matches, and take a
   softmax-weighted (temperature 0.07) average of their labels.
4. **Output:** patch labels are upsampled to full resolution and argmax gives the mask.
   Predicted labels are reused as memory for later frames.
5. **Evaluation:** J (region IoU) and F (boundary F-measure), averaged per object over
   frames 1..end and then over sequences. Metrics follow the DAVIS definitions but are
   not the official toolkit.

### Relevant files
| File | Description |
|---|---|
| `davis-dino-kaggle.ipynb` | DINOv2 label propagation + J&F evaluation |
| `davis_yolo_kaggle.ipynb` | YOLO segmentation + tracking baseline, same metrics |
| `figures/` | Example GIFs and comparison grids |

Dataset: [DAVIS 2017](https://davischallenge.org) (trainval, 480p)

## comparison

| Sequence | J&F YOLO | YOLO classes | J&F DINOv2 | Diff (YOLO − DINOv2) |
|---|---:|---|---:|---:|
| bike-packing | 0.652 | 1: bicycle, 2: person | **0.808** | −0.156 |
| blackswan | **0.945** | 1: bird | 0.900 | +0.044 |
| bmx-trees | **0.555** | 1: bicycle, 2: person | 0.516 | +0.039 |
| breakdance | 0.816 | 1: person | **0.881** | −0.065 |
| camel | **0.933** | 1: horse | 0.744 | +0.189 |
| car-roundabout | **0.969** | 1: car | 0.848 | +0.121 |
| car-shadow | **0.953** | 1: car | 0.878 | +0.075 |
| cows | **0.933** | 1: cow | 0.841 | +0.092 |
| dance-twirl | **0.833** | 1: person | 0.821 | +0.013 |
| dog | **0.946** | 1: dog | 0.731 | +0.215 |
| **Mean (10 seq.)** | **0.854** | | 0.797 | +0.057 |

## Shengkai Chen — DINOv3 vs. TinyCLIP for image retrieval on Imagenette

**Goal.** Compare two pretrained image encoders, **DINOv3 ViT-S/16** and **TinyCLIP ViT-22M/32**, on image-to-image retrieval. Given a query image, each model ranks the same gallery by visual feature similarity. Both models are used for inference without training or fine-tuning.

**Relevant file.** `dinov3-vs-tinyclip.ipynb` — dataset sampling, feature extraction, retrieval evaluation, visualizations, and optional retrieval with a custom query image.

### Models and method

| Model | Model identifier | Image encoder size | Retrieval feature |
|---|---|---|---|
| DINOv3 ViT-S/16 | `vit_small_patch16_dinov3.lvd1689m` | Approximately 21M parameters | Final global CLS token, 384 dimensions |
| TinyCLIP ViT-22M/32 | `TinyCLIP-auto-ViT-22M-32-Text-10M`, with `LAION400M` weights | 22,024,993 parameters in the saved run | Projected image embedding, 512 dimensions |

1. **Sample the data.** Select gallery images from Imagenette's `train/` split and query images from `val/`, with equal numbers per class. Both models use the same selected images.
2. **Preprocess each image.** Resize the shorter side to 224 pixels with bicubic interpolation, then center-crop to 224 × 224. Apply each model's corresponding input normalization.
3. **Extract features.** Run each encoder in evaluation mode with gradient computation disabled. L2-normalize every image embedding.
4. **Rank the gallery.** Compute the dot product between each query embedding and all gallery embeddings. For L2-normalized vectors, this is cosine similarity. Sort the scores in descending order.
5. **Evaluate retrieval.** A gallery image is relevant if its class matches the query's class. Labels are used for sampling, evaluation, and visualization; retrieval itself uses the image embeddings.

TinyCLIP is used through its image encoder only. This experiment does not use text prompts, text-to-image retrieval, or a separately trained classifier.

### Experiment setup

| Setting | Value |
|---|---|
| Dataset | Imagenette, 160 px version, 10 classes |
| Gallery | 300 images: 30 per class from `train/` |
| Queries | 100 images: 10 per class from `val/` |
| Sampling seeds | 42 for the gallery; 43 for the queries |
| Encoder input | 224 × 224 RGB images |
| Batch size | 32 |
| Retrieval cutoffs | K = 1, 5, 10 |
| Device | CUDA when available; otherwise CPU |

The classes are tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, and parachute. The selected image paths and labels are saved in `split_manifest.csv`.

### Metrics and results

- **Precision@K:** the fraction of the first K retrieved images that belong to the query's class.
- **Recall@K:** the fraction of all relevant gallery images found among the first K results.
- **Average Precision (AP):** precision averaged over the ranks at which relevant images appear, using the complete gallery ranking.
- **Mean Average Precision (mAP):** AP averaged over all query images.

All reported precision and recall values are also averaged over the queries. With 30 relevant gallery images per query, the maximum Recall@10 is 10/30, approximately 0.3333.

The following values come from the notebook's saved outputs for the default 300-image gallery and 100-query experiment:

| Metric | DINOv3 | TinyCLIP |
|---|---:|---:|
| mAP | 0.7880 | **0.8450** |
| Precision@1 | **0.9800** | 0.9700 |
| Precision@5 | 0.9100 | **0.9560** |
| Precision@10 | 0.8900 | **0.9410** |
| Recall@1 | **0.0327** | 0.0323 |
| Recall@5 | 0.1517 | **0.1593** |
| Recall@10 | 0.2967 | **0.3137** |

On this sampled split, DINOv3 has slightly higher top-1 precision, while TinyCLIP performs better at K = 5 and K = 10 and achieves higher mAP. The largest per-class mAP gap is for **church**: 0.3900 for DINOv3 and 0.9248 for TinyCLIP. The notebook also reports the complete per-class and per-query results.

This is a comparison on one small sampled split. Although the encoders have similar parameter counts, their training data, training objectives, and patch sizes differ, so the results do not isolate the effect of any single design choice.

## Rishit Reddy — DINOv3 for oral cancer cell classification

**Goal.** See how far DINOv3 alone can go on the [Multimodal Cancer Classification Challenge 2026](https://www.kaggle.com/competitions/multimodal-cancer-classification-challenge-2026): oral cytology cells imaged in brightfield and fluorescence, with labels only per patient (12 patients) and AUC over about 59,000 test cells from unseen patients.

**Method.** Frozen DINOv3 ViT-S/16 features with a linear probe, then LoRA adapted with Mean Teacher on the unlabelled test cells, WiSE-FT (moving the weights halfway back to frozen DINO), three self-training rounds and kNN smoothing of the scores.

**Result.** 0.685 → **0.768 public / 0.765 private AUC**. Details, all experiments and the code are in [`OralCancerClassification/`](OralCancerClassification/README.md).

## Mohammad Zahra — Video object tracking across DINO generations (v1, v2, v3)

**Goal.** Compare frozen DINO v1, DINOv2 and DINOv3 (four sizes, S to L) on semi-supervised video object segmentation: given the object masks in the first frame, follow the objects through the video with no training. Tested on all 30 [DAVIS 2017](https://davischallenge.org) validation videos and on 8 of our own phone clips, where SAM 2.1 serves as the reference.

**Method.** Label propagation as in the original DINO paper: each patch of a new frame copies the labels of its 5 most similar patches in frame 0 and the 7 previous frames. Every model gets the same settings and the same token grid, so only the features differ. An interactive tool (`track.py`) lets you pick any video, click the objects with SAM 2, check the masks on the whole video and get an MP4 and a GIF of the DINO tracking.

**Result.** DINOv3 is about 6 J&F ahead of v1 and v2 at the same size, and input resolution matters more than model size. Best: **76.3 J&F** on DAVIS (DINOv3-B at 2x) and **81.0 IoU** against SAM 2 on our clips (DINOv3-L at 2x). Every model fails on identical objects that cross. Details, all runs and the code are in [`DINOVideoTracking/`](DINOVideoTracking/README.md).
