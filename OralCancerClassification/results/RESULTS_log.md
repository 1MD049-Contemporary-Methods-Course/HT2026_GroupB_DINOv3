# Cancer-cell challenge: results so far (2026-11-11)

Public / private score on Kaggle (AUC over all test cells):

| Submission | Public | Private |
|---|---|---|
| Frozen DINOv3 ViT-S, original inputs, linear probe | 0.6849 | 0.6842 |
| Frozen, per-image per-channel standardised BF+FL, linear probe | **0.7163** | 0.7087 |
| LoRA r8 (attn qkv/proj), per-image std, gamma+rot90+flip aug, modality dropout 0.2, 2 epochs, all 12 patients | 0.6942 | 0.6877 |
| (older, from June) ResNet transfer notebooks | ~0.765 | ~0.763 |

Leave-patient-out CV, frozen linear probe, 35 folds (1 cancer + 1 healthy held out), 1000 cells/patient:
colour stats only 0.779 | orig BF+FL 0.814 | **norm BF+FL 0.871** | gray BF + norm FL 0.840 | norm BF only 0.806 | norm FL only 0.819 | str FL only 0.766.
Hardest folds: (17,09) 0.645, (16,09) 0.660, (05,15) 0.671; healthy pat_09 is the hard one.

LoRA vs frozen on held-out pairs (all cells of both patients): 17/09 frozen 0.683 vs LoRA 0.759; 16/09 frozen 0.706 vs LoRA 0.784; 03/07 (easy) 0.984 vs 0.997.
TTA (fold 17/09): 1 view 0.7586, 4 views 0.7599, 8 views 0.7604 (+0.002).

Takeaways: CV on a pair overstates test AUC (pooled over ~7 unseen patients). LoRA gains on pairs did not carry to the test set (0.694 < 0.716).
Untested ideas: rank-average LoRA + frozen; fewer epochs / lower rank / stronger regularisation; per-patient score calibration (the test pools patients).

## Check 2026-11-15 (explore_mil.py, old non-normalised embeddings, 1000 cells/patient, 40 of 350 splits)
Simulated pooled test, 2 cancer + 3 healthy held out: baseline LR 0.772 | malignant-like cells only 0.736 (wins 18%) | k-means pseudo-patient blend 0.783 (wins 65%) | oracle true-patient mean 0.942.
Unsupervised clustering recovers patients poorly (purity 0.30 on 12 patients, 0.44 on 5) although a supervised 12-way patient-ID classifier gets 0.63 (chance 0.08).
Only 27% of held-out cancer-patient cells score above the healthy 95th percentile: the signal is diffuse across cells, not concentrated in a few malignant ones.
Patient-learned clustering (explore_lda.py, same splits, k=5 on held-out cells, PCA re-standardised before k-means):
raw PCA purity 0.50, blend 0.795 (wins 75%, min 0.33) | LDA on train patient IDs purity 0.49, blend 0.790 | patient-ID logits purity 0.47, blend 0.776.
Directions that separate the training patients do not separate new patients: no gain over raw features.

## Mean Teacher, 2026-11-15 (mt.py, A40, ~3 min/epoch; files in pod_results/mt_2026-11-15)
Teacher = frozen DINOv3 + balanced LR (rebuilt: 0.7163 / 0.7087 exactly). Same teacher at 224 px: 0.7015 / 0.6970 (worse).
Student = LoRA r4 on last 6 blocks, starts identical to teacher; each step 64 labelled cells (patient/class-balanced) + 64 test
cells with soft labels from the EMA teacher (0.999); strong aug (crop, any rotation, blur, noise, erasing, gamma, modality dropout, drop-path).

| epoch | student | EMA teacher | student+frozen rank avg | spearman student~frozen | labelled train AUC |
|---|---|---|---|---|---|
| 1 | **0.7443** / 0.7411 | 0.7244 / 0.7189 | 0.7366 / 0.7314 | 0.868 | 0.936 |
| 2 | 0.7366 / 0.7350 | 0.7440 / 0.7403 | 0.7370 / 0.7325 | 0.790 | 0.921 |
| 3 | 0.7345 / 0.7330 | **0.7446 / 0.7420** | 0.7364 / 0.7320 | 0.780 | 0.918 |

+0.028 public / +0.033 private over the teacher; best DINOv3 result so far. Labelled train AUC went DOWN (0.955 -> 0.918):
no patient memorisation (unlike plain LoRA, 0.694). Student gain comes in epoch 1; EMA teacher catches up by epochs 2-3.

Seed 1, same recipe (pod_results/mask_seed_2026-11-15): student 0.7409 / 0.7342 / 0.7303, EMA 0.7239 / 0.7420 / 0.7411 (ep1/2/3 public).
Seed noise is ~0.003-0.004 public: differences below ~0.005 between runs are noise.
+ masked consistency (MIC; 50% of patch tokens -> DINOv3 mask token, student must match teacher; seed 0): student 0.7410 / 0.7298 / 0.7272,
EMA 0.7252 / 0.7413 / 0.7394. Drifts further from the teacher (spearman 0.746) and lower train AUC (0.900) but no test gain: masking does not help here.

Rebuild on a new pod: pod_setup.sh (deps + Kaggle data), push.sh (weights + scripts), embed_robust.py (5 input variants, ~7 min on RTX 2000 Ada), analyse3.py (GPU CV), train.py (LoRA, `--val none --predict_test` for the final model, `--load X.pt --epochs 0 --tta 4` to predict).

## kNN score smoothing, 2026-11-17 (explore_knn.py simulation, knn_smooth.py on test)
Simulated 40 splits: pooled AUC 0.772 -> 0.800 (k=30-100, lam 0.5, wins 95% of splits); same-patient neighbour rate only ~0.55.
Test (recipe fixed k=100, lam=0.5): frozen 0.7163 -> 0.7361 / 0.7294; Mean Teacher EMA 0.7446 -> 0.7476 / 0.7451 (new best, but +0.003 = noise).
Colour diagnostic (diag_colour.py): R2 of colour stats -> test logit: frozen 0.39, MT EMA 0.57; MT change vs frozen 0.25.
Next: see PLAN.md (autonomous Ada run).

## Overnight CPU checks, 2026-11-18 (Mac; cpu_tests.py, stack_check.py, headmt_test.py; logs in out/)
Re-extracted colour-normalised features on MPS (embed_mac.py -> emb_norm_{train,test}.npz; matches GPU teacher, spearman 0.9996).
40 simulated splits (2 cancer + 3 healthy, 1000 cells/patient), pooled AUC, normalised features:
linear probe 0.845 | patient-mean AUC 1.000 on ALL splits | 90% dropout 0.833 | MLP-256 0.829 | remove patient dirs 0.82 (all hurt)
head-only Mean Teacher 0.884 (wins 100%) | kNN colour-stats graph 0.886 (100%) | kNN feature graph 0.873.
Real density (all held-out cells, 6 splits): base 0.805, kNN 0.853, head-MT 0.848, head-MT + kNN 0.869 (6/6).
REAL TEST: head-only MT + kNN seeds 0/1/2: 0.7056/0.7081/0.7065, ensemble 0.7068/0.7033 -> WORSE than frozen 0.7163. Dropped.
Why: in the simulation the frozen probe orders every patient correctly (patient-mean AUC 1.0), so self-training only sharpens a right
ordering; on the real test at least one patient is likely misordered and self-training amplifies it (confirmation bias).
=> The easy simulation is fine for gating smoothing, NOT for self-training ideas. Gate those on hard split B (patients misordered).
kNN on colour-stats graph (k100, lam .5): frozen 0.7387/0.7325; Mean Teacher EMA 0.7505/0.7488 = NEW BEST (out/knn/mtema_knncolour.csv).

## Ada run, 2026-11-18 (RTX 2000 Ada, $0.24/h)
Fixed: mt.py defaults (LoRA r4 last 6 blocks, EMA .999, bs 64+64, lr 1e-4, 3 epochs, 128 px), C=0.05 teacher. Gate = hard split B
(`--holdout pat_16,pat_17,pat_09,pat_11,pat_14`): pooled AUC + patient-mean AUC on held-out cells. Submit only if it beats the r4 MT
baseline on split B; gain = seed-mean >= +0.01; max 4 submissions/idea; final models get colour-graph kNN smoothing.
Hypotheses: H1 LoRA overfits -> r2 >= r4 > r8 on split B. H2 ResNet-50 (Lian recipe, 6-ch early fusion) is a strong, different
model -> beats DINO MT alone on split B and helps in an ensemble. H3 surgical LoRA / WiSE-FT / attention pooling: small gains at best.
Scope (user): DINO-only study. ResNet dropped. H2 now = token-level BF+FL fusion (--fuse) beats late fusion on split B.
- Ada speed: ~4 min/epoch MT (781 steps), ~15 min per 3-epoch gate run. Split B full cells, frozen teacher: pooled 0.6563, patient-mean AUC 1.0
  (patients ordered right; difficulty is within-patient overlap). Smoke: --fuse, --attnpool, --lora_blocks, --mlp, --wise all run.
- Frozen probe by layer (layer_tests.py, pooled AUC split A / split B, all cells): L11 cls+mean (current) 0.8016 / 0.6564;
  L8 0.8215 / 0.6572; **L5 0.8512 / 0.6853**; L8+L11 0.8140 / 0.6586; all layers+std 0.8173 / 0.6792 (C=.01: 0.8282 / 0.6807);
  +std pooling and test-statistics standardisation (tstd) do not help. Middle layers carry the signal.
- Split B MT seed 0, student / EMA ep3: r2 0.7387 / 0.7221; r4 0.7501 / 0.7343; r8 0.7570 / 0.7481. H1 (LoRA overfits) NOT supported
  in MT: higher rank is better (differences ~1 seed-noise step, r4 seeds 1-2 pending). Student > EMA at ep3 in all three.
- Added mt.py --layer N (manual forward to block N) and --ptune norm|bitfit. Queue: L5 + LoRA blocks 0-5 (r4, r8).
- Token fusion (--fuse, r4, split B): frozen fused teacher 0.6479 (< late 0.6564); student ep3 0.6151, EMA ep3 0.6309 (patient-mean AUC 0.67-0.83). FAILS: student drifts, all patient means collapse to ~0.1. Try 1 of 2; only retry combined with --layer 5.
- REAL TEST frozen layer-5 probe: public 0.6595 / private 0.6527 (+colour kNN 0.6827 / 0.6768) vs L11 0.7163 / 0.7087. Splits A/B said L5 +0.03..0.05: they mislead for representation changes (likely stain/acquisition texture shared within the 12 train patients). Dropped L5 MT runs. Next: full-test MT r8 (pre-committed: submit EMA ep3 raw + colour kNN only).
- Split B r4 seed 1: student 0.7501 / EMA 0.7371 (seed 0: 0.7501 / 0.7343) -> seed noise ~0.003. So r8 (+0.007 student, +0.012 EMA) and r2 (-0.012) differences are real. Layer 40-split test killed (real test already answered it).
- REAL TEST MT r8 EMA ep3 seed0: 0.7374 / 0.7356; + colour kNN 0.7414 / 0.7405 (r4: 0.7446/0.7420, kNN 0.7505/0.7488). Split B said r8 > r4 by 0.012 -> 3rd time split B misleads. r8 student drifts further from frozen (spearman 0.75 at ep3). Hypothesis H4: hidden patients reward staying close to frozen DINO. Split-B queue dropped; testing WiSE-FT alpha .5 on r8 and full r2 on the real test (pre-committed: EMA ep3 raw + kNN). NOTE: from here the leaderboard is the only validation -> optimistic, flagged.
- REAL TEST WiSE-FT alpha 0.5 (paper default, not tuned) on MT r8 EMA ep3: 0.7418 / 0.7375; + colour kNN **0.7586 / 0.7552 = NEW BEST** (prev 0.7505/0.7488). Supports H4 (stay close to frozen). Next: WiSE 0.5 on r4 seed1 EMA ep3 (0.7420 public raw), full r2.
- REAL TEST WiSE 0.5 on r4 seed1 EMA ep3: raw 0.7420 / 0.7376; +kNN **0.7593 / 0.7557 (best)**. CONTROL r4s1 plain + kNN 0.7482 / 0.7466 -> WiSE adds +0.011/+0.009 with kNN (r8: +0.017/+0.015); raw alone ~+0-0.004. WiSE and kNN are synergistic.
- Ensemble WiSE r4s1 + WiSE r8 (z-logit) + kNN: 0.7591 / 0.7556 (no gain; after WiSE they correlate 0.994).
- Output-space blend frozen L11 + MT r4s1 (z-logit avg) + kNN: 0.7505 / 0.7468 vs WiSE (weight-space) 0.7593 / 0.7557, despite spearman 0.994 between the two -> weight-space interpolation matters (as in WiSE-FT paper).
- H5 (user idea + lit: Shen 2022 'Connect not collapse', Gandelsman 2022 TTT-MAE): label-free self-supervised adaptation on test+train images (iBOT-style: masked student vs EMA teacher, cls cross-view + masked-patch cosine, LoRA r4, 2 epochs), then refit probe on all 12 patients. No pseudo-labels -> no confirmation bias. Pre-committed submissions: EMA ep2 plain + colour kNN.
- REAL TEST MT r2 EMA ep3: +kNN 0.7456 / 0.7432; WiSE 0.5 + kNN 0.7515 / 0.7476. WiSE gain grows with rank (r2 +0.006, r4 +0.011, r8 +0.017 public): bigger adapter drifts further, so pulling back gains more. r4 best overall (0.7593 / 0.7557).
- REAL TEST SSL adaptation (iBOT-style, LoRA r4, 2 ep, refit probe): 0.7189 / 0.7111 (frozen 0.7163/0.7087); +kNN 0.7410 / 0.7351 (frozen+kNN 0.7387/0.7325) -> ~+0.002, no real gain; teacher cls batch-std 0.34 -> 0.23 (starting to collapse). H5 rejected.
- END of Ada run 2026-11-18: pod stopped after copying everything to pod_results/ada_2026-11-18/. Cost ~$0.80 BEST: MT r4 seed1 EMA ep3 + WiSE 0.5 + colour kNN = 0.7593 / 0.7557. Summary: SUMMARY_2026-11-18.md.
- REAL TEST frozen DINOv3 ViT-B/16 probe (same recipe, 3072-d, C=.05): 0.6832 / 0.6776; + colour kNN 0.7115 / 0.7080 (ViT-S: 0.7163/0.7087, kNN 0.7387/0.7325). Bigger model WORSE by ~0.03; probe fits train patients harder (patient logits to +8.4). ViT-L skipped (trend down). Weights downloaded to pod only.
- REAL TEST round-2 MT (--init WiSE-0.5 of r4s1 EMA ep3; student+EMA start there; LoRA r4, 3 ep): EMA ep3 + kNN 0.7272 / 0.7274 (drifts: spearman ema~init 0.89); + WiSE 0.5 + kNN **0.7651 / 0.7619 = NEW BEST**. Iterated self-training + pull-back works. Round 3 queued (same recipe, seed 1).
- PLATEAU ANALYSIS (ceiling_analysis.py, split B): patient-mean AUC 1.0 (frozen+MT) -> all loss within patients; per-pair cell AUC 0.61-0.82 (MT); ~14% clearly abnormal cells (frozen 6%); colour-graph neighbours 50% same patient (chance 20%); kNN lambda .5 best (0.761), .95 hurts (0.659); k-means patient purity 0.28-0.31, test silhouette 0.24. ~0.76 near ceiling of per-cell scoring without patient grouping.
- REAL TEST round-3 MT (init round-2 EMA ep3 WiSE .5, seed 1): EMA ep3 + kNN 0.7284 / 0.7288; WiSE .5 + kNN **0.7681 / 0.7651 = FINAL BEST** (+0.003, ~seed noise: flattened). Pod stopped after copying to pod_results/ada2_2026-11-18/.
