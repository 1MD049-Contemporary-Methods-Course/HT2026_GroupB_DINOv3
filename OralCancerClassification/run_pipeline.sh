#!/usr/bin/env bash
# Final pipeline (Kaggle public 0.7681 / private 0.7651). One GPU; each Mean Teacher round takes ~15 min on an RTX 2000 Ada.
# Expects: data/ (Kaggle files: train.csv, sampleSubmission.csv, BF/, FL/), dinov3/ (Meta repo), dinov3_vits16.pth.
set -euo pipefail
CK=emateacher_size128_r4b6_ema0.999      # checkpoint name prefix written by mt.py
mkdir -p runs/round1 runs/round2 runs/round3

# Colour statistics of every cell (mean/std of BF and FL RGB) for kNN smoothing
python embed_robust.py train && python embed_robust.py test
python -c "import numpy as np; tr=np.load('emb2_train.npz'); te=np.load('emb2_test.npz'); np.savez('colour_stats.npz', train_stats=tr['stats'], test_stats=te['stats'], test_names=te['names'])"

# Round 1: frozen probe -> Mean Teacher (LoRA r4, last 6 blocks, EMA 0.999, 3 epochs)
python mt.py --rank 4 --seed 1 --out runs/round1
# Round 2 and 3: start student and teacher from the previous model after WiSE-FT 0.5, train again
python mt.py --rank 4 --init runs/round1/${CK}_seed1_ep3.pt --wise 0.5 --out runs/round2
python mt.py --rank 4 --init runs/round2/${CK}_seed0_init_ep3.pt --wise 0.5 --seed 1 --out runs/round3

# WiSE-FT 0.5 on the round-3 teacher (relative to frozen DINO), then kNN smoothing -> submission CSV
python mt.py --rank 4 --load runs/round3/${CK}_seed1_init_ep3.pt --wise 0.5 --out runs/round3
python -c "import numpy as np, pandas as pd; p=np.load('runs/round3/${CK}_seed1_init_ep3_tta1_wise0.5.npy'); pd.DataFrame({'Name': pd.read_csv('data/sampleSubmission.csv').Name, 'Diagnosis': p}).to_csv('runs/final.csv', index=False)"
python knn_col.py runs/final.csv          # -> runs/final_knncol.csv (the submitted file)
