# AnoDe reproduction roadmap

This repository rebuilds **AnoDe**, a UGM Machine Learning capstone project
(image anomaly detection for manufacturing quality inspection), from scratch.
The original website prototype was lost, and the report PDF was the starting
point. Some of the original team notebooks turned out to still be on Kaggle
(see below). The goal is a benchmark that anyone can rerun, a deployed web
platform, and a rewritten paper.

## Decisions (2026-10-19)

| Topic | Decision |
|---|---|
| Compute | Kaggle notebooks (free T4 GPU, 30 h/week), driven from the Kaggle CLI |
| Datasets | MVTec AD (15 categories) and MVTec LOCO AD (5 categories) |
| Website | Full platform: React frontend + FastAPI backend on Hugging Face Spaces |
| Paper | Rewritten in LaTeX (IEEE format) with the new results |

### Dataset source

The official MVTec download links (mydrive.ch) returned HTTP 404 on
2026-10-19. The notebooks therefore attach Kaggle datasets instead:

- MVTec AD: [`mhasbiarz/mvtec-anomaly-detection`](https://www.kaggle.com/datasets/mhasbiarz/mvtec-anomaly-detection)
  (our own upload from April 2025, made for the original capstone).
- MVTec LOCO AD: [`rakibulhassannishat/mvtec-loco-ad`](https://www.kaggle.com/datasets/rakibulhassannishat/mvtec-loco-ad),
  a mirror of the original `mvtec_loco_anomaly_detection.tar.xz` (6.13 GB,
  the same size as the official file). The `anode-data` notebook extracts it
  once, and later notebooks mount that notebook's output.

`kaggle/data/prepare_data.py` compares the image and mask counts for every
category and split against the dataset papers:

- **MVTec AD:** matches exactly (5,354 images).
- **MVTec LOCO AD:** matches, except `splicing_connectors`. The paper lists
  354 train / 59 validation images, but the released archive has 360 / 60
  (3,651 images in total, not 3,644). Dataset Ninja's listing of the archive
  agrees with 3,651. We use the archive as released.

Both datasets are licensed CC BY-NC-SA 4.0 (non-commercial, attribution,
share-alike). This repo never commits the datasets. The website only shows a
few sample images, with attribution.

### Original notebooks

Some of the original team notebooks are still on Kaggle (Autoencoder,
ResNet50, "Res2Net" and PatchCore for zipper, wood and transistor). They are
the reference for what each model originally did.

## Models

Each model follows the original AnoDe notebooks or, where the report cites a
published method, that method. Deviations from the original are listed in
the last column.

| # | Model (registry name) | Definition | Datasets | Differences from the original notebook |
|---|---|---|---|---|
| 1 | Autoencoder L2 (`autoencoder_l2`) | The original fully convolutional AE (3 conv + avg-pool stages, 3 transposed convs), MSE loss, Adam 1e-3, 100 epochs. Score = image MSE. | AD, LOCO | Trained from scratch per category (the original warm-started from another category's checkpoint) |
| 2 | ResNet50 + kNN (`resnet50_knn`) | Global-pooled ResNet-50 features, the 500 highest-variance dimensions, mean distance to the k = 50 nearest training images | AD, LOCO | None |
| 3 | Feature AE, the report's "Res2Net" (`feature_ae_resnet50`, `feature_ae_res2net50`) | layer2+layer3 features (3x3 aggregation, 1536 ch) reconstructed by a 1x1-conv AE (latent 100), 50 epochs. Score = mean of the 10 largest feature errors. | AD, LOCO | The report called it Res2Net, but the notebook used ResNet-50. We run both backbones. |
| 4 | PatchCore (`patchcore`) | Roth et al. 2022: WideResNet-50 layer2+layer3, 10% greedy coreset, neighbourhood re-weighting (b = 9) | AD, LOCO | The notebook used ResNet-50 with a random 10% subsample and no re-weighting. We implement the published method that the report cites. |
| 5 | GLASS | Chen et al. 2024, official code: the released MVTec AD checkpoints, plus our own training | AD, LOCO | No notebook survives |
| 6 | CSAD | Hsieh & Lai 2024, official code: the released LOCO checkpoint, plus our own training | LOCO only (built for logical anomalies) | No notebook survives |

All models use the same input: resize to 256x256 (no crop), ImageNet
normalisation for pretrained backbones, and [0, 1] pixels for the image AE.
The original notebooks used 224x224 without ImageNet normalisation.

## Evaluation protocol (fixes for the original report)

- **Detection:** image-level AUROC (primary metric). On LOCO, logical and
  structural AUROC are reported separately, with their mean as the headline
  number.
- **Localisation:** pixel AUROC and AUPRO (FPR ≤ 30%) on MVTec AD, and sPRO
  (official LOCO definition) on MVTec LOCO AD.
- **Accuracy / F1:** the threshold is set **without test labels**: mean + 3
  std of the scores on held-out normal validation images (10% of MVTec AD
  train, the official LOCO validation split). The threshold-free F1-max is also
  reported, labelled clearly as an optimistic upper bound.
- **Latency:** milliseconds per image, batch size 1, after a warm-up, on a
  named GPU (Kaggle T4) and on CPU (the deployment target).
- **Seeds:** trained models are run with 3 seeds and reported as mean ± std.
- **Checkpoint selection:** never on the test set. Both official GLASS and
  official CSAD training scripts evaluate on the test set during training and
  keep the best test checkpoint. We report their released checkpoints as
  published, *and* our own training with the last checkpoint.
- **Process capability:** Cp/Cpk is computed from the continuous anomaly
  score against a one-sided upper spec limit (Cpu), not from pass/fail counts.
  The P-chart uses the per-batch defect proportion.

## Phases

1. Project skeleton and Kaggle access ✓
2. Dataset check on Kaggle ✓
3. Benchmark of models 1–4 ✓ (2026-10-20)
4. GLASS and CSAD ✓ (official checkpoints; own GLASS training, 50 epochs, on LOCO and 3 AD categories)
5. Evaluation, tables, figures (sPRO for LOCO)
6. Website (React + FastAPI) ✓ built and tested locally (2026-10-20)
7. Deployment to Hugging Face Spaces ✓ (static demo: Docker Spaces need a paid plan since 2026)
8. Paper rewrite (LaTeX, IEEE) ✓ draft in paper/
9. Public GitHub release

## Results so far (models 1–4)

Image AUROC, mean over categories (LOCO: mean of logical and structural).
Seeded models: mean over 3 seeds. Full tables: `results/tables/`.

| Model | MVTec AD | MVTec LOCO AD | Original report (MVTec AD) |
|---|---|---|---|
| AE (L2) | 0.606 | 0.662 | 0.663 |
| ResNet50 + kNN | 0.802 | 0.732 | 0.739 |
| Feature AE (ResNet-50) | 0.980 | 0.817 | 0.972 |
| Feature AE (Res2Net-50) | 0.901 | 0.754 | – |
| PatchCore | 0.982 | 0.822 | 0.959 |

PatchCore's 0.982 is close to the 0.991 reported by Roth et al. (who use a
224 px center crop). That agreement is a sanity check of the implementation.

With the label-free validation threshold (mean + 3 std), MVTec AD F1 is
0.931 for Feature AE (ResNet-50), 0.915 for PatchCore, 0.739 for Feature AE
(Res2Net-50), 0.515 for ResNet50+kNN and 0.152 for the AE. The original
report's F1 values were higher because their thresholds were tuned on test
labels.

The Res2Net-50 backbone is clearly worse than ResNet-50 in the feature AE
(0.901 vs 0.980). It also varies more between seeds (grid 0.820 ± 0.201). So
the original report's best model owed its results to ResNet-50, not Res2Net.

## GLASS and CSAD (official checkpoints)

Both run through `anode.external` (official code at pinned commits, with our
metrics and splits):

| Model | Dataset | Ours | Published |
|---|---|---|---|
| GLASS, released checkpoints | MVTec AD, image AUROC | 0.999 (AUPRO 0.969) | 0.999 |
| CSAD, released checkpoints | LOCO, mean of logical/structural AUROC | 0.957 (logical 0.977, structural 0.937) | 0.957 (PyTorch release) |

The original report's GLASS and CSAD tables match these released checkpoints
to the fourth decimal. So the report used the authors' checkpoints and did
not train these two models itself.

**Test-set checkpoint selection.** Both official training scripts evaluate
on the test set during training and keep the best test checkpoint. The
released GLASS file names record the chosen epoch (out of 640): bottle 543,
cable 72, capsule 56, carpet 431, grid 581, hazelnut 182, leather 119,
metal_nut 5, pill 233, screw 139, tile 369, toothbrush 543, transistor 52,
wood 637, zipper 149. We therefore treat the released numbers as optimistic
upper bounds. We train GLASS ourselves with the official code and report the
**last** epoch. The test evaluation is logged every 5 epochs only to measure
the gap between test-selected and last-epoch checkpoints (model name
`glass_testselected`, analysis only).

**Budget.** On a Kaggle T4 the official GLASS loop takes ~100 s per epoch
(LOCO pushpins), so 640 epochs would take ~18 h per category. We use 50
epochs, on all 5 LOCO categories and on MVTec AD zipper, wood and transistor
(the original notebooks' categories). CSAD retraining (segmentor + 60k LGST
iterations per category) is not yet scheduled.

## Our GLASS training (official code, 50 epochs, last vs test-selected epoch)

| Data | Last epoch (reported) | Test-selected epoch | Released checkpoint |
|---|---|---|---|
| MVTec AD zipper / wood / transistor | 0.998 / 0.997 / 0.992 | 0.998 / 0.997 / 0.994 | 0.9995 / 0.9991 / 0.9987 |
| LOCO, mean of logical and structural | 0.813 | 0.828 (+0.015) | – |
| LOCO logical | 0.713 | 0.759 (+0.046) | – |
| LOCO structural | **0.913** | 0.898 | – |

Test-set selection adds at most +0.002 on MVTec AD but +0.015 on LOCO (up
to +0.10 logical AUROC on pushpins). The official criterion (image + pixel
AUROC) also trades structural for logical performance. GLASS has the best
structural AUROC on LOCO of all our own models (0.913), but like every model
without explicit component reasoning it stays far behind CSAD on logical
anomalies (0.713 vs 0.977).

## Deployment findings (website, 2026-10-20)

**Thresholds need unseen normal images.** The website sets each product's
decision threshold as mean + 3 std of the scores on held-out normal images,
the same rule as the benchmark. For the released GLASS checkpoints no such
images exist: they were trained on every normal training image and score new
normal images much higher than their own training images. Neither the
validation rule (mean false-alarm rate 33.5% on good MVTec AD test images,
recall 99.6%) nor the discriminator's natural 0.5 cut-off (31.7%) gives a
usable operating point.
Models trained on our split do: at the validation threshold, PatchCore
flags 10.1% of good MVTec AD parts and catches 88.7% of defects (Feature AE
ResNet-50: 7.2% / 90.5%). **The website therefore deploys PatchCore for
MVTec AD** (and for custom products), and CSAD's released checkpoints for
LOCO, since CSAD never trains on the official validation split (false-alarm
rate 1.8%).

**CSAD is very sensitive to JPEG compression.** On original LOCO PNGs our
CPU runtime reproduces the official scores (0.72x and -0.22x the threshold
for two good juice bottles, official 0.51x and -0.20x). Saving the same
images as JPEG (quality 90) moves them to 1.93x / 0.98x at full resolution
and 3.17x / 1.36x at 512 px, which turns good parts into false alarms.
Resizing alone (512 px PNG: 0.49x / 0.19x) does not. PatchCore classified all
12 bottle samples correctly even from JPEG. The website ships PNG samples and
the camera captures PNG frames.

**Process capability.** Scores of a batch that mixes good and defective
parts are bimodal, so the normal-theory Cpk = (USL - mean) / 3 sigma is
dragged down by the inflated sigma. Reports also show the yield-based
equivalent Phi^-1(1 - p) / 3 (e.g. 0.62 instead of 0.31 at p = 3.2%).

## Known errors in the original report

- The whole PatchCore Pill row (AUC, accuracy, F1, runtime) duplicates the
  Res2Net Pill row.
- The Autoencoder Zipper row shares its F1 (0.7111) and runtime (6.81) with
  the ResNet50+kNN Zipper row.
- The LOCO table lists "Bottle"; the category is "Juice Bottle".
- CSAD was the only model run on LOCO, so "most effective" was not shown.
- The GLASS and CSAD numbers are the authors' released checkpoints, which
  were selected on the test set. They were not produced by the team's own
  training.
- Runtime units and hardware were not stated. The runtime column appears to
  mix training time (Autoencoder, "Res2Net": the notebooks time the training
  loop) with test time (ResNet50+kNN, PatchCore: the notebooks time the test
  loop).
- The Accuracy/F1 threshold selection was not described. In the "Res2Net"
  notebook it was the F1-optimal threshold chosen on the test labels, which
  inflates accuracy and F1.
- The "Res2Net" model used a ResNet-50 backbone, not Res2Net.
