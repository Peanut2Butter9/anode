# AnoDe — Anomaly Detection Platform for Manufacturing

**Live demo:** [huggingface.co/spaces/peanut2butter9/anode](https://huggingface.co/spaces/peanut2butter9/anode) (static: precomputed results for the sample images) ·
**Models:** [huggingface.co/peanut2butter9/anode-models](https://huggingface.co/peanut2butter9/anode-models) ·
**Protocol and findings:** [docs/ROADMAP.md](docs/ROADMAP.md)

AnoDe benchmarks unsupervised image anomaly detection on
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and
[MVTec LOCO AD](https://www.mvtec.com/company/research/datasets/mvtec-loco),
then deploys the best model per product family in a web platform for visual
quality inspection. Each inspection shows a heatmap of where the model sees
the defect, and finished batches feed a P-chart and a process-capability
report.

This repository is a from-scratch reproduction of a Machine Learning capstone
project at Universitas Gadjah Mada. The original notebooks were mostly lost,
so every model was reimplemented (following the surviving notebooks where
they exist) and rerun under one strict protocol. Doing so corrected several
errors in the original report.

## Findings

- **Reproduction.** The original report's numbers are reproduced where they
  were sound. Our autoencoder matches its per-category pattern (0.606 vs 0.663
  image AUROC). The released GLASS (0.999) and CSAD (0.957) checkpoints match
  the report to four decimals, which shows the report used the authors'
  checkpoints rather than its own training.
- **The "Res2Net" model was a feature autoencoder on ResNet-50.** With the
  Res2Net-50 backbone the report names, it drops from 0.980 to 0.901.
- **Thresholds from test labels.** The original accuracy/F1 used thresholds
  tuned on the test labels. Here every threshold comes from held-out normal
  images (mean + 3σ).
- **Test-set checkpoint selection.** The official GLASS and CSAD training
  scripts both keep the checkpoint with the best *test* score (GLASS's
  released metal-nut model is from epoch 5 of 640). Retraining both and
  comparing the last checkpoint with the test-selected one: selection adds
  ≤ 0.002 on MVTec AD but +0.015 (GLASS) and +0.006 (CSAD) on LOCO, and up to
  +0.046 logical AUROC.
- **Deployable ≠ best benchmark score.** The released GLASS checkpoints leave
  no unseen normal images to calibrate a threshold. On new normal images they
  flag about a third of good parts (33.5% at the validation rule). The platform therefore deploys PatchCore
  (0.982, 10% false alarms at its validation threshold) for MVTec AD.
- **CSAD is fragile under JPEG compression.** Saving LOCO images as JPEG
  (quality 90) shifts CSAD's score by 1–3 thresholds and turns good parts
  into false alarms. PatchCore is unaffected.

## Results

Image AUROC, mean over categories. LOCO: mean of logical and structural
AUROC. ± is the std over 3 seeds for trained models. "Released" checkpoints
were selected on the test set by their authors, so they are upper bounds.
Per-category tables: [results/tables/tables.md](results/tables/tables.md).

| Model | MVTec AD | AUPRO (AD) | F1 @ val. threshold (AD) | LOCO | LOCO logical | LOCO structural |
|---|---|---|---|---|---|---|
| Autoencoder (L2) | 0.606 ± 0.001 | 0.509 | 0.152 | 0.662 ± 0.005 | 0.587 | 0.738 |
| ResNet-50 + kNN | 0.802 | – | 0.515 | 0.732 | 0.747 | 0.717 |
| Feature AE, ResNet-50 (the report's "Res2Net") | 0.980 ± 0.000 | 0.918 | **0.931** | 0.817 ± 0.002 | 0.749 | 0.886 |
| Feature AE, Res2Net-50 | 0.901 ± 0.016 | 0.890 | 0.739 | 0.754 ± 0.033 | 0.737 | 0.771 |
| PatchCore (deployed for AD) | 0.982 | 0.928 | 0.915 | 0.822 | 0.786 | 0.857 |
| GLASS, our training (50 epochs, last epoch) | 0.995 (3 categories) | – | – | 0.813 | 0.713 | 0.913 |
| GLASS, released checkpoints | **0.999** | **0.969** | – | – | – | – |
| CSAD, our training (12k of 60k LGST iterations, last state) | – | – | – | 0.929 | 0.945 | 0.913 |
| CSAD, released checkpoints (deployed for LOCO) | – | – | – | **0.957** | **0.977** | **0.937** |

## Limitations

- **GLASS and CSAD were retrained with short schedules and one seed.**
  GLASS: 50 of 640 epochs (about 100 s per epoch on a T4), on all 5 LOCO
  categories and 3 MVTec AD categories (zipper, wood, transistor). CSAD: the
  full 120 segmentor epochs but 12,000 of 60,000 LGST iterations. The LGST
  runs at 1.2 iterations/s on a T4, so the full schedule would take about
  14 h per category; our run took 15.9 GPU-hours on a Kaggle T4×2. The gap
  between our CSAD (0.929) and the released one (0.957) therefore mixes the
  shorter schedule with test-set selection, which adds +0.006 on our run.
  F1 at the validation threshold is 0.816 for our CSAD against 0.802 for the
  released one.
- **Thresholds come from few images.** Validation sets hold 6 (toothbrush) to
  69 (pushpins) normal images, so the mean + 3σ thresholds, and the
  false-alarm rates they produce, vary noticeably between categories.
- Images are resized to 256 × 256 without a center crop for our own models.
  Published PatchCore numbers use a 224 px center crop, so they are not
  strictly comparable (0.982 here vs 0.991 reported).

## Repository layout

| Path | What it is |
|---|---|
| `src/anode/` | Benchmark package: datasets, metrics (AUROC, AUPRO, F1), models, reports, statistical process control |
| `src/anode/external/` | Adapters that run the official GLASS and CSAD code inside our protocol |
| `kaggle/` | One folder per Kaggle notebook (data check, benchmarks, GLASS/CSAD, website bundle) |
| `results/` | Per-run metrics and scores from Kaggle, and the generated tables |
| `web/` | Website: FastAPI server (`web/server`), React frontend (`web/frontend`), bundle and deploy scripts |
| `docs/` | Roadmap, protocol, findings, and the original report's numbers |
| `tests/` | Unit and end-to-end tests (synthetic data, no downloads) |

## Reproduce

```bash
uv venv --python 3.12 && uv pip install -e ".[report,dev]"
.venv/bin/python -m pytest -q
```

The experiments run as Kaggle notebooks (free T4 GPU). The package is
uploaded as a private dataset with `kaggle/publish_code.sh`, and each
notebook is pushed with `kaggle kernels push -p kaggle/<folder>`. Order:
`data` → `baselines-features`, `baselines-ae` → `glass-csad-probe`,
`external`, `glass-loco`, `glass-ad` → `bundle`; `csad-probe` → `csad-train`
for our CSAD training. Then rebuild the tables:

```bash
.venv/bin/python -m anode.report results/kaggle --out results/tables
```

The public demo is a free static build: `web/build_static.py` runs the deployed
models once on every sample image, and `python web/deploy.py static --repo
USER/anode` publishes the result. Live inference (your own images, the camera,
custom products) needs the full app. To run it locally, put the model bundle in
`bundle/` (`hf download peanut2butter9/anode-models --local-dir bundle`) and the official
CSAD code in `third_party/CSAD` (commit `d46cf85`):

```bash
cd web/frontend && npm install && npm run build && cd ../..
ANODE_BUNDLE=bundle ANODE_REPOS=third_party .venv/bin/python -m uvicorn anode_web.app:app --app-dir web/server --port 7860
```

## Credits

The original capstone (Machine Learning Capstone Project, Department of
Mathematics, Universitas Gadjah Mada) was built by Albertus Reinaldo Manibuy,
Daffa Elgo Santosa, Harun Yahya, Jeremy Christian Hutagalung, and
Moh Hasbi Alfarizi. This reproduction is by Moh Hasbi Alfarizi.

Methods: PatchCore (Roth et al., CVPR 2022), GLASS (Chen et al., ECCV 2024,
[code](https://github.com/cqylunlun/GLASS), MIT), CSAD (Hsieh & Lai,
BMVC 2024, [code](https://github.com/Tokichan/CSAD), Apache-2.0).

## Data license

MVTec AD and MVTec LOCO AD are © MVTec Software GmbH, licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). They
are not included in this repository. Models trained on them, and the website,
are for non-commercial use.
