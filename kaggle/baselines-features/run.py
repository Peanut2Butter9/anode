"""Kaggle notebook: the feature-based baselines on MVTec AD and MVTec LOCO AD.

- ResNet-50 + kNN and PatchCore: not trained and deterministic (apart from
  PatchCore's coreset start point), so one seed.
- Feature AE (the report's "Res2Net"), with ResNet-50 and Res2Net-50
  backbones: trained, so three seeds.
"""

import subprocess
import sys
from pathlib import Path

wheel = next(Path("/kaggle/input").rglob("anode-*.whl"))
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-q", str(wheel)], check=True)

from anode.benchmark import main  # noqa: E402
from anode.kaggle import dataset_roots, summarize  # noqa: E402

OUT = "/kaggle/working/results"
for dataset, root in dataset_roots().items():
    args = ["--dataset", dataset, "--root", str(root), "--out", OUT]
    main(args + ["--models", "resnet50_knn"])
    # PatchCore memory banks are small (~60 MB per category), so keep them for the website
    main(args + ["--models", "patchcore", "--save-model"])
    main(args + ["--models", "feature_ae_resnet50", "feature_ae_res2net50", "--seeds", "0", "1", "2"])
summarize(OUT)
