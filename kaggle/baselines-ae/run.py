"""Kaggle notebook: the image autoencoder (L2) on MVTec AD and MVTec LOCO AD,
three seeds per category."""

import subprocess
import sys
from pathlib import Path

wheel = next(Path("/kaggle/input").rglob("anode-*.whl"))
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-q", str(wheel)], check=True)

from anode.benchmark import main  # noqa: E402
from anode.kaggle import dataset_roots, summarize  # noqa: E402

OUT = "/kaggle/working/results"
for dataset, root in dataset_roots().items():
    main(["--dataset", dataset, "--root", str(root), "--models", "autoencoder_l2", "--seeds", "0", "1", "2",
          "--out", OUT])
summarize(OUT)
