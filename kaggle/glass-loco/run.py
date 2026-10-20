"""Kaggle notebook (GPU): our own GLASS training on MVTec LOCO AD with the
official code. 50 epochs instead of the official 640, because a T4 needs
~100 s per epoch. The official test evaluation runs every 5 epochs, only so we
can measure how optimistic test-set checkpoint selection is. We report the
last epoch."""

import shlex
import subprocess
import sys
from pathlib import Path

GLASS_COMMIT = "f788d567db552820d744ca2d93e989265aad8c45"
EPOCHS, EVAL_EVERY = 50, 5
REPO = Path("/tmp/repos/GLASS")


def sh(cmd: str, check: bool = True) -> int:
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode:
        raise RuntimeError(f"exit {r.returncode}: {cmd}")
    return r.returncode


wheel = next(Path("/kaggle/input").rglob("anode-*.whl"))
sh(f"{sys.executable} -m pip install -q --no-deps {wheel}")
sh(f"{sys.executable} -m pip install -q imgaug openpyxl")
sh(f"git clone -q https://github.com/cqylunlun/GLASS {REPO} && git -C {REPO} checkout -q {GLASS_COMMIT}")

from anode.kaggle import dataset_roots, summarize  # noqa: E402

root = dataset_roots()["mvtec_loco"]
dtd = next(p for p in Path("/kaggle/input/datasets").rglob("images") if p.is_dir() and any(p.glob("*/*.jpg")))
OUT = Path("/kaggle/working/results")
q = shlex.quote
sh(f"{sys.executable} -m anode.external.run glass-train --repo {q(str(REPO))} --dataset mvtec_loco --root {q(str(root))} "
   f"--dtd {q(str(dtd))} --work /tmp/glass --epochs {EPOCHS} --eval-every {EVAL_EVERY} --out {q(str(OUT))}", check=False)
summarize(OUT)
