"""Kaggle notebook (GPU): our own GLASS training on three MVTec AD categories (zipper, wood,
transistor: the categories of the original notebooks) with the
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

root = dataset_roots()["mvtec_ad"]
dtd = next(p for p in Path("/kaggle/input/datasets").rglob("images") if p.is_dir() and any(p.glob("*/*.jpg")))
# Foreground masks released by the GLASS authors (used by the official MVTec setting, fg=1)
FG_ID = "1Fn84QCfMtgBGEDcmY44v97Ci8wwpABK8"  # All_fg_mask.tar.gz
sh(f"{sys.executable} -m pip install -q gdown")
sh(f"mkdir -p /tmp/fg && gdown {FG_ID} -O /tmp/fg/fg.tar.gz --quiet && tar -xzf /tmp/fg/fg.tar.gz -C /tmp/fg")
fg_root = next(p.parent for p in Path("/tmp/fg").rglob("zipper") if p.is_dir() and any(p.glob("*.png")))
print("fg masks:", fg_root, sorted(x.name for x in fg_root.iterdir())[:20], flush=True)
OUT = Path("/kaggle/working/results")
q = shlex.quote  # the mask folder is called "MVTec AD" (with a space)
sh(f"{sys.executable} -m anode.external.run glass-train --repo {q(str(REPO))} --dataset mvtec_ad --root {q(str(root))} "
   f"--categories zipper wood transistor --dtd {q(str(dtd))} --fg-root {q(str(fg_root))} --work /tmp/glass "
   f"--epochs {EPOCHS} --eval-every {EVAL_EVERY} --out {q(str(OUT))}", check=False)
summarize(OUT)
