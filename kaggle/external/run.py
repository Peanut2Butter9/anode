"""Kaggle notebook (GPU): official GLASS and CSAD checkpoints in the AnoDe protocol.

1. Clone the official repositories at pinned commits.
2. Download the 15 released GLASS checkpoints one file at a time (Google Drive
   throttles popular files, so one failure must not stop the others).
   The CSAD checkpoints come from the anode-glass-csad-probe notebook output.
3. Evaluate glass_official on MVTec AD and csad_official on MVTec LOCO AD.
4. Time a 3-epoch GLASS training run on one LOCO category, to plan the budget.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPOS = Path("/tmp/repos")
OUT = WORK / "results"
GLASS_COMMIT = "f788d567db552820d744ca2d93e989265aad8c45"
CSAD_COMMIT = "d46cf856b99bf056088b5f510fd984e04813583a"
# results/models/backbone_0/mvtec_<class>/<file> in the authors' Drive folder
GLASS_CKPTS = {
    "bottle": ["ckpt_best_543.pth", "1lwdgpG8u7MqoaNqyQn2fo0u0Gd7Qd8Db"],
    "cable": ["ckpt_best_72.pth", "153eHCetjGK30d3wU3Mir_IApVWgEy1Cp"],
    "capsule": ["ckpt_best_56.pth", "1J5kNCsyt9frM9nWA_x6K6vcqf2NfEygF"],
    "carpet": ["ckpt_best_431.pth", "1bNht25rvPjADjbd6m844ZB3JAFvHguDN"],
    "grid": ["ckpt_best_581.pth", "1ZuQxDlespjAQmerLOsIXw2we6tMJAile"],
    "hazelnut": ["ckpt_best_182.pth", "1otUn9YjxpgB51JHQxspvGVLlP_dsS71D"],
    "leather": ["ckpt_best_119.pth", "1TnB7aP6JQMqoxA3d1FzQA_gkifp1imgA"],
    "metal_nut": ["ckpt_best_5.pth", "16uUx_EXMNVk-G0APetbMVe0t1XI78dpJ"],
    "pill": ["ckpt_best_233.pth", "1Sif5D32PJ7w5TPKf2ySD0vtjS0BeG5Hh"],
    "screw": ["ckpt_best_139.pth", "1sUV4c8IssF89rTMo1uaSMjK8Hu5VB7JP"],
    "tile": ["ckpt_best_369.pth", "1s6p91LbJy6mPtwmlL4pxqWKVxO0aPo-G"],
    "toothbrush": ["ckpt_best_543.pth", "1EsBx26Ay8_21_4AoJP-LP3SKWWXKNuzV"],
    "transistor": ["ckpt_best_52.pth", "1GeUVIzyby35M3vHy5-CC2hX8Fsuf32FO"],
    "wood": ["ckpt_best_637.pth", "1iEyBlhvIRWS6l82f1Gg22imDYeMx7qOD"],
    "zipper": ["ckpt_best_149.pth", "1fCaDTVeOT74W5mSoaWCa2E1mDf-0skcJ"],
}


def sh(cmd: str, check: bool = True) -> int:
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode:
        raise RuntimeError(f"exit {r.returncode}: {cmd}")
    return r.returncode


wheel = next(Path("/kaggle/input").rglob("anode-*.whl"))
sh(f"{sys.executable} -m pip install -q --no-deps {wheel}")
sh(f"{sys.executable} -m pip install -q gdown imgaug openpyxl einops empatches")
for name, url, commit in (("GLASS", "https://github.com/cqylunlun/GLASS", GLASS_COMMIT),
                          ("CSAD", "https://github.com/Tokichan/CSAD", CSAD_COMMIT)):
    if not (REPOS / name).exists():
        sh(f"git clone -q {url} {REPOS / name} && git -C {REPOS / name} checkout -q {commit}")

from anode.kaggle import dataset_roots, summarize  # noqa: E402

roots = dataset_roots()
INPUT = Path("/kaggle/input")
csad_ckpts = next(INPUT.rglob("csad_ckpt"))
dtd = next(p for p in INPUT.rglob("images") if p.is_dir() and any(p.glob("*/*.jpg")))
print("csad checkpoints:", csad_ckpts, "| dtd:", dtd, flush=True)

# GLASS checkpoints, one file at a time with retries
glass_ckpts = WORK / "glass_ckpt"
status = {}
for cat, (fname, fid) in GLASS_CKPTS.items():
    dst = glass_ckpts / f"mvtec_{cat}" / fname
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        if dst.exists() and dst.stat().st_size > 0:
            break
        if sh(f"gdown {fid} -O {dst} --quiet", check=False):
            time.sleep(20)
    status[cat] = dst.exists() and dst.stat().st_size > 0
print("GLASS checkpoints downloaded:", status, flush=True)
(WORK / "glass_ckpt_status.json").write_text(json.dumps(status, indent=2))

run = f"{sys.executable} -m anode.external.run"
sh(f"{run} glass-official --repo {REPOS / 'GLASS'} --ckpts {glass_ckpts} --root {roots['mvtec_ad']} --out {OUT}",
   check=False)
sh(f"{run} csad-official --repo {REPOS / 'CSAD'} --ckpts {csad_ckpts} --root {roots['mvtec_loco']} --out {OUT}",
   check=False)
# Timing only: 3 epochs on one LOCO category (results go to a separate folder)
sh(f"{run} glass-train --repo {REPOS / 'GLASS'} --dataset mvtec_loco --root {roots['mvtec_loco']} --dtd {dtd} "
   f"--work /tmp/glass --epochs 3 --categories pushpins --out {WORK / 'timing'}", check=False)
summarize(OUT)
summarize(WORK / "timing")
