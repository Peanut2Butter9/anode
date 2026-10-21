"""Kaggle notebook (GPU, T4 x2): timing probe for training CSAD from scratch
with the official code (``anode.external.csad_train``). It runs shortened
schedules on one category and times them, so the full retraining (120
segmentor epochs and 60,000 LGST iterations per category) can be planned
within the Kaggle GPU quota. None of the scores here are results.

1. LGST alone on GPU 0 (500 iterations): the single-job speed.
2. Segmentor (4 epochs, test log at epoch 2) on GPU 0 and LGST (1,000
   iterations) on GPU 1 at the same time: the speed with both GPUs busy,
   which is how the full run would go.
3. Export of both stages and scoring with ``csad-official``, to check that
   the full pipeline runs end to end.
Also prints the per-category sizes that scale the segmentor epoch time."""

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CSAD_COMMIT = "d46cf856b99bf056088b5f510fd984e04813583a"
REPO = Path("/tmp/repos/CSAD")
CATEGORY = "pushpins"
OUT = Path("/kaggle/working/probe")
q = shlex.quote


def sh(cmd: str, check: bool = True, env: dict | None = None) -> int:
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, env=None if env is None else {**os.environ, **env})
    if check and r.returncode:
        raise RuntimeError(f"exit {r.returncode}: {cmd}")
    return r.returncode


def stage(args: str, gpu: int, log: str) -> subprocess.Popen:
    cmd = f"{sys.executable} -m anode.external.csad_train {args} --repo {q(str(REPO))} --out {q(str(OUT))}"
    print(f"$ [GPU {gpu}] {cmd} > {log}", flush=True)
    return subprocess.Popen(cmd, shell=True, stdout=open(OUT / log, "w"), stderr=subprocess.STDOUT,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})


def wait(*procs: subprocess.Popen) -> None:
    for p in procs:
        p.wait()
    for log in sorted(OUT.glob("*.log")):
        print(f"----- {log.name} (tail)\n" + "".join(log.read_text().splitlines(True)[-25:]), flush=True)
    bad = [p.args for p in procs if p.returncode]
    if bad:
        raise RuntimeError(f"failed: {bad}")


wheel = next(Path("/kaggle/input").rglob("anode-*.whl"))
sh(f"{sys.executable} -m pip install -q --no-deps {wheel}")
sh(f"{sys.executable} -m pip install -q empatches tensorboardX")
sh(f"git clone -q https://github.com/Tokichan/CSAD {REPO} && git -C {REPO} checkout -q {CSAD_COMMIT}")
sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv")

from anode.kaggle import dataset_roots  # noqa: E402

root = dataset_roots()["mvtec_loco"]
masks = next(Path("/kaggle/input").rglob("csad_masks")) / "masks"
print("masks:", masks, sorted(p.name for p in masks.iterdir()), flush=True)
OUT.mkdir(parents=True, exist_ok=True)

sizes = {}
for cat in sorted(p.name for p in masks.iterdir() if (p / "info").is_dir()):
    n_train = len(list((root / cat / "train" / "good").glob("*.png")))
    n_sup = len(np.load(masks / cat / "info" / "filtered_histogram_indices.npy"))
    sizes[cat] = {"n_train": n_train, "n_sup": n_sup, "n_unsup": n_train - n_sup,
                  "n_val": len(list((root / cat / "validation").rglob("*.png"))),
                  "n_test": len(list((root / cat / "test").rglob("*.png"))),
                  "mask_dirs": sum(1 for d in (masks / cat).iterdir() if d.is_dir() and d.name != "info")}
print(json.dumps(sizes, indent=1), flush=True)
(OUT / "sizes.json").write_text(json.dumps(sizes, indent=1))

sh(f"{sys.executable} -m anode.external.csad_train prepare --repo {q(str(REPO))} --root {q(str(root))} "
   f"--masks {q(str(masks))} --category {CATEGORY}")

t0 = time.time()
wait(stage(f"lgst --category {CATEGORY} --iterations 500 --seed 0", 0, "1_lgst_alone.log"))
print(f"phase 1: {time.time() - t0:.0f} s", flush=True)

t0 = time.time()
wait(stage(f"segmentor --category {CATEGORY} --epochs 4 --eval-every 2", 0, "2_segmentor.log"),
     stage(f"lgst --category {CATEGORY} --iterations 1000 --seed 1", 1, "2_lgst_parallel.log"))
print(f"phase 2: {time.time() - t0:.0f} s", flush=True)

t0 = time.time()
sh(f"{sys.executable} -m anode.external.csad_train export --repo {q(str(REPO))} --category {CATEGORY} --out {q(str(OUT))}")
sh(f"{sys.executable} -m anode.external.run csad-official --repo {q(str(REPO))} --ckpts {q(str(REPO / 'ckpt/pytorch_models'))} "
   f"--root {q(str(root))} --out {q(str(OUT / 'pipeline_check'))} --categories {CATEGORY}")
print(f"phase 3: {time.time() - t0:.0f} s", flush=True)

for f in sorted(OUT.glob("timing_*.json")):
    info = json.loads(f.read_text())
    for k in ("epoch_s", "loss"):
        info.pop(k, None)
    print(f.name, json.dumps(info), flush=True)
