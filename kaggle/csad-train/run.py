"""Kaggle notebook (GPU, T4 x2): our CSAD training on MVTec LOCO AD with the
official code (``anode.external.csad_train``), seed 0, one session.

The official schedule (60,000 LGST iterations per category) would need ~14 h
per category on a T4 (1.2 it/s), so the LGST is cut to 12,000 iterations
(StepLR still drops at 95 %). The segmentors run their full 120 epochs. Both
stages keep their last state (model ``csad``). The test set is logged every
2,000 LGST iterations and 10 segmentor epochs, only to measure how optimistic
the official test-set selection is (model ``csad_testselected``).

GPU 0: LGST breakfast_box, juice_bottle, pushpins.
GPU 1: the 5 segmentors, then LGST screw_bag, splicing_connectors.
Each category is exported and scored as soon as both of its stages are done.
A failed job never stops the script (Kaggle keeps no output from a failed
run), and no LGST starts unless it can finish within the session budget."""

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

CSAD_COMMIT = "d46cf856b99bf056088b5f510fd984e04813583a"
SMOKE = False  # True: one category with tiny schedules, to check the pipeline before the real run
ITERATIONS, LGST_EVAL_EVERY, SEG_EPOCHS, SEG_EVAL_EVERY, SEED = 12000, 2000, None, 10, 0
LANES = {0: ["breakfast_box", "juice_bottle", "pushpins"], 1: ["screw_bag", "splicing_connectors"]}
CATEGORIES = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
if SMOKE:
    ITERATIONS, LGST_EVAL_EVERY, SEG_EPOCHS, SEG_EVAL_EVERY = 300, 100, 4, 2
    LANES, CATEGORIES = {0: ["pushpins"], 1: []}, ["pushpins"]
BUDGET_H, LGST_H = 10.0, 3.0  # session budget (GPU quota left ~10.7 h) and one LGST job with margin
REPO, TS = Path("/tmp/repos/CSAD"), Path("/tmp/csad_testselected")
OUT = Path("/kaggle/working")
q = shlex.quote
T0 = time.time()
status: dict[str, str] = {}
seg_done = {c: threading.Event() for c in CATEGORIES}


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    if subprocess.run(cmd, shell=True).returncode:
        raise RuntimeError(cmd)


def job(name: str, args: str, gpu: int) -> bool:
    cmd = f"{sys.executable} -m {args}"
    log = OUT / "logs" / f"{name}.log"
    print(f"[{(time.time() - T0) / 3600:5.2f} h] GPU {gpu} start {name}", flush=True)
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, shell=True, stdout=fh, stderr=subprocess.STDOUT,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}).returncode
    status[name] = "ok" if rc == 0 else f"failed ({rc})"
    print(f"[{(time.time() - T0) / 3600:5.2f} h] GPU {gpu} {status[name]} {name} in {time.time() - t0:.0f} s", flush=True)
    if rc:
        print("".join(log.read_text().splitlines(True)[-30:]), flush=True)
    return rc == 0


def train(stage: str, cat: str, gpu: int, extra: str) -> bool:
    return job(f"{stage}_{cat}", f"anode.external.csad_train {stage} --repo {q(str(REPO))} --category {cat} "
                                 f"--seed {SEED} --ts-dir {q(str(TS))} --out {q(str(OUT / 'timings'))} {extra}", gpu)


def finish(cat: str, gpu: int) -> None:
    """Export the last and the test-selected states and score both."""
    seg_done[cat].wait()
    if status.get(f"segmentor_{cat}") != "ok":
        return
    for model, src in (("csad", REPO), ("csad_testselected", TS)):
        exported = job(f"export_{model}_{cat}", f"anode.external.csad_train export --repo {q(str(REPO))} --category {cat} "
                                                f"--from-dir {q(str(src))}", gpu)
        if exported:
            job(f"score_{model}_{cat}",
                f"anode.external.run csad-official --model {model} --seed {SEED} --repo {q(str(REPO))} "
                f"--ckpts {q(str(src / 'ckpt' / 'pytorch_models'))} --root {q(str(root))} --out {q(str(OUT / 'results'))} "
                f"--timings {q(str(OUT / 'timings'))} --categories {cat}", gpu)
    if status.get(f"export_csad_{cat}") == "ok":  # keep our last model (deployable like the released one)
        (OUT / "ckpt").mkdir(exist_ok=True)
        sh(f"cp {q(str(REPO / 'ckpt' / 'pytorch_models' / (cat + '.pth')))} {q(str(OUT / 'ckpt'))}/")


def lane(gpu: int) -> None:
    if gpu == 1:
        for cat in CATEGORIES:
            train("segmentor", cat, gpu, f"--eval-every {SEG_EVAL_EVERY}" + (f" --epochs {SEG_EPOCHS}" if SEG_EPOCHS else ""))
            seg_done[cat].set()
    for cat in LANES[gpu]:
        if (time.time() - T0) / 3600 + LGST_H > BUDGET_H:
            status[f"lgst_{cat}"] = "skipped (session budget)"
            print(f"GPU {gpu} skipped lgst_{cat}: not enough session time left", flush=True)
            continue
        if train("lgst", cat, gpu, f"--iterations {ITERATIONS} --eval-every {LGST_EVAL_EVERY}"):
            finish(cat, gpu)


try:
    wheel = next(Path("/kaggle/input").rglob("anode-*.whl"))
    sh(f"{sys.executable} -m pip install -q --no-deps {wheel}")
    sh(f"{sys.executable} -m pip install -q empatches tensorboardX")
    sh(f"git clone -q https://github.com/Tokichan/CSAD {REPO} && git -C {REPO} checkout -q {CSAD_COMMIT}")
    from anode.kaggle import dataset_roots, summarize

    root = dataset_roots()["mvtec_loco"]
    masks = next(Path("/kaggle/input").rglob("csad_masks")) / "masks"
    for d in ("logs", "timings", "results"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES:
        sh(f"{sys.executable} -m anode.external.csad_train prepare --repo {q(str(REPO))} --root {q(str(root))} "
           f"--masks {q(str(masks))} --category {cat}")
    threads = [threading.Thread(target=lane, args=(g,)) for g in LANES]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    summarize(OUT / "results")
except Exception as e:  # noqa: BLE001 - keep the outputs whatever happens
    status["setup"] = f"failed: {e}"
    print("setup failed:", e, flush=True)
(OUT / "status.json").write_text(json.dumps({"hours": (time.time() - T0) / 3600, **status}, indent=1))
print(json.dumps(status, indent=1), flush=True)
