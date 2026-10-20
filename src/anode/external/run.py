"""Run the official GLASS / CSAD code and write results in the AnoDe format.

    python -m anode.external.run glass-official --repo GLASS --ckpts DIR --root MVTEC_AD --out results
    python -m anode.external.run csad-official  --repo CSAD  --ckpts DIR --root MVTEC_LOCO --out results
    python -m anode.external.run glass-train    --repo GLASS --dataset mvtec_loco --root ROOT --dtd DTD \\
        --work /tmp/glass --epochs 640 --out results [--categories ...] [--fg-root FG]

Model names in the results: ``glass_official`` and ``csad_official`` (the
authors' released checkpoints, which their training scripts selected on the
test set), ``glass`` (our training, last-epoch checkpoint), and
``glass_testselected`` (the same run, with the epoch the official script picks
on the test set; for measuring that selection's optimism only).
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ..benchmark import evaluate, pick_device, seed_everything, write_run
from ..data import DATASETS, load_split


def _env(device) -> dict:
    return {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else device.type,
            "torch": torch.__version__, "python": platform.python_version()}


def _latency(times_ms: list[float], warmup: int = 5) -> dict:
    t = times_ms[warmup:] or times_ms
    return {"ms_per_image_mean": float(np.mean(t)), "ms_per_image_std": float(np.std(t)), "n": len(t)}


def _glass_latency(model, ds, n: int = 30, warmup: int = 5) -> dict:
    times = []
    for i in range(min(len(ds), n + warmup)):
        x = ds[i]["image"].unsqueeze(0)
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model._predict(x)
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return _latency(times, warmup)


def glass_official(a) -> None:
    from . import glass as G
    device = pick_device(a.device)
    for cat in a.categories or DATASETS["mvtec_ad"]:
        run_dir = a.out / "mvtec_ad" / "glass_official" / cat / "seed0"
        if (run_dir / "metrics.json").exists():
            continue
        ckpts = sorted(Path(a.ckpts).rglob(f"mvtec_{cat}/ckpt_best_*.pth"))
        if not ckpts:
            print(f"glass_official {cat}: no checkpoint, skipped", flush=True)
            continue
        _, _, test = load_split("mvtec_ad", a.root, cat, val_fraction=0.1, seed=0)
        model = G.build_model(a.repo, device, ckpt=ckpts[0])
        ds = G.GlassTestDataset(test, cat)
        t0 = time.perf_counter()
        scores, maps, masks = G.predict(model, ds)
        result = {
            "dataset": "mvtec_ad", "category": cat, "model": "glass_official", "seed": 0,
            "image_size": G.IMAGESIZE, "n_test": len(test), "fit_seconds": 0.0,
            "test_seconds_total": time.perf_counter() - t0, "latency": _glass_latency(model, ds),
            "checkpoint": ckpts[0].name, "best_epoch": int(re.findall(r"\d+", ckpts[0].stem)[-1]),
            "note": "released checkpoint; the official training script keeps the epoch with the best test AUROC",
            **_env(device),
            # Trained on the full training set, so no held-out normal images exist: no validation threshold
            **evaluate("mvtec_ad", test, scores, maps, None, G.IMAGESIZE, masks=masks),
        }
        write_run(run_dir, a.root, test, result, scores, maps, None)


def csad_official(a) -> None:
    from . import csad as C
    device = pick_device(a.device)
    for cat in a.categories or DATASETS["mvtec_loco"]:
        run_dir = a.out / "mvtec_loco" / "csad_official" / cat / "seed0"
        if (run_dir / "metrics.json").exists():
            continue
        ckpt = next(Path(a.ckpts).rglob(f"{cat}.pth"))
        _, val, test = load_split("mvtec_loco", a.root, cat)
        model = C.load_model(a.repo, ckpt, device)
        t0 = time.perf_counter()
        test_map, times = C.score_split(model, a.repo, a.root, cat, "test", device)
        test_s = time.perf_counter() - t0
        val_map, _ = C.score_split(model, a.repo, a.root, cat, "eval", device)
        test_scores = np.array([test_map[str(s.image.resolve())] for s in test])
        val_scores = np.array([val_map[str(s.image.resolve())] for s in val])
        result = {
            "dataset": "mvtec_loco", "category": cat, "model": "csad_official", "seed": 0, "image_size": 256,
            "n_val": len(val), "n_test": len(test), "fit_seconds": 0.0, "test_seconds_total": test_s,
            "latency": _latency(times), "checkpoint": ckpt.name,
            "note": "released checkpoint; the official training script keeps the iteration with the best test AUROC. "
                    "CSAD also uses the validation split to normalise its scores.",
            **_env(device),
            **evaluate("mvtec_loco", test, test_scores, None, val_scores, 256),
        }
        write_run(run_dir, a.root, test, result, test_scores, None, val_scores)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def glass_train(a) -> None:
    from . import glass as G
    device = pick_device(a.device)
    loco = a.dataset == "mvtec_loco"
    for cat in a.categories or DATASETS[a.dataset]:
        for seed in a.seeds:
            run_dir = a.out / a.dataset / "glass" / cat / f"seed{seed}"
            if (run_dir / "metrics.json").exists():
                continue
            seed_everything(seed)
            train, val, test = load_split(a.dataset, a.root, cat, val_fraction=0.1, seed=0)
            view = G.make_view(train, test, cat, a.work / "view" / a.dataset, square=G.IMAGESIZE if loco else None,
                               fg_root=None if loco else a.fg_root)
            name = f"mvtec_loco_{cat}" if loco else f"mvtec_{cat}"  # "mvtec_<cat>" finds the authors' spreadsheet
            models_dir = a.work / "models" / a.dataset / f"seed{seed}"
            shutil.rmtree(models_dir / name, ignore_errors=True)
            t0 = time.perf_counter()
            model, ckpts, svd = G.train(a.repo, device, view, cat, a.dtd, models_dir, name, epochs=a.epochs,
                                        fg=0 if loco else 1, eval_every=a.eval_every)
            fit_s = time.perf_counter() - t0
            test_ds = G.GlassTestDataset(test, cat, square=loco)
            val_ds = G.GlassTestDataset(val, cat, square=loco)
            # "glass" = last epoch (reported). "glass_testselected" = the official script's test-selected
            # epoch, kept only to measure the optimism of selecting on the test set.
            for model_name, which in (("glass", "last"), ("glass_testselected", "best")):
                G.load_checkpoint(model, ckpts[which], device)
                t0 = time.perf_counter()
                scores, maps, masks = G.predict(model, test_ds)
                test_s = time.perf_counter() - t0
                val_scores, _, _ = G.predict(model, val_ds)
                epoch = a.epochs - 1 if which == "last" else int(re.findall(r"\d+", ckpts["best"].stem)[-1])
                result = {
                    "dataset": a.dataset, "category": cat, "model": model_name, "seed": seed,
                    "image_size": G.IMAGESIZE, "n_train": len(train), "n_val": len(val), "n_test": len(test),
                    "epochs": a.epochs, "checkpoint_epoch": epoch, "eval_every": a.eval_every, "svd": svd,
                    "fit_seconds": fit_s, "seconds_per_epoch": fit_s / a.epochs, "test_seconds_total": test_s,
                    "latency": _glass_latency(model, test_ds),
                    "note": ("official GLASS code, last-epoch checkpoint" if which == "last" else
                             "ANALYSIS ONLY: epoch chosen by the official script on the test set"),
                    **_env(device),
                    **evaluate(a.dataset, test, scores, maps, val_scores, G.IMAGESIZE, masks=masks),
                }
                out_dir = a.out / a.dataset / model_name / cat / f"seed{seed}"
                write_run(out_dir, a.root, test, result, scores, maps, val_scores)
                shutil.copy(ckpts[which], out_dir / "model.pt")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("glass-official", "csad-official", "glass-train"):
        s = sub.add_parser(name)
        s.add_argument("--repo", type=Path, required=True)
        s.add_argument("--root", type=Path, required=True)
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--categories", nargs="*")
        s.add_argument("--device", default="auto")
        if name != "glass-train":
            s.add_argument("--ckpts", type=Path, required=True)
    t = sub.choices["glass-train"]
    t.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    t.add_argument("--dtd", type=Path, required=True, help="DTD images folder (anomaly source for GLASS)")
    t.add_argument("--work", type=Path, required=True, help="scratch folder for views and checkpoints")
    t.add_argument("--fg-root", type=Path, help="GLASS foreground masks (MVTec AD only)")
    t.add_argument("--epochs", type=int, default=640)
    t.add_argument("--seeds", nargs="+", type=int, default=[0])
    t.add_argument("--eval-every", type=int, default=None,
                   help="log the official test evaluation every N epochs (analysis of test-set selection only)")
    a = p.parse_args(argv)
    a.out, a.root, a.repo = a.out.resolve(), a.root.resolve(), a.repo.resolve()
    {"glass-official": glass_official, "csad-official": csad_official, "glass-train": glass_train}[a.cmd](a)


if __name__ == "__main__":
    main()
