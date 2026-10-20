"""Run models on MVTec AD / MVTec LOCO AD and save metrics, scores and anomaly maps.

Example:
    python -m anode.benchmark --dataset mvtec_ad --root /data/mvtec_anomaly_detection \
        --models patchcore resnet50_knn --categories all --seeds 0 --out results
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import metrics as M
from .data import DATASETS, ImageDataset, load_mask, load_split
from .models import MODELS


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def predict_all(model, ds: ImageDataset, batch_size: int = 16):
    scores, maps = [], []
    for b in DataLoader(ds, batch_size=batch_size, num_workers=2):
        s, m = model.predict(b["image"])
        scores.append(s.float().cpu())
        if m is not None:
            maps.append(m.half().cpu())
    scores = torch.cat(scores).numpy()
    maps = torch.cat(maps).float().numpy() if maps else None
    return scores, maps


def measure_latency(model, ds: ImageDataset, n: int = 30, warmup: int = 5) -> dict:
    """Model time per image (batch size 1, preprocessing excluded), after warm-up."""
    times = []
    for i in range(min(len(ds), n + warmup)):
        x = ds[i]["image"].unsqueeze(0)
        _sync(model.device)
        t0 = time.perf_counter()
        model.predict(x)
        _sync(model.device)
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return {"ms_per_image_mean": float(np.mean(times)), "ms_per_image_std": float(np.std(times)), "n": len(times)}


def evaluate(dataset: str, test, test_scores, test_maps, val_scores, size: int, masks=None) -> dict:
    """Metrics for one run. ``masks`` (N, H, W) overrides the masks loaded at
    ``size``, for models with their own preprocessing (e.g. a center crop).
    ``val_scores`` may be None when no held-out normal images exist for the
    model (e.g. a released checkpoint trained on the full training set)."""
    labels = np.array([s.label for s in test])
    types = np.array([s.defect_type for s in test])
    f1max, f1max_thr = M.f1_max(labels, test_scores)
    out = {
        "image_auroc": M.image_auroc(labels, test_scores),
        "f1_max": f1max,
        "f1_max_threshold": f1max_thr,
        "val_threshold": None,
        "n_val": 0,
        "per_defect_auroc": {},
    }
    if val_scores is not None:
        thr = M.validation_threshold(val_scores)
        out["val_threshold"] = M.threshold_metrics(labels, test_scores, thr)
        out["n_val"] = int(len(val_scores))
    good = types == "good"
    for t in sorted(set(types) - {"good"}):
        sel = good | (types == t)
        out["per_defect_auroc"][t] = M.image_auroc(labels[sel], test_scores[sel])

    if dataset == "mvtec_loco":
        # The LOCO convention reports logical and structural anomalies separately
        out["image_auroc_logical"] = out["per_defect_auroc"].get("logical_anomalies")
        out["image_auroc_structural"] = out["per_defect_auroc"].get("structural_anomalies")
        # sPRO needs the full-resolution masks and defects_config.json; it is
        # computed later from the saved maps (anode.evaluate_loco)
    elif test_maps is not None:
        if masks is None:
            masks = np.stack([load_mask(s, (size, size)) for s in test])
        out["pixel_auroc"] = M.pixel_auroc(masks, test_maps)
        out["aupro_30"] = M.aupro(masks, test_maps, fpr_limit=0.3)
    return out


def write_run(run_dir: Path, root: Path, test, result: dict, test_scores, test_maps=None, val_scores=None) -> None:
    """Write metrics.json, scores.csv, val_scores.npy and (optionally) maps.npz, then print a summary."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "label", "defect_type", "score"])
        for s, sc in zip(test, test_scores):
            w.writerow([str(s.image.relative_to(root)), s.label, s.defect_type, f"{sc:.6g}"])
    if val_scores is not None:
        np.save(run_dir / "val_scores.npy", val_scores)
    if test_maps is not None:
        np.savez_compressed(run_dir / "maps.npz", maps=test_maps.astype(np.float16))
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    summary = f"{result['dataset']}/{result['category']} {result['model']} seed{result['seed']}: " \
              f"I-AUROC {result['image_auroc']:.4f}"
    if "pixel_auroc" in result:
        summary += f"  P-AUROC {result['pixel_auroc']:.4f}  AUPRO {result['aupro_30']:.4f}"
    if result.get("image_auroc_logical") is not None:
        summary += f"  (logical {result['image_auroc_logical']:.4f}, structural {result['image_auroc_structural']:.4f})"
    print(summary + f"  fit {result.get('fit_seconds', 0):.0f}s  {result['latency']['ms_per_image_mean']:.1f} ms/img",
          flush=True)


def run_one(dataset: str, root: Path, category: str, model_name: str, seed: int, out: Path,
            device: torch.device, size: int = 256, save_maps: bool = True, save_model: bool = False,
            model_kwargs: dict | None = None) -> dict:
    run_dir = out / dataset / model_name / category / f"seed{seed}"
    if (run_dir / "metrics.json").exists():
        print(f"skip {run_dir} (done)")
        return json.loads((run_dir / "metrics.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    train, val, test = load_split(dataset, root, category, val_fraction=0.1, seed=0)
    model = MODELS[model_name](device=device, seed=seed, **(model_kwargs or {}))
    norm = model.normalize_input
    train_ds = ImageDataset(train, size, norm)
    val_ds = ImageDataset(val, size, norm)
    test_ds = ImageDataset(test, size, norm)

    _sync(device)
    t0 = time.perf_counter()
    model.fit(train_ds)
    _sync(device)
    fit_s = time.perf_counter() - t0

    val_scores, _ = predict_all(model, val_ds)
    _sync(device)
    t0 = time.perf_counter()
    test_scores, test_maps = predict_all(model, test_ds)
    _sync(device)
    test_s = time.perf_counter() - t0

    result = {
        "dataset": dataset, "category": category, "model": model_name, "seed": seed,
        "image_size": size, "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "fit_seconds": fit_s, "test_seconds_total": test_s,
        "latency": measure_latency(model, test_ds),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else device.type,
        "torch": torch.__version__, "python": platform.python_version(),
        **evaluate(dataset, test, test_scores, test_maps, val_scores, size),
    }
    if hasattr(model, "loss_history"):
        result["loss_history"] = model.loss_history
    write_run(run_dir, root, test, result, test_scores, test_maps if save_maps else None, val_scores)
    if save_model:
        torch.save(model.state_dict(), run_dir / "model.pt")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--root", type=Path, required=True, help="folder that contains the category folders")
    p.add_argument("--models", nargs="+", choices=sorted(MODELS), required=True)
    p.add_argument("--categories", nargs="+", default=["all"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-maps", action="store_true", help="do not save anomaly maps")
    p.add_argument("--save-model", action="store_true", help="save fitted model state for deployment")
    a = p.parse_args(argv)

    cats = DATASETS[a.dataset] if a.categories == ["all"] else a.categories
    device = pick_device(a.device)
    print(f"device: {device}")
    for model_name in a.models:
        for cat in cats:
            for seed in a.seeds:
                run_one(a.dataset, a.root, cat, model_name, seed, a.out, device, a.size,
                        save_maps=not a.no_maps, save_model=a.save_model)


if __name__ == "__main__":
    main()
