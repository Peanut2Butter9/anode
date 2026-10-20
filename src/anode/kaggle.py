"""Helpers for running the benchmark inside Kaggle notebooks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .data import MVTEC_AD_CATEGORIES, MVTEC_LOCO_CATEGORIES

INPUT = Path("/kaggle/input")


def find_root(base: Path, categories, max_depth: int = 6) -> Path | None:
    """Breadth-first search for the folder that holds every category folder."""
    frontier = [base]
    for _ in range(max_depth):
        nxt = []
        for d in frontier:
            if all((d / c).is_dir() for c in categories):
                return d
            nxt.extend(p for p in sorted(d.iterdir()) if p.is_dir())
        frontier = nxt
    return None


def dataset_roots(base: Path = INPUT) -> dict[str, Path]:
    """Locate MVTec AD (a dataset input) and MVTec LOCO AD (the output of the
    ``anode-data`` notebook, mounted as an input)."""
    roots = {"mvtec_ad": find_root(base, MVTEC_AD_CATEGORIES), "mvtec_loco": find_root(base, MVTEC_LOCO_CATEGORIES)}
    for name, root in roots.items():
        print(f"{name}: {root}")
    return {k: v for k, v in roots.items() if v is not None}


def summarize(results: Path) -> list[dict]:
    """Collect every metrics.json under ``results`` into summary.csv and print it."""
    rows = []
    for f in sorted(Path(results).rglob("metrics.json")):
        m = json.loads(f.read_text())
        rows.append({
            "dataset": m["dataset"], "model": m["model"], "category": m["category"], "seed": m["seed"],
            "image_auroc": m["image_auroc"],
            "auroc_logical": m.get("image_auroc_logical"), "auroc_structural": m.get("image_auroc_structural"),
            "pixel_auroc": m.get("pixel_auroc"), "aupro_30": m.get("aupro_30"),
            "f1_max": m["f1_max"], "val_acc": (m["val_threshold"] or {}).get("accuracy"),
            "val_f1": (m["val_threshold"] or {}).get("f1"),
            "fit_s": m["fit_seconds"], "ms_per_image": m["latency"]["ms_per_image_mean"], "device": m["device"],
        })
    if rows:
        with open(Path(results) / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            print("  ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.values()))
    return rows
