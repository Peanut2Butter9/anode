"""Build the static data files the website ships with, from the benchmark results.

    python web/server/scripts/build_data.py results/kaggle

Writes web/server/anode_web/data/benchmark.json (metrics per dataset, model
and category) and demo_scores.json (test-set scores of the deployed models,
used to simulate production history).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

from anode.report import MODEL_LABELS, MODEL_ORDER, load_runs

OUT = Path(__file__).resolve().parents[1] / "anode_web" / "data"
DEPLOYED = {"mvtec_ad": "patchcore", "mvtec_loco": "csad_official"}
METRICS = ["image_auroc", "auroc_logical", "auroc_structural", "auroc_loco_mean", "pixel_auroc", "aupro_30",
           "val_f1", "f1_max", "ms_per_image"]


def benchmark(df: pd.DataFrame) -> dict:
    out = {"models": {m: MODEL_LABELS[m] for m in MODEL_ORDER if m in set(df["model"])}, "datasets": {}}
    for ds, d in df.groupby("dataset"):
        per_cat = d.groupby(["model", "category"])[METRICS].mean()
        cats = sorted(d["category"].unique())
        models = [m for m in MODEL_ORDER if m in set(d["model"])]
        rows = {}
        for m in models:
            rows[m] = {c: {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in per_cat.loc[(m, c)].items()}
                       for c in cats if (m, c) in per_cat.index}
        means = {}
        for m in models:
            full = d[d["model"] == m]
            if full["category"].nunique() == len(cats):
                means[m] = {k: (None if pd.isna(v) else round(float(v), 4))
                            for k, v in full.groupby("seed")[METRICS].mean().mean().items()}
        out["datasets"][ds] = {"categories": cats, "models": models, "per_category": rows, "mean": means,
                               "deployed": DEPLOYED.get(ds)}
    return out


def demo_scores(results: Path) -> dict:
    out = {}
    for ds, model in DEPLOYED.items():
        for f in sorted(results.rglob(f"{ds}/{model}/*/seed0/scores.csv")):
            cat = f.parent.parent.name
            good, bad = [], []
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    (bad if row["label"] == "1" else good).append(round(float(row["score"]), 6))
            out[cat] = {"good": good, "defective": bad}
    return out


def main() -> None:
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results/kaggle")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "benchmark.json").write_text(json.dumps(benchmark(load_runs(results)), indent=1))
    (OUT / "demo_scores.json").write_text(json.dumps(demo_scores(results)))
    print("wrote", *(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
