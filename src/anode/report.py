"""Aggregate benchmark runs into tables (mean ± std over seeds).

Example:
    python -m anode.report results/kaggle --out results/tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import MVTEC_AD_CATEGORIES, MVTEC_LOCO_CATEGORIES

MODEL_ORDER = ["autoencoder_l2", "resnet50_knn", "feature_ae_resnet50", "feature_ae_res2net50", "patchcore",
               "glass", "glass_testselected", "glass_official", "csad", "csad_official"]
MODEL_LABELS = {
    "autoencoder_l2": "AE (L2)",
    "resnet50_knn": "ResNet50+kNN",
    "feature_ae_resnet50": "Feature AE (ResNet-50)",
    "feature_ae_res2net50": "Feature AE (Res2Net-50)",
    "patchcore": "PatchCore",
    "glass": "GLASS (ours, last epoch)",
    "glass_testselected": "GLASS (ours, test-selected)",
    "glass_official": "GLASS (released ckpt)",
    "csad": "CSAD (ours)",
    "csad_official": "CSAD (released ckpt)",
}


def _val_metrics(v: dict | None) -> dict:
    if not v:  # no held-out normal images for this model (released checkpoint)
        return {"val_accuracy": None, "val_f1": None, "val_fpr": None}
    return {"val_accuracy": v["accuracy"], "val_f1": v["f1"], "val_fpr": v["fp"] / max(1, v["fp"] + v["tn"])}


def load_runs(root: Path) -> pd.DataFrame:
    """One row per (dataset, model, category, seed) from every metrics.json under ``root``."""
    rows = []
    for f in sorted(Path(root).rglob("metrics.json")):
        m = json.loads(f.read_text())
        rows.append({
            "dataset": m["dataset"], "model": m["model"], "category": m["category"], "seed": m["seed"],
            "image_auroc": m["image_auroc"],
            "auroc_logical": m.get("image_auroc_logical"), "auroc_structural": m.get("image_auroc_structural"),
            "pixel_auroc": m.get("pixel_auroc"), "aupro_30": m.get("aupro_30"),
            "f1_max": m["f1_max"],
            **_val_metrics(m.get("val_threshold")),
            "fit_s": m["fit_seconds"], "ms_per_image": m["latency"]["ms_per_image_mean"], "device": m["device"],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        # The LOCO convention is the mean of the logical and structural AUROCs
        loco = df["dataset"] == "mvtec_loco"
        df.loc[loco, "auroc_loco_mean"] = (df.loc[loco, "auroc_logical"] + df.loc[loco, "auroc_structural"]) / 2
    return df


def _fmt(mean: float, std: float, n: int) -> str:
    if pd.isna(mean):
        return "–"
    return f"{mean:.3f}" if n <= 1 or pd.isna(std) else f"{mean:.3f} ± {std:.3f}"


def table(df: pd.DataFrame, dataset: str, metric: str) -> pd.DataFrame:
    """Categories x models, each cell "mean ± std" over seeds, plus a "mean" row.

    The mean row first averages each seed over categories, then takes mean ± std
    over seeds. So its std is the seed-to-seed spread of the dataset average.
    """
    d = df[(df["dataset"] == dataset) & df[metric].notna()]
    cats = MVTEC_AD_CATEGORIES if dataset == "mvtec_ad" else MVTEC_LOCO_CATEGORIES
    models = [m for m in MODEL_ORDER if m in set(d["model"])]
    out = pd.DataFrame(index=[*cats, "mean"], columns=[MODEL_LABELS[m] for m in models], dtype=object)
    for m in models:
        dm = d[d["model"] == m]
        g = dm.groupby("category")[metric].agg(["mean", "std", "count"])
        for c in cats:
            if c in g.index:
                out.loc[c, MODEL_LABELS[m]] = _fmt(*g.loc[c])
        if set(cats) <= set(g.index):
            per_seed = dm.groupby("seed")[metric].mean()
            out.loc["mean", MODEL_LABELS[m]] = _fmt(per_seed.mean(), per_seed.std(), len(per_seed))
    return out


def compare_original(df: pd.DataFrame, original_csv: Path) -> pd.DataFrame:
    """Our image AUROC (mean over seeds) next to the original report's AUROC."""
    orig = pd.read_csv(original_csv)
    ours = (df.groupby(["dataset", "model", "category"])["image_auroc"].mean().rename("auroc_ours").reset_index())
    loco_mean = df[df["dataset"] == "mvtec_loco"].groupby(["dataset", "model", "category"])["auroc_loco_mean"].mean()
    ours = ours.set_index(["dataset", "model", "category"])
    ours.loc[loco_mean.index, "auroc_ours"] = loco_mean  # LOCO: mean of logical and structural
    merged = orig.merge(ours.reset_index(), on=["dataset", "model", "category"], how="left")
    merged["delta"] = merged["auroc_ours"] - merged["auroc"]
    return merged.rename(columns={"auroc": "auroc_original"})[
        ["dataset", "model", "category", "auroc_original", "auroc_ours", "delta"]]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path, help="folder searched recursively for metrics.json")
    p.add_argument("--out", type=Path, default=Path("results/tables"))
    p.add_argument("--original", type=Path, default=Path("docs/original_results.csv"))
    a = p.parse_args(argv)

    df = load_runs(a.results)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "runs.csv", index=False)

    tables = {
        "mvtec_ad_image_auroc": ("mvtec_ad", "image_auroc"),
        "mvtec_ad_pixel_auroc": ("mvtec_ad", "pixel_auroc"),
        "mvtec_ad_aupro": ("mvtec_ad", "aupro_30"),
        "mvtec_ad_val_f1": ("mvtec_ad", "val_f1"),
        "mvtec_ad_f1_max": ("mvtec_ad", "f1_max"),
        "mvtec_loco_auroc_mean": ("mvtec_loco", "auroc_loco_mean"),
        "mvtec_loco_auroc_logical": ("mvtec_loco", "auroc_logical"),
        "mvtec_loco_auroc_structural": ("mvtec_loco", "auroc_structural"),
        "mvtec_loco_val_f1": ("mvtec_loco", "val_f1"),
        "latency_ms_mvtec_ad": ("mvtec_ad", "ms_per_image"),
    }
    md = []
    for name, (dataset, metric) in tables.items():
        t = table(df, dataset, metric)
        if t.empty or t.isna().all().all():
            continue
        t.to_csv(a.out / f"{name}.csv")
        md.append(f"### {name}\n\n{t.fillna('–').to_markdown()}\n")
    if a.original.exists():
        cmp = compare_original(df, a.original)
        cmp.to_csv(a.out / "compare_original.csv", index=False)
        summary = cmp.dropna(subset=["auroc_ours"]).groupby(["dataset", "model"])[
            ["auroc_original", "auroc_ours", "delta"]].mean().round(3)
        md.append(f"### compare_original (mean over categories)\n\n{summary.to_markdown()}\n")
    (a.out / "tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
