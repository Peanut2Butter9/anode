"""Figures for the paper, built from results/tables/runs.csv and the website bundle.

    python paper/figures/make_figures.py

Writes PDF figures next to this script. Colors: the first two slots of the
validated categorical palette (blue #2a78d6, orange #eb6834), muted gray for
context.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "web" / "server"))

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#b8b6b0", "#0b0b0b", "#52514e"
COL = 3.45  # IEEE column width, inches
plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

LABELS = {
    "autoencoder_l2": "Autoencoder (L2)", "resnet50_knn": "ResNet-50 + kNN",
    "feature_ae_resnet50": "Feature AE (ResNet-50)", "feature_ae_res2net50": "Feature AE (Res2Net-50)",
    "patchcore": "PatchCore", "glass": "GLASS (ours, 50 ep.)", "glass_official": "GLASS (released)",
    "csad_official": "CSAD (released)",
}


def runs() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "tables" / "runs.csv")


def fig_auroc(df: pd.DataFrame) -> None:
    """Image AUROC per model: MVTec AD (left), LOCO logical vs structural (right)."""
    order = ["autoencoder_l2", "resnet50_knn", "feature_ae_res2net50", "feature_ae_resnet50", "patchcore", "glass",
             "glass_official", "csad_official"]
    ad = df[df.dataset == "mvtec_ad"].groupby("model")["image_auroc"].mean()
    ad_n = df[df.dataset == "mvtec_ad"].groupby("model")["category"].nunique()
    loco = df[df.dataset == "mvtec_loco"].groupby("model")[["auroc_logical", "auroc_structural"]].mean()
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL * 2, 2.1), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(order))
    vals = [ad.get(m, np.nan) if ad_n.get(m, 0) == 15 else np.nan for m in order]
    why = {"glass": "3 of 15 categories (0.995)", "csad_official": "LOCO only"}
    a.barh(y, [v - 0.5 if not np.isnan(v) else 0 for v in vals], left=0.5, height=0.6, color=BLUE)
    for yi, v in zip(y, vals):
        a.text(v + 0.006 if not np.isnan(v) else 0.505, yi, f"{v:.3f}" if not np.isnan(v) else why[order[yi]],
               va="center", fontsize=6.5, color=MUTED if np.isnan(v) else INK)
    a.set_xlim(0.5, 1.05)
    a.set_title("MVTec AD: image AUROC", loc="left")
    h = 0.36
    for k, (col, color, name) in enumerate([("auroc_logical", ORANGE, "logical"), ("auroc_structural", BLUE, "structural")]):
        v = [loco[col].get(m, np.nan) for m in order]
        b.barh(y + (k - 0.5) * h, [x - 0.5 if not np.isnan(x) else 0 for x in v], left=0.5, height=h * 0.9,
               color=color, label=name)
    b.text(0.505, order.index("glass_official"), "AD only", va="center", fontsize=6.5, color=MUTED)
    b.set_xlim(0.5, 1.02)
    b.set_title("MVTec LOCO AD: image AUROC", loc="left")
    b.legend(loc="lower right", frameon=False, handlelength=1, borderaxespad=0.2)
    for ax in (a, b):
        ax.grid(axis="y", visible=False)
        ax.set_yticks(y, [LABELS[m] for m in order])
        ax.invert_yaxis()
    fig.savefig(OUT / "auroc.pdf")


def fig_selection(df: pd.DataFrame) -> None:
    """GLASS on LOCO: AUROC gain of the official, test-selected epoch over the last epoch."""
    loco = df[(df.dataset == "mvtec_loco") & df.model.isin(["glass", "glass_testselected"])]
    piv = loco.pivot_table(index="category", columns="model", values=["auroc_logical", "auroc_structural"])
    cats = list(piv.index)
    y = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(COL, 1.75))
    h = 0.36
    for k, (col, color, name) in enumerate([("auroc_logical", ORANGE, "logical"), ("auroc_structural", BLUE, "structural")]):
        gain = piv[(col, "glass_testselected")] - piv[(col, "glass")]
        ax.barh(y + (k - 0.5) * h, gain, height=h * 0.9, color=color, label=name)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, [c.replace("_", " ") for c in cats])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUROC(test-selected epoch) − AUROC(last epoch)")
    ax.set_xticks([-0.05, 0, 0.05, 0.1])
    ax.legend(frameon=False, loc="lower right", handlelength=1)
    fig.savefig(OUT / "selection.pdf")


def fig_calibration() -> None:
    """False alarms vs recall at the label-free validation threshold (MVTec AD)."""
    rows = []
    for f in (ROOT / "results" / "kaggle").rglob("metrics.json"):
        m = json.loads(f.read_text())
        v = m.get("val_threshold")
        if m["dataset"] == "mvtec_ad" and v and m["model"] in LABELS:
            rows.append((m["model"], v["fp"] / max(1, v["fp"] + v["tn"]), v["tp"] / max(1, v["tp"] + v["fn"])))
    d = pd.DataFrame(rows, columns=["model", "fpr", "tpr"]).groupby("model").mean()
    # The released GLASS checkpoints have no unseen normal images: their best label-free threshold
    # (mean + 3 std of the scores of normal training images) evaluated on the test set
    glass_rows = json.loads((OUT / "glass_released_threshold.json").read_text())
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    # Label positions in data coordinates; the three crowded points get thin leader lines
    spots = {"feature_ae_resnet50": (0.2, 0.74), "patchcore": (0.2, 0.835), "glass": (0.2, 0.93),
             "feature_ae_res2net50": (0.095, 0.665), "resnet50_knn": (0.07, 0.38), "autoencoder_l2": (0.115, 0.09)}
    for m, r in d.iterrows():
        ax.scatter(r.fpr, r.tpr, s=22, color=BLUE, zorder=3)
        label = "GLASS, ours (3 categories)" if m == "glass" else LABELS[m]
        tx, ty = spots.get(m, (r.fpr + 0.01, r.tpr))
        leader = m in ("feature_ae_resnet50", "patchcore", "glass")
        ax.annotate(label, (r.fpr, r.tpr), xytext=(tx, ty), fontsize=6, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6, shrinkA=0, shrinkB=3) if leader else None)
    ax.scatter(glass_rows["fpr"], glass_rows["tpr"], s=22, color=ORANGE, zorder=3)
    ax.annotate("GLASS (released)", (glass_rows["fpr"], glass_rows["tpr"]), xytext=(0.335, 0.6), fontsize=6,
                color=INK, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6, shrinkA=0, shrinkB=3))
    ax.set_xlabel("false-alarm rate on good test parts")
    ax.set_ylabel("recall on defective parts")
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 1.04)
    fig.savefig(OUT / "calibration.pdf")


def fig_heatmaps() -> None:
    """PatchCore heatmaps on bundled test samples (requires the local bundle)."""
    from PIL import Image

    from anode_web.heatmap import overlay
    from anode_web.runtimes import PatchCoreRuntime

    manifest = json.loads((ROOT / "bundle" / "manifest.json").read_text())
    rt = PatchCoreRuntime(Path("/tmp/anode-fig"))
    picks = [("bottle", "broken_large"), ("hazelnut", "crack"), ("screw", "thread_top"), ("carpet", "color"),
             ("transistor", "misplaced")]
    fig, axes = plt.subplots(2, len(picks), figsize=(COL * 2, 2.9), gridspec_kw={"wspace": 0.04, "hspace": 0.02})
    for j, (cat, defect) in enumerate(picks):
        p = next(x for x in manifest["products"] if x["id"] == cat)
        s = next((x for x in p["samples"] if x["defect_type"] == defect), next(x for x in p["samples"] if x["label"]))
        im = Image.open(ROOT / "bundle" / s["file"]).convert("RGB")
        pred = rt.predict(cat, [im], bank_file=ROOT / "bundle" / p["model_file"])[0]
        axes[0, j].imshow(pred.view)
        axes[1, j].imshow(overlay(pred.view, pred.amap, p["threshold"]))
        axes[0, j].set_title(f"{cat.replace('_', ' ')}: {s['defect_type'].replace('_', ' ')}", fontsize=7)
        axes[1, j].text(0.5, -0.06, f"score {pred.score / p['threshold']:.2f}× threshold", transform=axes[1, j].transAxes,
                        ha="center", va="top", fontsize=6.5, color=MUTED)
        for i in range(2):
            axes[i, j].set_axis_off()
    fig.savefig(OUT / "heatmaps.pdf", dpi=300)


def fig_spc() -> None:
    """P-chart and score histogram of 30 days of simulated bottle production."""
    from anode import qc
    from anode_web import demo

    manifest = json.loads((ROOT / "bundle" / "manifest.json").read_text())
    p = next(x for x in manifest["products"] if x["id"] == "bottle")
    scores = json.loads((ROOT / "web" / "server" / "anode_web" / "data" / "demo_scores.json").read_text())["bottle"]
    import datetime as dt
    hist = demo.history("bottle", scores, p["threshold"], days=30, today=dt.date(2026, 10, 20))
    pc = qc.p_chart([b["defective"] for b in hist], [b["n"] for b in hist])
    allscores = np.concatenate([b["scores"] for b in hist])
    cap = qc.capability(allscores, usl=1.0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL * 2, 1.9), gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.22})
    x = np.arange(len(hist))
    pts = pc["points"]
    a.step(x, [q["ucl"] for q in pts], where="mid", color=ORANGE, lw=1, ls="--", label="UCL / LCL")
    a.step(x, [q["lcl"] for q in pts], where="mid", color=ORANGE, lw=1, ls="--")
    a.axhline(pc["p_bar"], color=MUTED, lw=1, label=r"$\bar p$")
    a.plot(x, [q["p"] for q in pts], color=BLUE, lw=1.2, marker="o", ms=2.8, label="defect rate")
    ooc = [i for i, q in enumerate(pts) if q["out_of_control"]]
    a.scatter(ooc, [pts[i]["p"] for i in ooc], s=40, facecolor="none", edgecolor=ORANGE, lw=1.2, zorder=4)
    a.set_xticks(x[::5], [hist[i]["date"][5:] for i in x[::5]])
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a.set_title("P-chart, simulated bottle line (25 batches)", loc="left")
    a.legend(frameon=False, ncol=3, loc="upper left", handlelength=1.4)
    a.set_ylim(0, max(q["ucl"] for q in pts) * 1.35)
    bins = np.arange(0, 2.6, 0.1)
    b.hist(np.clip(allscores, 0, 2.5), bins=bins, color=BLUE, edgecolor="white", linewidth=0.5)
    b.set_yscale("log")
    b.set_ylabel("units (log scale)")
    b.axvline(1.0, color=ORANGE, lw=1.2)
    b.text(1.04, b.get_ylim()[1] * 0.85, "USL", color=ORANGE, fontsize=6.5)
    b.set_xlabel("score / threshold")
    b.set_title(f"Cpk = {cap['cpk']:.2f}; yield-based {cap['cpk_from_yield']:.2f}", loc="left")
    b.grid(axis="x", visible=False)
    fig.savefig(OUT / "spc.pdf")
    (OUT / "spc_numbers.json").write_text(json.dumps({"p_bar": pc["p_bar"], "n_batches": len(hist),
                                                      "units": int(sum(b["n"] for b in hist)),
                                                      "out_of_control": len(ooc), **cap}, indent=1))


if __name__ == "__main__":
    df = runs()
    fig_auroc(df)
    fig_selection(df)
    fig_calibration()
    fig_spc()
    fig_heatmaps()
    print("figures:", sorted(p.name for p in OUT.glob("*.pdf")))
