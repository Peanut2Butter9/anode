"""Kaggle notebook (CPU): build the model bundle for the AnoDe website.

Output (/kaggle/working/bundle):
    manifest.json            products, model per product, decision thresholds
    models/glass/<cat>.pth   released GLASS checkpoints (MVTec AD)
    models/csad/<cat>.pth    released CSAD checkpoints (MVTec LOCO AD)
    samples/<cat>/...        a few test images per category (512 px), with masks
    LICENSE-DATA.md          MVTec attribution (CC BY-NC-SA 4.0)

Decision thresholds use the benchmark rule: mean + 3 std of the scores on
held-out normal validation images. For the released GLASS checkpoints those
images were part of the authors' training set, so the threshold is somewhat
low, i.e. conservative (more false alarms rather than missed defects).
"""

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

GLASS_COMMIT = "f788d567db552820d744ca2d93e989265aad8c45"
REPO = Path("/tmp/repos/GLASS")
B = Path("/kaggle/working/bundle")


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


wheel = next(Path("/kaggle/input").rglob("anode-*.whl"))
sh(f"{sys.executable} -m pip install -q --no-deps {wheel}")
sh(f"{sys.executable} -m pip install -q imgaug openpyxl")
sh(f"git clone -q https://github.com/cqylunlun/GLASS {REPO} && git -C {REPO} checkout -q {GLASS_COMMIT}")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from anode.data import MVTEC_AD_CATEGORIES, MVTEC_LOCO_CATEGORIES, load_mask, load_split  # noqa: E402
from anode.external import glass as G  # noqa: E402
from anode.kaggle import dataset_roots  # noqa: E402
from anode.metrics import validation_threshold  # noqa: E402

roots = dataset_roots()
INPUT = Path("/kaggle/input")
# Both the probe (partial download) and anode-external (all 15) have a glass_ckpt folder: take the full one
glass_ckpts = max((p for p in INPUT.rglob("glass_ckpt") if p.is_dir()), key=lambda p: len(list(p.rglob("*.pth"))))
csad_ckpts = next(p for p in INPUT.rglob("csad_ckpt") if p.is_dir())
external_results = next(p for p in INPUT.rglob("results") if (p / "mvtec_loco" / "csad_official").is_dir())
(B / "models" / "glass").mkdir(parents=True, exist_ok=True)
(B / "models" / "csad").mkdir(parents=True, exist_ok=True)
device = torch.device("cpu")
torch.set_num_threads(4)


def export_samples(dataset: str, cat: str, test) -> list[dict]:
    """Up to 6 normal and 6 anomalous test images (spread over defect types), 512 px PNG.
    Not JPEG: CSAD's score shifts by one to three thresholds under JPEG q90 compression."""
    rng = random.Random(0)
    good = [s for s in test if not s.label]
    bad_by_type: dict[str, list] = {}
    for s in test:
        if s.label:
            bad_by_type.setdefault(s.defect_type, []).append(s)
    picks = rng.sample(good, min(6, len(good)))
    types = sorted(bad_by_type)
    i = 0
    while sum(1 for p in picks if p.label) < 6 and any(bad_by_type.values()):
        t = types[i % len(types)]
        if bad_by_type[t]:
            picks.append(bad_by_type[t].pop(rng.randrange(len(bad_by_type[t]))))
        i += 1
    out = []
    d = B / "samples" / cat
    d.mkdir(parents=True, exist_ok=True)
    for j, s in enumerate(picks):
        name = f"{j:02d}_{s.defect_type}"
        with Image.open(s.image) as im:
            im = im.convert("RGB")
            im.thumbnail((512, 512))
            im.save(d / f"{name}.png", optimize=True)
            size = im.size
        entry = {"file": f"samples/{cat}/{name}.png", "label": s.label, "defect_type": s.defect_type,
                 "source": str(s.image.relative_to(roots[dataset]))}
        if s.label:
            Image.fromarray(load_mask(s).astype(np.uint8) * 255).resize(size, Image.NEAREST).save(d / f"{name}_mask.png")
            entry["mask"] = f"samples/{cat}/{name}_mask.png"
        out.append(entry)
    return out


products = []
for cat in MVTEC_AD_CATEGORIES:
    _, val, test = load_split("mvtec_ad", roots["mvtec_ad"], cat, val_fraction=0.1, seed=0)
    ckpt = next(glass_ckpts.rglob(f"mvtec_{cat}/ckpt_best_*.pth"))
    shutil.copy(ckpt, B / "models" / "glass" / f"{cat}.pth")
    model = G.build_model(REPO, device, ckpt=ckpt)
    val_scores, _, _ = G.predict(model, G.GlassTestDataset(val, cat))
    thr = validation_threshold(val_scores)
    products.append({
        "id": cat, "dataset": "mvtec_ad", "category": cat, "model": "glass",
        "model_file": f"models/glass/{cat}.pth", "checkpoint": ckpt.name, "threshold": thr,
        "val_scores": [round(float(x), 6) for x in val_scores], "samples": export_samples("mvtec_ad", cat, test),
    })
    print(cat, "threshold", round(thr, 4), flush=True)
    del model

for cat in MVTEC_LOCO_CATEGORIES:
    _, val, test = load_split("mvtec_loco", roots["mvtec_loco"], cat)
    shutil.copy(next(csad_ckpts.rglob(f"{cat}.pth")), B / "models" / "csad" / f"{cat}.pth")
    val_scores = np.load(external_results / "mvtec_loco" / "csad_official" / cat / "seed0" / "val_scores.npy")
    thr = validation_threshold(val_scores)
    products.append({
        "id": cat, "dataset": "mvtec_loco", "category": cat, "model": "csad",
        "model_file": f"models/csad/{cat}.pth", "threshold": thr,
        "val_scores": [round(float(x), 6) for x in val_scores], "samples": export_samples("mvtec_loco", cat, test),
    })
    print(cat, "threshold", round(thr, 4), flush=True)

manifest = {"version": 1, "glass_commit": GLASS_COMMIT, "csad_commit": "d46cf856b99bf056088b5f510fd984e04813583a",
            "threshold_rule": "mean + 3 std of validation normal scores", "products": products}
(B / "manifest.json").write_text(json.dumps(manifest, indent=1))
(B / "LICENSE-DATA.md").write_text(
    "Sample images and masks are from MVTec AD (Bergmann et al., IJCV 2021) and MVTec LOCO AD "
    "(Bergmann et al., IJCV 2022), (c) MVTec Software GmbH, licensed CC BY-NC-SA 4.0 "
    "(https://creativecommons.org/licenses/by-nc-sa/4.0/). Images were downscaled to 512 px.\n\n"
    "GLASS checkpoints: Chen et al., ECCV 2024, https://github.com/cqylunlun/GLASS (MIT).\n"
    "CSAD checkpoints: Hsieh & Lai, BMVC 2024, https://github.com/Tokichan/CSAD.\n"
    "Both were trained on the MVTec datasets above, so the same non-commercial terms apply.\n")
# One archive of the samples: hundreds of small files are slow to fetch from Kaggle one by one
shutil.make_archive("/kaggle/working/samples", "zip", root_dir=B, base_dir="samples")
total = sum(f.stat().st_size for f in B.rglob("*") if f.is_file())
print(f"bundle: {len(products)} products, {total / 1e6:.0f} MB", flush=True)
