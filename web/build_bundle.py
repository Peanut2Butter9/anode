"""Assemble the model bundle the website serves.

    python web/build_bundle.py --patchcore DIR --csad DIR --results results/kaggle --out bundle

- MVTec AD products use PatchCore (our implementation). These models were
  trained on our split, so their threshold comes from genuinely held-out normal
  images. The released GLASS checkpoints were trained on every normal
  training image and scored new normal images much higher than their own
  training images, so no label-free threshold worked for them (about 33%
  false alarms). See docs/ROADMAP.md.
- MVTec LOCO AD products use the released CSAD checkpoints. CSAD never trains
  on the validation split, so that split calibrates the threshold.

The sample images and their metadata come from the Kaggle bundle notebook
(kaggle/bundle/run.py), which must already be in ``--out``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from anode.metrics import validation_threshold


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patchcore", type=Path, required=True, help="folder with <category>/seed0/model.pt")
    p.add_argument("--csad", type=Path, required=True, help="folder with <category>.pth")
    p.add_argument("--results", type=Path, default=Path("results/kaggle"))
    p.add_argument("--out", type=Path, default=Path("bundle"))
    a = p.parse_args()

    manifest = json.loads((a.out / "manifest.json").read_text())
    (a.out / "models" / "patchcore").mkdir(parents=True, exist_ok=True)
    (a.out / "models" / "csad").mkdir(parents=True, exist_ok=True)
    products = []
    for prod in manifest["products"]:
        cat, ds = prod["category"], prod["dataset"]
        if ds == "mvtec_ad":
            model, src = "patchcore", next(a.patchcore.rglob(f"{cat}/seed0/model.pt"))
            dst = f"models/patchcore/{cat}.pt"
            val = np.load(next(a.results.rglob(f"mvtec_ad/patchcore/{cat}/seed0/val_scores.npy")))
        else:
            model, src = "csad", next(a.csad.rglob(f"{cat}.pth"))
            dst = f"models/csad/{cat}.pth"
            val = np.load(next(a.results.rglob(f"mvtec_loco/csad_official/{cat}/seed0/val_scores.npy")))
        if not (a.out / dst).exists():
            shutil.copy(src, a.out / dst)
        products.append({**{k: v for k, v in prod.items() if k != "checkpoint"}, "model": model, "model_file": dst,
                         "threshold": validation_threshold(val), "val_scores": [round(float(x), 6) for x in val]})
        print(f"{cat:20s} {model:9s} threshold {products[-1]['threshold']:.4f}")
    shutil.rmtree(a.out / "models" / "glass", ignore_errors=True)
    manifest.update(version=2, products=products,
                    models={"mvtec_ad": "patchcore (ours)", "mvtec_loco": "csad (released checkpoint)"})
    manifest.pop("glass_commit", None)
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (a.out / "LICENSE-DATA.md").write_text(
        "Sample images and masks are from MVTec AD (Bergmann et al., IJCV 2021) and MVTec LOCO AD "
        "(Bergmann et al., IJCV 2022), (c) MVTec Software GmbH, licensed CC BY-NC-SA 4.0 "
        "(https://creativecommons.org/licenses/by-nc-sa/4.0/). Images were downscaled to 512 px.\n\n"
        "PatchCore memory banks: our implementation (Roth et al., CVPR 2022), built from MVTec AD training images.\n"
        "CSAD checkpoints: Hsieh & Lai, BMVC 2024, https://github.com/Tokichan/CSAD.\n"
        "All models were trained on the MVTec datasets above, so the same non-commercial terms apply.\n")
    total = sum(f.stat().st_size for f in a.out.rglob("*") if f.is_file())
    print(f"bundle: {len(products)} products, {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
