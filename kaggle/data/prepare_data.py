"""Prepare and check MVTec AD and MVTec LOCO AD on Kaggle.

MVTec AD comes from the Kaggle dataset ``mhasbiarz/mvtec-anomaly-detection``
(our own upload from April 2025, already extracted).
MVTec LOCO AD comes from ``rakibulhassannishat/mvtec-loco-ad``, which ships the
original ``.tar.xz`` archive. It is extracted into /kaggle/working so later
notebooks can mount this notebook's output instead of extracting it again.

Every category and split is counted and compared with the image counts
published in the dataset papers. The script exits non-zero on any mismatch.
"""

import json
import platform
import shutil
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

from PIL import Image

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Bergmann et al., IJCV 2021, Table 1: (train good, test good, test defective)
MVTEC_AD = {
    "bottle": (209, 20, 63),
    "cable": (224, 58, 92),
    "capsule": (219, 23, 109),
    "carpet": (280, 28, 89),
    "grid": (264, 21, 57),
    "hazelnut": (391, 40, 70),
    "leather": (245, 32, 92),
    "metal_nut": (220, 22, 93),
    "pill": (267, 26, 141),
    "screw": (320, 41, 119),
    "tile": (230, 33, 84),
    "toothbrush": (60, 12, 30),
    "transistor": (213, 60, 40),
    "wood": (247, 19, 60),
    "zipper": (240, 32, 119),
}

# Bergmann et al., IJCV 2022, Table 1:
# (train, validation, test good, test logical, test structural)
# Exception: the paper lists 354 train / 59 validation images for
# splicing_connectors, but the released archive has 360 / 60 (3,651 images in
# total, not 3,644). We check against the released archive.
MVTEC_LOCO = {
    "breakfast_box": (351, 62, 102, 83, 90),
    "juice_bottle": (335, 54, 94, 142, 94),
    "pushpins": (372, 69, 138, 91, 81),
    "screw_bag": (360, 60, 122, 137, 82),
    "splicing_connectors": (360, 60, 119, 108, 85),
}


def images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)


def find_root(base: Path, categories, max_depth: int = 5) -> Path | None:
    """Breadth-first search for the directory that holds every category folder."""
    frontier = [base]
    for _ in range(max_depth):
        nxt = []
        for d in frontier:
            if all((d / c).is_dir() for c in categories):
                return d
            nxt.extend(p for p in d.iterdir() if p.is_dir())
        frontier = nxt
    return None


def find_file(base: Path, name: str) -> Path | None:
    return next(iter(base.rglob(name)), None)


def image_sizes(paths: list[Path], n: int = 5) -> list[list[int]]:
    sizes = Counter()
    for p in paths[:n]:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            sizes[im.size] += 1
    return [list(s) for s in sizes]


def check_mvtec_ad(root: Path) -> tuple[dict, list[str]]:
    report, errors = {}, []
    for cat, (n_train, n_good, n_defect) in MVTEC_AD.items():
        train = images(root / cat / "train")
        test = images(root / cat / "test")
        good = images(root / cat / "test" / "good")
        masks = images(root / cat / "ground_truth")
        got = {
            "train": len(train),
            "test_good": len(good),
            "test_defective": len(test) - len(good),
            "masks": len(masks),
            "defect_types": sorted(p.name for p in (root / cat / "test").iterdir() if p.is_dir() and p.name != "good"),
            "image_sizes": image_sizes(train),
        }
        want = {"train": n_train, "test_good": n_good, "test_defective": n_defect, "masks": n_defect}
        for k, v in want.items():
            if got[k] != v:
                errors.append(f"MVTec AD {cat}/{k}: expected {v}, found {got[k]}")
        report[cat] = got
    return report, errors


def check_mvtec_loco(root: Path) -> tuple[dict, list[str]]:
    report, errors = {}, []
    for cat, expected in MVTEC_LOCO.items():
        c = root / cat
        train = images(c / "train" / "good")
        got = {
            "train": len(train),
            "validation": len(images(c / "validation" / "good")),
            "test_good": len(images(c / "test" / "good")),
            "test_logical": len(images(c / "test" / "logical_anomalies")),
            "test_structural": len(images(c / "test" / "structural_anomalies")),
            # One folder of masks per anomalous test image
            "gt_logical_dirs": sum(p.is_dir() for p in (c / "ground_truth" / "logical_anomalies").iterdir()),
            "gt_structural_dirs": sum(p.is_dir() for p in (c / "ground_truth" / "structural_anomalies").iterdir()),
            "defects_config": (c / "defects_config.json").is_file(),
            "image_sizes": image_sizes(train),
        }
        keys = ["train", "validation", "test_good", "test_logical", "test_structural"]
        want = dict(zip(keys, expected))
        want["gt_logical_dirs"] = want["test_logical"]
        want["gt_structural_dirs"] = want["test_structural"]
        for k, v in want.items():
            if got[k] != v:
                errors.append(f"MVTec LOCO {cat}/{k}: expected {v}, found {got[k]}")
        if not got["defects_config"]:
            errors.append(f"MVTec LOCO {cat}: defects_config.json missing")
        report[cat] = got
    return report, errors


def environment() -> dict:
    env = {"python": platform.python_version()}
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda
    except ImportError:
        env["torch"] = None
    try:
        import torchvision

        env["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    total, used, free = shutil.disk_usage(WORKING)
    env["working_disk_free_gb"] = round(free / 1e9, 1)
    return env


def main() -> int:
    print("Mounted inputs:", sorted(str(p) for p in INPUT.rglob("*") if p.is_dir() and len(p.relative_to(INPUT).parts) <= 3))

    ad_root = find_root(INPUT, MVTEC_AD)
    if ad_root is None:
        print("MVTec AD root not found under /kaggle/input")
        return 1
    print("MVTec AD root:", ad_root)

    loco_root = find_root(WORKING, MVTEC_LOCO, max_depth=3)
    if loco_root is None:
        archive = find_file(INPUT, "mvtec_loco_anomaly_detection.tar.xz")
        if archive is None:
            print("MVTec LOCO archive not found under /kaggle/input")
            return 1
        print(f"Extracting {archive} ({archive.stat().st_size / 1e9:.2f} GB) ...", flush=True)
        t0 = time.time()
        with tarfile.open(archive, "r:xz") as tar:
            tar.extractall(WORKING / "mvtec_loco", filter="data")
        print(f"Extracted in {time.time() - t0:.0f} s")
        loco_root = find_root(WORKING / "mvtec_loco", MVTEC_LOCO)
    print("MVTec LOCO root:", loco_root)

    ad_report, ad_errors = check_mvtec_ad(ad_root)
    loco_report, loco_errors = check_mvtec_loco(loco_root)
    errors = ad_errors + loco_errors

    report = {
        "environment": environment(),
        "mvtec_ad_root": str(ad_root),
        "mvtec_loco_root": str(loco_root),
        "mvtec_ad": ad_report,
        "mvtec_loco": loco_report,
        "totals": {
            "mvtec_ad_images": sum(r["train"] + r["test_good"] + r["test_defective"] for r in ad_report.values()),
            "mvtec_loco_images": sum(
                r["train"] + r["validation"] + r["test_good"] + r["test_logical"] + r["test_structural"]
                for r in loco_report.values()
            ),
        },
        "errors": errors,
    }
    (WORKING / "dataset_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("ALL CHECKS PASSED" if not errors else f"{len(errors)} CHECK(S) FAILED")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
