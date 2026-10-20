"""MVTec AD and MVTec LOCO AD datasets."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

MVTEC_AD_CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)
MVTEC_LOCO_CATEGORIES = ("breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors")
DATASETS = {"mvtec_ad": MVTEC_AD_CATEGORIES, "mvtec_loco": MVTEC_LOCO_CATEGORIES}


@dataclass(frozen=True)
class Sample:
    image: Path
    label: int  # 0 = normal, 1 = anomalous
    defect_type: str  # "good", a defect name, "logical_anomalies" or "structural_anomalies"
    masks: tuple[Path, ...] = ()  # ground-truth masks; empty for normal images


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.is_dir() else []


def mvtec_ad_samples(root: Path, category: str, split: str) -> list[Sample]:
    """``split`` is "train" (normal only) or "test"."""
    base = Path(root) / category / split
    samples = []
    for defect_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        defect = defect_dir.name
        for img in _images(defect_dir):
            if defect == "good":
                samples.append(Sample(img, 0, defect))
            else:
                mask = Path(root) / category / "ground_truth" / defect / f"{img.stem}_mask.png"
                samples.append(Sample(img, 1, defect, (mask,)))
    return samples


def mvtec_loco_samples(root: Path, category: str, split: str) -> list[Sample]:
    """``split`` is "train", "validation" (both normal only) or "test".

    Each anomalous LOCO test image has a folder of masks, one per defect region.
    The pixel value of a mask encodes the defect type (see defects_config.json).
    """
    base = Path(root) / category / split
    samples = []
    for defect_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        defect = defect_dir.name
        for img in _images(defect_dir):
            if defect == "good":
                samples.append(Sample(img, 0, defect))
            else:
                masks = tuple(_images(Path(root) / category / "ground_truth" / defect / img.stem))
                samples.append(Sample(img, 1, defect, masks))
    return samples


def load_split(dataset: str, root: Path, category: str, val_fraction: float = 0.1, seed: int = 0):
    """Return (train, validation, test) sample lists.

    MVTec AD has no validation split, so ``val_fraction`` of the normal training
    images is held out (with a fixed seed). The held-out images are used only to
    pick the decision threshold, never for training. MVTec LOCO AD ships an
    official validation split.
    """
    if dataset == "mvtec_ad":
        train = mvtec_ad_samples(root, category, "train")
        rng = random.Random(seed)
        idx = list(range(len(train)))
        rng.shuffle(idx)
        n_val = max(1, round(val_fraction * len(train)))
        val_idx = set(idx[:n_val])
        val = [s for i, s in enumerate(train) if i in val_idx]
        train = [s for i, s in enumerate(train) if i not in val_idx]
        test = mvtec_ad_samples(root, category, "test")
    elif dataset == "mvtec_loco":
        train = mvtec_loco_samples(root, category, "train")
        val = mvtec_loco_samples(root, category, "validation")
        test = mvtec_loco_samples(root, category, "test")
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return train, val, test


def load_mask(sample: Sample, size: tuple[int, int] | None = None) -> np.ndarray:
    """Union of a sample's ground-truth masks as a bool array (H, W)."""
    if not sample.masks:
        with Image.open(sample.image) as im:
            w, h = im.size
        mask = np.zeros((h, w), dtype=bool)
    else:
        mask = None
        for p in sample.masks:
            with Image.open(p) as im:
                m = np.asarray(im.convert("L")) > 0
            mask = m if mask is None else mask | m
    if size is not None:
        mask = np.asarray(Image.fromarray(mask).resize((size[1], size[0]), Image.NEAREST))
    return mask


class ImageDataset(Dataset):
    """Resizes to ``size`` x ``size`` (no center crop, so defects near the border
    are kept). With ``normalize`` the image is ImageNet-normalized for pretrained
    backbones; otherwise it is scaled to [0, 1]."""

    def __init__(self, samples: list[Sample], size: int = 256, normalize: bool = True, with_masks: bool = False):
        self.samples = samples
        self.size = size
        self.normalize = normalize
        self.with_masks = with_masks
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        with Image.open(s.image) as im:
            im = im.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
            x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
        if self.normalize:
            x = (x - self.mean) / self.std
        item = {"image": x, "label": s.label, "index": i}
        if self.with_masks:
            item["mask"] = torch.from_numpy(load_mask(s, (self.size, self.size)))
        return item
