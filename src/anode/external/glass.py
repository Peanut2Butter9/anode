"""GLASS (Chen et al., ECCV 2024) via its official code, inside the AnoDe protocol.

Official repository: https://github.com/cqylunlun/GLASS (pinned commit in
``GLASS_COMMIT``). We use the authors' model class and training loop
unchanged. Only data preparation and evaluation are ours:

- ``GlassTestDataset`` reproduces the official test-time preprocessing
  (resize the shorter side, center crop to 288, ImageNet normalisation),
  including the authors' special 329-px resize for toothbrush and wood.
  Ground-truth masks get the same geometry, so pixel metrics are computed
  exactly where the model sees the image.
- ``make_view`` builds the MVTec-style folder that the official data loader
  expects, from our own train/test split (and squares LOCO images first,
  since the official center crop would cut off the sides of LOCO's wide
  images).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..data import IMAGENET_MEAN, IMAGENET_STD, Sample, load_mask

GLASS_COMMIT = "f788d567db552820d744ca2d93e989265aad8c45"
IMAGESIZE = 288

# shell/run-mvtec.sh of the official repository
GLASS_ARGS = dict(
    layers_to_extract_from=["layer2", "layer3"], pretrain_embed_dimension=1536, target_embed_dimension=1536,
    patchsize=3, meta_epochs=640, eval_epochs=1, dsc_layers=2, dsc_hidden=1024, dsc_margin=0.5,
    train_backbone=False, pre_proj=1, mining=1, noise=0.015, radius=0.75, p=0.5, lr=0.0001, svd=0, step=20,
    limit=392,
)
DATASET_ARGS = dict(distribution=0, mean=0.5, std=0.1, fg=1, rand_aug=1, batch_size=8)


def _numpy2_shim() -> None:
    """imgaug 0.4.0 (imported by GLASS's perlin.py) uses ``np.sctypes``,
    which NumPy 2 removed. Restore it before importing imgaug."""
    if not hasattr(np, "sctypes"):
        np.sctypes = {"int": [np.int8, np.int16, np.int32, np.int64],
                      "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
                      "float": [np.float16, np.float32, np.float64], "complex": [np.complex64, np.complex128],
                      "others": [bool, object, bytes, str, np.void]}


def import_glass(repo: Path, with_data: bool = False):
    """Import the official modules. The code reads ``./datasets/excel/...``
    relative to the repository, so the working directory changes too.
    The data loader (``with_data``) is only needed for training, and it pulls
    in imgaug."""
    _numpy2_shim()
    repo = Path(repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.chdir(repo)
    import backbones
    import glass
    mvtec = None
    if with_data:
        from datasets import mvtec
    return glass, backbones, mvtec


def build_model(repo: Path, device, ckpt: Path | None = None, **overrides):
    glass, backbones, _ = import_glass(repo)
    model = glass.GLASS(device)
    model.load(backbone=backbones.load("wideresnet50"), device=device, input_shape=(3, IMAGESIZE, IMAGESIZE),
               **{**GLASS_ARGS, **overrides})
    if ckpt is not None:
        load_checkpoint(model, ckpt, device)
    return model


def load_checkpoint(model, ckpt: Path, device) -> None:
    sd = torch.load(ckpt, map_location=device)
    model.discriminator.load_state_dict(sd["discriminator"])
    if "pre_projection" in sd:
        model.pre_projection.load_state_dict(sd["pre_projection"])


def resize_for(category: str) -> int:
    """The official loader resizes toothbrush and wood to 329 px before the 288 crop."""
    return round(IMAGESIZE * 329 / 288) if category in ("toothbrush", "wood") else IMAGESIZE


class GlassTestDataset(Dataset):
    """``square=True`` resizes straight to 288 x 288 (no crop), matching the
    squared LOCO views that GLASS is trained on."""

    def __init__(self, samples: list[Sample], category: str, square: bool = False):
        r = (IMAGESIZE, IMAGESIZE) if square else resize_for(category)
        self.samples = samples
        self.tf_img = transforms.Compose([transforms.Resize(r), transforms.CenterCrop(IMAGESIZE), transforms.ToTensor(),
                                          transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.tf_mask = transforms.Compose([transforms.Resize(r), transforms.CenterCrop(IMAGESIZE),
                                           transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        with Image.open(s.image) as im:
            x = self.tf_img(im.convert("RGB"))
        mask = Image.fromarray(load_mask(s).astype(np.uint8) * 255)
        return {"image": x, "mask": self.tf_mask(mask)[0] > 0.5, "index": i}


@torch.no_grad()
def predict(model, ds: GlassTestDataset, batch_size: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image scores (N,), anomaly maps (N, 288, 288) and the matching masks."""
    scores, maps, masks = [], [], []
    for b in DataLoader(ds, batch_size=batch_size, num_workers=2):
        s, m = model._predict(b["image"])
        scores += [float(np.asarray(x)) for x in s]
        maps += [np.asarray(x, dtype=np.float32) for x in m]
        masks.append(b["mask"].numpy())
    return np.array(scores), np.stack(maps), np.concatenate(masks)


def make_view(samples_train: list[Sample], samples_test: list[Sample], category: str, out: Path,
              square: int | None = None, fg_root: Path | None = None) -> Path:
    """MVTec-style folder for the official loader, built from our split.

    Images are symlinked, or written as ``square`` x ``square`` PNGs when
    ``square`` is set (for LOCO). Each anomalous test image gets one merged
    ``<stem>_mask.png``. The official loader pairs images and masks by sorted
    order, so the names must sort the same way.
    """
    c = Path(out) / category

    def put(src: Path, dst: Path, mask: np.ndarray | None = None):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mask is not None:
            im = Image.fromarray(mask.astype(np.uint8) * 255)
            if square:
                im = im.resize((square, square), Image.NEAREST)
            im.save(dst)
        elif square:
            with Image.open(src) as im:
                im.convert("RGB").resize((square, square), Image.BILINEAR).save(dst)
        elif not dst.exists():
            dst.symlink_to(src)

    for s in samples_train:
        put(s.image, c / "train" / "good" / s.image.name)
    for s in samples_test:
        put(s.image, c / "test" / s.defect_type / s.image.name)
        if s.label:
            put(s.image, c / "ground_truth" / s.defect_type / f"{s.image.stem}_mask.png", mask=load_mask(s))
    if fg_root is not None and (Path(fg_root) / category).is_dir():
        fg = Path(out) / "fg_mask" / category
        if not fg.exists():
            fg.parent.mkdir(parents=True, exist_ok=True)
            fg.symlink_to(Path(fg_root) / category)
    return Path(out)


def train(repo: Path, device, view_root: Path, category: str, dtd: Path, models_dir: Path, name: str,
          epochs: int = 640, fg: int = 1, num_workers: int = 4, eval_every: int | None = None):
    """Train with the official loop and return two checkpoints:

    - ``last``: the final epoch (``ckpt.pth``). This is what we report as GLASS.
    - ``best``: the official script's choice (``ckpt_best_<epoch>.pth``), i.e.
      the epoch with the highest test image AUROC + pixel AUROC, checked every
      ``eval_every`` epochs. It is kept only to measure how optimistic
      test-set selection is, and is never reported as the model's result.

    Classes missing from the authors' distribution spreadsheet (all of LOCO)
    first go through the official "judge" pass, which picks manifold or
    hypersphere training from the average image's spectrum.
    """
    eval_every = eval_every or epochs
    glass_mod, backbones, mvtec = import_glass(repo, with_data=True)

    def loaders(distribution: int):
        kw = dict(DATASET_ARGS, distribution=distribution, fg=fg)
        tr = mvtec.MVTecDataset(str(view_root), str(dtd), dataset_name="mvtec", classname=category, resize=IMAGESIZE,
                                imagesize=IMAGESIZE, split=mvtec.DatasetSplit.TRAIN, **kw)
        te = mvtec.MVTecDataset(str(view_root), str(dtd), classname=category, resize=IMAGESIZE, imagesize=IMAGESIZE,
                                split=mvtec.DatasetSplit.TEST)
        dl = dict(num_workers=num_workers, prefetch_factor=2, pin_memory=True)
        return (DataLoader(tr, batch_size=DATASET_ARGS["batch_size"], shuffle=True, **dl),
                DataLoader(te, batch_size=DATASET_ARGS["batch_size"], shuffle=False, **dl))

    model = build_model(repo, device, meta_epochs=epochs, eval_epochs=eval_every)
    model.set_model_dir(str(models_dir), name)
    train_dl, test_dl = loaders(distribution=0)
    out = model.trainer(train_dl, test_dl, name)
    svd = None
    if isinstance(out, (int, np.integer)):  # judge pass: 0 = manifold, 1 = hypersphere
        svd = int(out)
        model = build_model(repo, device, meta_epochs=epochs, eval_epochs=eval_every)
        model.set_model_dir(str(models_dir), name)
        train_dl, test_dl = loaders(distribution=3 if svd else 2)
        model.trainer(train_dl, test_dl, name)
    else:
        svd = int(model.svd)
    ckpt_dir = Path(models_dir, name)
    best = next(ckpt_dir.glob("ckpt_best_*.pth"))
    return model, {"last": ckpt_dir / "ckpt.pth", "best": best}, svd
