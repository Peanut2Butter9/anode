"""CPU inference for the two model families the website uses.

- PatchCore (MVTec AD products and custom products): one WideResNet-50
  backbone shared by all products. Each product only swaps in its memory bank.
  Bundled banks come from the benchmark. Custom products are built on the
  server from the user's normal images, with 20% held out to set the threshold.
- CSAD (MVTec LOCO AD products): one full model per product, cached (LRU).
  CSAD gives an image score but no anomaly map.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from anode.data import IMAGENET_MEAN, IMAGENET_STD
from anode.metrics import validation_threshold

DEVICE = torch.device("cpu")
NORM = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


@dataclass
class Prediction:
    score: float
    view: Image.Image  # the image as the model saw it (after resize/crop)
    amap: np.ndarray | None  # anomaly map in the coordinates of ``view``
    latency_ms: float


@contextlib.contextmanager
def _no_pretrained_download():
    """CSAD's constructor asks timm for ImageNet weights from the Hugging Face
    Hub, but the checkpoint overwrites them anyway. Build the same
    architecture locally, without the 260 MB download or a Hub request."""
    import timm
    original = timm.create_model

    def create(name, *args, **kwargs):
        kwargs["pretrained"] = False
        if isinstance(name, str) and name.startswith("hf_hub:timm/"):
            name = name.removeprefix("hf_hub:timm/").split(".")[0]  # e.g. wide_resnet50_2
        return original(name, *args, **kwargs)

    timm.create_model = create
    try:
        yield
    finally:
        timm.create_model = original


class CsadRuntime:
    def __init__(self, repo: Path, cache_size: int = 2):
        from anode.external import csad as C
        self.C, self.repo = C, repo
        C.import_csad(repo, with_data=False)
        self.cache: OrderedDict[str, torch.nn.Module] = OrderedDict()
        self.cache_size = cache_size
        self.lock = threading.Lock()
        self.tf = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR)

    def _model(self, pid: str, model_file: Path) -> torch.nn.Module:
        if pid in self.cache:
            self.cache.move_to_end(pid)
            return self.cache[pid]
        with _no_pretrained_download():
            model = self.C.load_model(self.repo, model_file, DEVICE)
        self.cache[pid] = model
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return model

    @torch.no_grad()
    def predict(self, product: dict, model_file: Path, images: list[Image.Image]) -> list[Prediction]:
        out = []
        with self.lock:
            model = self._model(product["id"], model_file)
            for im in images:
                view = self.tf(im.convert("RGB"))
                x = NORM(transforms.functional.to_tensor(view)).unsqueeze(0)
                t0 = time.perf_counter()
                score = float(model(x).item())
                out.append(Prediction(score, view, None, (time.perf_counter() - t0) * 1000))
        return out


class _ImageList(Dataset):
    def __init__(self, images: list[Image.Image], size: int = 256):
        self.x = [NORM(transforms.functional.to_tensor(im.convert("RGB").resize((size, size), Image.BILINEAR)))
                  for im in images]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"image": self.x[i], "label": 0, "index": i}


@dataclass
class CustomProduct:
    id: str
    name: str
    created: float
    threshold: float
    n_train: int
    n_val: int
    val_scores: list[float]
    state: dict = field(repr=False)  # PatchCore memory bank


class PatchCoreRuntime:
    """Custom products. Models live in memory and in ``data_dir`` (lost when
    the Space restarts, since free Spaces have no persistent disk)."""

    MAX_PRODUCTS = 50

    def __init__(self, data_dir: Path):
        from anode.models import patchcore
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model = patchcore(device=DEVICE)
        self.model.num_workers = 0
        self.products: OrderedDict[str, CustomProduct] = OrderedDict()
        self.banks: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.lock = threading.RLock()
        for f in sorted(self.data_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime):
            try:
                cp = torch.load(f, weights_only=False)
                self.products[cp.id] = cp
            except Exception:
                f.unlink(missing_ok=True)

    def train(self, name: str, images: list[Image.Image], progress=lambda msg: None) -> CustomProduct:
        if len(images) < 10:
            raise ValueError("upload at least 10 normal images")
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(images))
        n_val = max(2, round(0.2 * len(images)))
        val = [images[i] for i in idx[:n_val]]
        train = [images[i] for i in idx[n_val:]]
        with self.lock:
            progress(f"building the memory bank from {len(train)} images")
            self.model.fit(_ImageList(train))
            progress(f"calibrating the threshold on {len(val)} held-out images")
            val_scores = self._scores(_ImageList(val))
            cp = CustomProduct(id="custom-" + uuid.uuid4().hex[:12], name=name.strip()[:60] or "Custom product",
                               created=time.time(), threshold=validation_threshold(np.array(val_scores)),
                               n_train=len(train), n_val=len(val), val_scores=val_scores,
                               state={"bank": self.model.bank.clone()})
            self.products[cp.id] = cp
            torch.save(cp, self.data_dir / f"{cp.id}.pt")
            while len(self.products) > self.MAX_PRODUCTS:
                old, _ = self.products.popitem(last=False)
                (self.data_dir / f"{old}.pt").unlink(missing_ok=True)
        return cp

    def _bundled_bank(self, pid: str, bank_file: Path) -> torch.Tensor:
        """Memory banks of bundled products (60-110 MB each), cached LRU."""
        with self.lock:
            if pid in self.banks:
                self.banks.move_to_end(pid)
            else:
                self.banks[pid] = torch.load(bank_file, map_location="cpu", weights_only=False)["bank"]
                while len(self.banks) > 4:
                    self.banks.popitem(last=False)
            return self.banks[pid]

    def _scores(self, ds: _ImageList) -> list[float]:
        return [float(self.model.predict(ds[i]["image"].unsqueeze(0))[0][0]) for i in range(len(ds))]

    def predict(self, pid: str, images: list[Image.Image], bank_file: Path | None = None) -> list[Prediction]:
        """``pid`` is a custom product id, or a bundled product id with its ``bank_file``."""
        bank = self._bundled_bank(pid, bank_file) if bank_file else self.products[pid].state["bank"]
        out = []
        with self.lock:
            self.model.bank = bank
            for im in images:
                view = im.convert("RGB").resize((256, 256), Image.BILINEAR)
                x = NORM(transforms.functional.to_tensor(view)).unsqueeze(0)
                t0 = time.perf_counter()
                s, m = self.model.predict(x)
                out.append(Prediction(float(s[0]), view, m[0].numpy(), (time.perf_counter() - t0) * 1000))
        return out
