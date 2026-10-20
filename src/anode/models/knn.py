"""Feature-embedding models scored by nearest-neighbour distance.

- ``ResNetKNN``: one global ResNet-50 embedding per image (image-level only),
  configured as in the original AnoDe notebook.
- ``PatchKNN``: a memory bank of locally aware patch features. With a
  WideResNet-50 backbone, a 10% greedy coreset and neighbourhood re-weighting
  it is PatchCore (Roth et al., CVPR 2022). The original notebook used a
  simplified variant (ResNet-50, a random 10% subsample, no re-weighting).
  Here we implement the published method, which is what the report cites.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .base import AnomalyModel, gaussian_blur
from .features import Backbone, greedy_coreset, knn, patch_embedding


class ResNetKNN(AnomalyModel):
    """Global-average-pooled ResNet-50 features (2048-d). As in the original
    notebook, only the ``n_features`` dimensions with the largest standard
    deviation over the normal training images are kept. The score is the mean
    distance to the ``k`` nearest training embeddings."""

    name = "resnet50_knn"
    localizes = False

    def __init__(self, k: int = 50, n_features: int = 500, batch_size: int = 32, pretrained: bool = True, **kw):
        super().__init__(**kw)
        self.k, self.n_features, self.batch_size = k, n_features, batch_size
        self.backbone = Backbone("resnet50", ("pool",), pretrained).to(self.device)
        self.bank: torch.Tensor | None = None
        self.selected: torch.Tensor | None = None

    def fit(self, train: Dataset) -> None:
        loader = DataLoader(train, batch_size=self.batch_size, num_workers=2)
        bank = torch.cat([self.backbone(b["image"].to(self.device))["pool"] for b in loader])
        self.selected = bank.std(dim=0).argsort(descending=True)[:self.n_features]
        self.bank = bank[:, self.selected]

    @torch.no_grad()
    def predict(self, images):
        z = self.backbone(images.to(self.device))["pool"][:, self.selected]
        d, _ = knn(z, self.bank, k=min(self.k, len(self.bank)))
        return d.mean(dim=1), None

    def state_dict(self):
        return {"k": self.k, "selected": self.selected.cpu(), "bank": self.bank.half().cpu()}


class PatchKNN(AnomalyModel):
    """Memory bank of patch features with nearest-neighbour scoring.

    Image score = the largest patch score. Anomaly map = patch scores, upsampled
    to the input size and smoothed with a Gaussian (sigma = 4).
    """

    name = "patch_knn"

    def __init__(self, backbone: str = "wide_resnet50_2", layers=("layer2", "layer3"),
                 coreset_ratio: float | None = 0.1, reweight_neighbors: int | None = 9,
                 batch_size: int = 16, pretrained: bool = True, **kw):
        super().__init__(**kw)
        self.layers = tuple(layers)
        self.coreset_ratio = coreset_ratio
        self.reweight_neighbors = reweight_neighbors
        self.batch_size = batch_size
        self.num_workers = 2
        self.backbone = Backbone(backbone, self.layers, pretrained).to(self.device)
        self.bank: torch.Tensor | None = None

    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, h, w, C)"""
        return patch_embedding(self.backbone(images.to(self.device)), self.layers).permute(0, 2, 3, 1)

    def fit(self, train: Dataset) -> None:
        loader = DataLoader(train, batch_size=self.batch_size, num_workers=self.num_workers)
        feats = []
        for b in loader:
            z = self._embed(b["image"])
            feats.append(z.reshape(-1, z.shape[-1]).half())  # fp16 halves memory
        feats = torch.cat(feats)
        if self.coreset_ratio is not None and self.coreset_ratio < 1:
            feats = feats[greedy_coreset(feats, self.coreset_ratio, seed=self.seed)]
        self.bank = feats  # distances are computed in fp32

    @torch.no_grad()
    def predict(self, images):
        z = self._embed(images)  # (B, h, w, C)
        b, h, w, c = z.shape
        flat = z.reshape(-1, c)
        d, idx = knn(flat, self.bank, k=1)
        patch_scores = d[:, 0].reshape(b, h * w)

        scores = patch_scores.max(dim=1).values
        if self.reweight_neighbors:
            scores = self._reweight(flat.reshape(b, h * w, c), patch_scores, idx[:, 0].reshape(b, h * w))

        maps = F.interpolate(patch_scores.reshape(b, 1, h, w), size=images.shape[-2:], mode="bilinear",
                             align_corners=False).squeeze(1)
        return scores, gaussian_blur(maps, sigma=4.0)

    def _reweight(self, feats, patch_scores, nn_idx):
        """PatchCore Eq. 7: down-weight the image score when the matched memory
        patch m* lies in a dense region of the bank (so the test patch is less
        surprising)."""
        best = patch_scores.argmax(dim=1)
        rows = torch.arange(feats.shape[0], device=feats.device)
        test_patch = feats[rows, best]  # (B, C)
        m_star = self.bank[nn_idx[rows, best]].float()  # (B, C)
        _, support = knn(m_star, self.bank, k=self.reweight_neighbors)  # neighbours of m*
        dists = torch.linalg.norm(test_patch.unsqueeze(1) - self.bank[support].float(), dim=2)  # (B, k)
        weight = 1 - torch.softmax(dists, dim=1)[:, 0]
        return weight * patch_scores[rows, best]

    def state_dict(self):
        return {"layers": self.layers, "reweight_neighbors": self.reweight_neighbors, "bank": self.bank.cpu()}


def patchcore(**kw) -> PatchKNN:
    m = PatchKNN(backbone="wide_resnet50_2", layers=("layer2", "layer3"), coreset_ratio=0.1,
                 reweight_neighbors=9, **kw)
    m.name = "patchcore"
    return m

