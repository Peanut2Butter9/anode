"""Autoencoder on pretrained multi-level features (the report's "Res2Net" model).

The original report called this model "Res2Net (ResNet backbone multilevel)".
The original notebook actually did the following: take ImageNet ResNet-50
layer2 + layer3 features, aggregate each over a 3x3 neighbourhood, resize
them to layer2's resolution, and concatenate them (1536 channels). A
1x1-convolution autoencoder (latent size 100) is then trained to reconstruct
these feature maps. The anomaly map is the per-location feature
reconstruction error, and the image score is the mean of its 10 largest
values.

We run it with the ResNet-50 backbone, which is what the notebook used, and
with a Res2Net-50 backbone, which is what the report's name and citation
describe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .base import AnomalyModel, gaussian_blur
from .features import Backbone, patch_embedding


class FeatCAE(nn.Module):
    """1x1-conv autoencoder over feature maps (from the original notebook)."""

    def __init__(self, in_channels: int = 1536, latent_dim: int = 100):
        super().__init__()
        mid = (in_channels + 2 * latent_dim) // 2

        def block(a, b):
            return [nn.Conv2d(a, b, 1), nn.BatchNorm2d(b), nn.ReLU()]

        self.encoder = nn.Sequential(*block(in_channels, mid), *block(mid, 2 * latent_dim),
                                     nn.Conv2d(2 * latent_dim, latent_dim, 1))
        self.decoder = nn.Sequential(*block(latent_dim, 2 * latent_dim), *block(2 * latent_dim, mid),
                                     nn.Conv2d(mid, in_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class FeatureAE(AnomalyModel):
    name = "feature_ae"

    def __init__(self, backbone: str = "resnet50", layers=("layer2", "layer3"), latent_dim: int = 100,
                 epochs: int = 50, lr: float = 1e-3, batch_size: int = 16, top_k: int = 10,
                 pretrained: bool = True, **kw):
        super().__init__(**kw)
        self.layers = tuple(layers)
        self.epochs, self.lr, self.batch_size, self.top_k = epochs, lr, batch_size, top_k
        self.latent_dim = latent_dim
        self.backbone = Backbone(backbone, self.layers, pretrained).to(self.device)
        self.net: FeatCAE | None = None
        self.loss_history: list[float] = []

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        return patch_embedding(self.backbone(images.to(self.device)), self.layers)

    def fit(self, train: Dataset) -> None:
        # The backbone is frozen, so features are computed once and cached (fp16)
        loader = DataLoader(train, batch_size=self.batch_size, num_workers=2)
        feats = torch.cat([self._features(b["image"]).half() for b in loader])
        torch.manual_seed(self.seed)
        self.net = FeatCAE(feats.shape[1], self.latent_dim).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(feats), generator=g)
            total = 0.0
            for i in range(0, len(feats), self.batch_size):
                x = feats[perm[i:i + self.batch_size].to(feats.device)].float()
                loss = torch.mean((self.net(x) - x) ** 2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item() * len(x)
            self.loss_history.append(total / len(feats))
        self.net.eval()

    @torch.no_grad()
    def predict(self, images):
        f = self._features(images)
        err = ((self.net(f) - f) ** 2).mean(dim=1)  # (B, h, w)
        scores = err.flatten(1).topk(self.top_k, dim=1).values.mean(dim=1)
        maps = F.interpolate(err.unsqueeze(1), size=images.shape[-2:], mode="bilinear",
                             align_corners=False).squeeze(1)
        return scores, gaussian_blur(maps, sigma=4.0)

    def state_dict(self):
        return {"layers": self.layers, "net": self.net.state_dict()}


def feature_ae_resnet50(**kw) -> FeatureAE:
    m = FeatureAE(backbone="resnet50", **kw)
    m.name = "feature_ae_resnet50"
    return m


def feature_ae_res2net50(**kw) -> FeatureAE:
    m = FeatureAE(backbone="res2net50_26w_4s", **kw)
    m.name = "feature_ae_res2net50"
    return m
