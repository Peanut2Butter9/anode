"""Pretrained backbones and nearest-neighbour search."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone(nn.Module):
    """Frozen ImageNet backbone that returns intermediate feature maps.

    ``layers`` uses ResNet naming ("layer1" ... "layer4", "pool"). "pool" is
    the global-average-pooled output of layer4.
    """

    def __init__(self, name: str, layers: tuple[str, ...], pretrained: bool = True):
        super().__init__()
        self.name = name
        self.layers = layers
        if name in ("resnet50", "wide_resnet50_2"):
            import torchvision.models as tvm

            # ResNet-50: torchvision's DEFAULT (V2) weights, as in the original
            # notebooks. WideResNet-50: the V1 weights used by the PatchCore paper.
            weights = {"resnet50": tvm.ResNet50_Weights.IMAGENET1K_V2,
                       "wide_resnet50_2": tvm.Wide_ResNet50_2_Weights.IMAGENET1K_V1}[name]
            self.net = getattr(tvm, name)(weights=weights if pretrained else None)
        elif name.startswith("res2net"):
            import timm

            self.net = timm.create_model(name, pretrained=pretrained)
        else:
            raise ValueError(f"unsupported backbone {name!r}")
        self.net.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        n = self.net
        act = n.act1 if hasattr(n, "act1") else n.relu  # timm vs torchvision naming
        x = n.maxpool(act(n.bn1(n.conv1(x))))
        out = {}
        # Stop at the deepest layer needed
        last = 4 if "pool" in self.layers else max(int(l[-1]) for l in self.layers)
        for i in range(1, last + 1):
            x = getattr(n, f"layer{i}")(x)
            if f"layer{i}" in self.layers:
                out[f"layer{i}"] = x
        if "pool" in self.layers:
            out["pool"] = x.mean(dim=(2, 3))
        return out


def patch_embedding(feats: dict[str, torch.Tensor], layers: tuple[str, ...]) -> torch.Tensor:
    """PatchCore-style locally aware patch features.

    Each feature map gets 3x3 average pooling (local neighbourhood aggregation),
    deeper maps are upsampled to the first map's resolution, and the maps are
    concatenated along channels. Returns (B, C, h, w).
    """
    maps = [F.avg_pool2d(feats[l], 3, stride=1, padding=1) for l in layers]
    h, w = maps[0].shape[-2:]
    maps = [m if m.shape[-2:] == (h, w) else F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
            for m in maps]
    return torch.cat(maps, dim=1)


@torch.no_grad()
def knn(query: torch.Tensor, bank: torch.Tensor, k: int = 1, chunk: int = 32768) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact k nearest neighbours (Euclidean) of each query row in ``bank``.

    The bank is processed in chunks so the full distance matrix never has to
    fit in memory. Returns (distances, indices), each (n_query, k).
    """
    best_d = torch.full((query.shape[0], k), float("inf"), device=query.device)
    best_i = torch.zeros((query.shape[0], k), dtype=torch.long, device=query.device)
    q = query.float()
    for start in range(0, bank.shape[0], chunk):
        b = bank[start:start + chunk].float()
        d = torch.cdist(q, b)
        kk = min(k, b.shape[0])
        d, i = d.topk(kk, dim=1, largest=False)
        d = torch.cat([best_d, d], dim=1)
        i = torch.cat([best_i, i + start], dim=1)
        best_d, sel = d.topk(k, dim=1, largest=False)
        best_i = i.gather(1, sel)
    return best_d, best_i


@torch.no_grad()
def greedy_coreset(features: torch.Tensor, ratio: float, proj_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """Indices of a greedy (k-center) coreset, as in PatchCore.

    Distances are computed on a random Gaussian projection to ``proj_dim``
    dimensions, which keeps them approximately (Johnson-Lindenstrauss) while
    making selection much faster.
    """
    n = features.shape[0]
    m = max(1, int(n * ratio))
    g = torch.Generator(device="cpu").manual_seed(seed)
    if features.shape[1] > proj_dim:
        proj = torch.randn(features.shape[1], proj_dim, generator=g) / proj_dim**0.5
        z = features.float() @ proj.to(features.device)
    else:
        z = features.float()
    start = int(torch.randint(n, (1,), generator=g))
    selected = torch.empty(m, dtype=torch.long, device=features.device)
    selected[0] = start
    min_d = torch.linalg.norm(z - z[start], dim=1)
    for j in range(1, m):
        idx = torch.argmax(min_d)
        selected[j] = idx
        min_d = torch.minimum(min_d, torch.linalg.norm(z - z[idx], dim=1))
    return selected
