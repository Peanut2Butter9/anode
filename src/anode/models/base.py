"""Common interface for all anomaly detection models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class AnomalyModel(ABC):
    """Each model is fitted on normal images only. ``predict`` returns an image
    score (higher = more anomalous) and, if the model localises, an anomaly map
    at the input resolution."""

    name: str = "base"
    normalize_input: bool = True  # ImageNet normalisation (False = images in [0, 1])
    localizes: bool = True

    def __init__(self, device: str | torch.device = "cuda", seed: int = 0):
        self.device = torch.device(device)
        self.seed = seed

    @abstractmethod
    def fit(self, train: Dataset) -> None: ...

    @abstractmethod
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``images`` (B, 3, H, W) -> scores (B,), maps (B, H, W) or None."""

    def state_dict(self) -> dict:
        """Everything needed to rebuild the fitted model for inference."""
        raise NotImplementedError


def gaussian_blur(maps: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """Smooth (B, H, W) anomaly maps with a Gaussian kernel, as in PatchCore."""
    radius = int(4 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=maps.device, dtype=maps.dtype)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    k = k / k.sum()
    m = maps.unsqueeze(1)
    m = F.conv2d(F.pad(m, (radius, radius, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    m = F.conv2d(F.pad(m, (0, 0, radius, radius), mode="reflect"), k.view(1, 1, -1, 1))
    return m.squeeze(1)
