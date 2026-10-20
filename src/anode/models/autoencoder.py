"""Convolutional autoencoder trained with an L2 (MSE) reconstruction loss.

The architecture is the one used in the original AnoDe notebooks: three
conv + average-pool stages and three transposed convolutions, with no dense
bottleneck. The original was warm-started from a checkpoint trained on another
category. Here every category is trained from scratch, so the result depends
only on that category's normal images.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .base import AnomalyModel, gaussian_blur


class ConvAutoencoder(nn.Module):
    """Works for 224x224 and 256x256 inputs (the output size equals the input size)."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4), nn.ReLU(), nn.AvgPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=4), nn.ReLU(), nn.AvgPool2d(2, 2),
            nn.Conv2d(256, 256, kernel_size=3), nn.ReLU(), nn.AvgPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=5, stride=2, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class Autoencoder(AnomalyModel):
    """Anomaly map = per-pixel squared reconstruction error (channel mean),
    smoothed with a Gaussian. Image score = mean squared error over the whole
    image, the same quantity as the training loss (as in the original)."""

    name = "autoencoder_l2"
    normalize_input = False  # pixels in [0, 1], to match the sigmoid output

    def __init__(self, epochs: int = 100, lr: float = 1e-3, batch_size: int = 16, **kw):
        super().__init__(**kw)
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        torch.manual_seed(self.seed)
        self.net = ConvAutoencoder().to(self.device)
        self.loss_history: list[float] = []

    def fit(self, train: Dataset) -> None:
        # Decode and resize every training image once, then train from memory
        images = torch.cat([b["image"] for b in DataLoader(train, batch_size=32, num_workers=2)])
        g = torch.Generator().manual_seed(self.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        # Mixed precision on CUDA: about 3-4x faster on T4 tensor cores
        amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        self.net.train()
        for _ in range(self.epochs):
            total, n = 0.0, 0
            perm = torch.randperm(len(images), generator=g)
            for i in range(0, len(images), self.batch_size):
                x = images[perm[i:i + self.batch_size]].to(self.device, non_blocking=True)
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=amp):
                    loss = F.mse_loss(self.net(x), x)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total += loss.item() * len(x)
                n += len(x)
            self.loss_history.append(total / n)
        self.net.eval()

    @torch.no_grad()
    def predict(self, images):
        x = images.to(self.device)
        err = ((self.net(x) - x) ** 2).mean(dim=1)  # (B, H, W)
        return err.flatten(1).mean(dim=1), gaussian_blur(err, sigma=4.0)

    def state_dict(self):
        return {"net": self.net.state_dict()}
