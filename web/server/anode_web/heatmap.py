"""Render anomaly maps as image overlays (no matplotlib needed)."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

# Turbo colormap (Mikhailov, Google 2019), polynomial approximation
_TURBO_R = (0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943)
_TURBO_G = (0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604)
_TURBO_B = (0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973)


def turbo(x: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB uint8 with the Turbo colormap."""
    x = np.clip(x, 0, 1)[..., None]
    powers = np.concatenate([x**i for i in range(6)], axis=-1)
    rgb = np.stack([powers @ np.array(c) for c in (_TURBO_R, _TURBO_G, _TURBO_B)], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def overlay(view: Image.Image, amap: np.ndarray, threshold: float, alpha: float = 0.55) -> Image.Image:
    """Heatmap on top of the image the model saw.

    The map is divided by the decision threshold, so the color scale means the
    same thing for every product: 1.0 (yellow-red) is the threshold, and the
    scale saturates at 2x the threshold. Areas below 0.7x the threshold stay
    transparent, so normal parts look clean.
    """
    v = np.asarray(amap, dtype=np.float32) / max(threshold, 1e-12)
    m = Image.fromarray(turbo(v / 2)).resize(view.size, Image.BILINEAR)
    a = np.clip((v - 0.7) / 0.5, 0, 1) * alpha
    a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(view.size, Image.BILINEAR)) / 255.0
    base = np.asarray(view.convert("RGB"), dtype=np.float32)
    out = base * (1 - a[..., None]) + np.asarray(m, dtype=np.float32) * a[..., None]
    return Image.fromarray(out.astype(np.uint8))


def segmentation(view: Image.Image, amap: np.ndarray, threshold: float) -> Image.Image:
    """Regions above the threshold, filled in translucent red with a solid outline."""
    mask = np.asarray(Image.fromarray((np.asarray(amap) > threshold).astype(np.uint8) * 255)
                      .resize(view.size, Image.NEAREST)) > 0
    base = np.asarray(view.convert("RGB"), dtype=np.float32)
    red = np.array([230, 40, 40], dtype=np.float32)
    out = base.copy()
    out[mask] = base[mask] * 0.55 + red * 0.45
    # Outline: mask pixels with at least one non-mask 4-neighbour
    pad = np.pad(mask, 1)
    edge = mask & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    out[edge] = red
    return Image.fromarray(out.astype(np.uint8))


def data_url(im: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format=fmt, quality=88)
    mime = "jpeg" if fmt == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode()
