"""CSAD (Hsieh & Lai, BMVC 2024) via its official code, inside the AnoDe protocol.

Official repository: https://github.com/Tokichan/CSAD (pinned commit in
``CSAD_COMMIT``). Inference uses the authors' exported single-file PyTorch
model (``CSAD_ONNX``) and their LOCO data loader (256 px). CSAD outputs an
image score only, with no anomaly map.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

CSAD_COMMIT = "d46cf856b99bf056088b5f510fd984e04813583a"


def import_csad(repo: Path, with_data: bool = True):
    """Import the official model (and, with ``with_data``, the LOCO loader,
    which needs the HuggingFace ``datasets`` package)."""
    repo = Path(repo).resolve()
    # GLASS ships a local ``datasets`` package that would shadow HuggingFace
    # ``datasets``, which CSAD imports. Drop it if it is cached.
    for name in [m for m in sys.modules if m == "datasets" or m.startswith("datasets.")]:
        del sys.modules[name]
    sys.path[:] = [p for p in sys.path if "GLASS" not in p]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.chdir(repo)
    from models.onnx_model import CSAD_ONNX
    if not with_data:
        return CSAD_ONNX, None
    from data_loader import MVTecLOCODataset
    return CSAD_ONNX, MVTecLOCODataset


def load_model(repo: Path, ckpt: Path, device):
    CSAD_ONNX, _ = import_csad(repo, with_data=False)
    sd = torch.load(ckpt, map_location="cpu")
    model = CSAD_ONNX(dim=512, num_classes=sd["segmentor.fc2.conv3.weight"].shape[0])
    model.load_state_dict(sd)
    return model.to(device).eval()


@torch.no_grad()
def score_split(model, repo: Path, root: Path, category: str, phase: str, device) -> tuple[dict, list[float]]:
    """Scores keyed by resolved image path for ``phase`` "test" or "eval"
    (the official name for the validation split), plus per-image latencies (ms)."""
    _, MVTecLOCODataset = import_csad(repo)
    ds = MVTecLOCODataset(root=str(root), image_size=256, phase=phase, category=category,
                          to_gpu=device.type == "cuda")
    scores, times = {}, []
    for i in range(len(ds)):
        s = ds[i]
        x = s["image"].unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        scores[str(Path(s["path"]).resolve())] = float(y.item())
    return scores, times
