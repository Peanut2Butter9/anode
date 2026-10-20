"""End-to-end runs on a tiny synthetic dataset laid out like MVTec AD / LOCO.

Backbones use random weights (no download) and training is cut to a few
epochs, so this only checks that the pipeline runs and writes sane outputs.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from anode.benchmark import run_one
from anode.data import load_split

SIZE = 64


def _img(path, defect=False, rng=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    a = (128 + 10 * rng.standard_normal((48, 48, 3))).clip(0, 255)
    if defect:
        a[10:20, 10:20] = 255
    Image.fromarray(a.astype(np.uint8)).save(path)


def _mask(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.zeros((48, 48), dtype=np.uint8)
    m[10:20, 10:20] = 255
    Image.fromarray(m).save(path)


@pytest.fixture
def mvtec_ad(tmp_path):
    rng = np.random.default_rng(0)
    c = tmp_path / "bottle"
    for i in range(20):
        _img(c / "train" / "good" / f"{i:03}.png", rng=rng)
    for i in range(6):
        _img(c / "test" / "good" / f"{i:03}.png", rng=rng)
        _img(c / "test" / "broken" / f"{i:03}.png", defect=True, rng=rng)
        _mask(c / "ground_truth" / "broken" / f"{i:03}_mask.png")
    return tmp_path


@pytest.fixture
def mvtec_loco(tmp_path):
    rng = np.random.default_rng(1)
    c = tmp_path / "pushpins"
    for split, n in (("train", 16), ("validation", 4)):
        for i in range(n):
            _img(c / split / "good" / f"{i:03}.png", rng=rng)
    for i in range(4):
        _img(c / "test" / "good" / f"{i:03}.png", rng=rng)
        for kind in ("logical_anomalies", "structural_anomalies"):
            _img(c / "test" / kind / f"{i:03}.png", defect=True, rng=rng)
            _mask(c / "ground_truth" / kind / f"{i:03}" / "000.png")
    return tmp_path


FAST = {
    "autoencoder_l2": {"epochs": 2},
    "resnet50_knn": {"pretrained": False},
    "feature_ae_resnet50": {"pretrained": False, "epochs": 2},
    "feature_ae_res2net50": {"pretrained": False, "epochs": 2},
    "patchcore": {"pretrained": False},
}


def test_mvtec_ad_split(mvtec_ad):
    train, val, test = load_split("mvtec_ad", mvtec_ad, "bottle", val_fraction=0.1)
    assert (len(train), len(val), len(test)) == (18, 2, 12)
    assert not {s.image for s in train} & {s.image for s in val}
    assert all(s.masks and s.masks[0].exists() for s in test if s.label)


@pytest.mark.parametrize("model", sorted(FAST))
def test_run_mvtec_ad(mvtec_ad, tmp_path, model):
    out = tmp_path / "results"
    r = run_one("mvtec_ad", mvtec_ad, "bottle", model, 0, out, torch.device("cpu"), size=SIZE,
                model_kwargs=FAST[model])
    run_dir = out / "mvtec_ad" / model / "bottle" / "seed0"
    assert 0 <= r["image_auroc"] <= 1 and r["n_test"] == 12
    assert json.loads((run_dir / "metrics.json").read_text())["model"] == model
    if model == "resnet50_knn":
        assert "pixel_auroc" not in r
    else:
        assert 0 <= r["pixel_auroc"] <= 1 and 0 <= r["aupro_30"] <= 1
        assert np.load(run_dir / "maps.npz")["maps"].shape == (12, SIZE, SIZE)
    # A second call reuses the saved result
    assert run_one("mvtec_ad", mvtec_ad, "bottle", model, 0, out, torch.device("cpu"), size=SIZE,
                   model_kwargs=FAST[model]) == r


def test_run_mvtec_loco(mvtec_loco, tmp_path):
    r = run_one("mvtec_loco", mvtec_loco, "pushpins", "patchcore", 0, tmp_path / "results", torch.device("cpu"),
                size=SIZE, model_kwargs=FAST["patchcore"])
    assert r["n_val"] == 4 and r["n_test"] == 12
    assert set(r["per_defect_auroc"]) == {"logical_anomalies", "structural_anomalies"}
    # The bright square is easy to spot even with random features
    assert r["image_auroc_structural"] > 0.9
