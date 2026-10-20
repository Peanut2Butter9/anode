"""Data preparation for the official GLASS code (no GLASS repository needed)."""

import numpy as np
from PIL import Image

from anode.data import load_split
from anode.external import glass as G
from tests.test_benchmark import mvtec_ad, mvtec_loco  # noqa: F401  (fixtures)


def test_numpy2_shim_restores_sctypes():
    G._numpy2_shim()
    assert np.float32 in np.sctypes["float"]


def test_make_view_mvtec_ad_links_and_masks(mvtec_ad, tmp_path):
    train, _, test = load_split("mvtec_ad", mvtec_ad, "bottle")
    view = G.make_view(train, test, "bottle", tmp_path / "view")
    c = view / "bottle"
    assert len(list((c / "train" / "good").iterdir())) == len(train)
    assert (c / "train" / "good" / train[0].image.name).is_symlink()
    imgs = sorted((c / "test" / "broken").iterdir())
    masks = sorted((c / "ground_truth" / "broken").iterdir())
    # The official loader pairs images and masks by sorted order
    assert [m.name for m in masks] == [f"{i.stem}_mask.png" for i in imgs]


def test_make_view_loco_squares_images(mvtec_loco, tmp_path):
    train, _, test = load_split("mvtec_loco", mvtec_loco, "pushpins")
    view = G.make_view(train, test, "pushpins", tmp_path / "view", square=32)
    img = next((view / "pushpins" / "test" / "logical_anomalies").iterdir())
    mask = next((view / "pushpins" / "ground_truth" / "logical_anomalies").iterdir())
    assert Image.open(img).size == (32, 32) and Image.open(mask).size == (32, 32)
    assert set(np.unique(np.asarray(Image.open(mask)))) <= {0, 255}


def test_glass_test_dataset_geometry(mvtec_ad):
    _, _, test = load_split("mvtec_ad", mvtec_ad, "bottle")
    i = next(i for i, s in enumerate(test) if s.label)  # an anomalous image
    item = G.GlassTestDataset(test, "bottle")[i]
    assert item["image"].shape == (3, G.IMAGESIZE, G.IMAGESIZE)
    assert item["mask"].shape == (G.IMAGESIZE, G.IMAGESIZE) and item["mask"].any()
    assert G.resize_for("wood") == 329 and G.resize_for("bottle") == 288
