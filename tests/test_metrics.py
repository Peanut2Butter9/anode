import numpy as np
import torch

from anode import metrics as M
from anode.models.features import greedy_coreset, knn


def _masks():
    masks = np.zeros((4, 32, 32), dtype=bool)
    masks[1, 4:10, 4:10] = True
    masks[2, 20:30, 2:6] = True
    masks[2, 1:3, 25:31] = True  # two regions in one image
    return masks


def test_aupro_perfect_and_inverted():
    masks = _masks()
    assert np.isclose(M.aupro(masks, masks.astype(float)), 1.0)
    assert M.aupro(masks, 1.0 - masks.astype(float)) < 0.01


def test_aupro_random_is_about_0_15():
    # With random scores PRO tracks FPR, so the normalised area up to 0.3 is
    # about 0.3 / 2 = 0.15. Many regions keep the estimate's noise small.
    rng = np.random.default_rng(0)
    masks = np.zeros((20, 64, 64), dtype=bool)
    for m in masks:
        for _ in range(5):
            y, x = rng.integers(0, 56, 2)
            m[y:y + 6, x:x + 6] = True
    assert abs(M.aupro(masks, rng.random(masks.shape)) - 0.15) < 0.01


def test_f1_max_and_threshold_metrics():
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    f1, thr = M.f1_max(labels, scores)
    assert f1 == 1.0 and 0.3 < thr <= 0.8
    r = M.threshold_metrics(labels, scores, 0.5)
    assert r["accuracy"] == 1.0 and r["fp"] == 0 and r["fn"] == 0


def test_validation_threshold_is_mean_plus_3_std():
    v = np.array([1.0, 2.0, 3.0])
    assert np.isclose(M.validation_threshold(v), 2.0 + 3 * 1.0)


def test_knn_matches_brute_force():
    g = torch.Generator().manual_seed(0)
    q, b = torch.randn(50, 8, generator=g), torch.randn(1000, 8, generator=g)
    d, i = knn(q, b, k=3, chunk=128)  # several chunks
    ref_d, ref_i = torch.cdist(q, b).topk(3, largest=False)
    assert torch.allclose(d, ref_d, atol=1e-5) and torch.equal(i, ref_i)


def test_greedy_coreset_covers_clusters():
    # Three tight clusters: a 3-point k-center coreset must pick one from each
    centers = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    pts = torch.cat([c + 0.01 * torch.randn(100, 2) for c in centers])
    idx = greedy_coreset(pts, ratio=3 / 300)
    assert len(set((idx // 100).tolist())) == 3
